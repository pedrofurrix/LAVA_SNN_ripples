import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

def compute_channel_metrics(info, num_channels=8):
    ripple_hits = info["ripple_hits"]
    fp_timing_to_channels = info["fp_timing_to_channels"]
    filtered_spikes = info.get("filtered_spikes", [[]]*num_channels)
    
    total_ripples = len(ripple_hits)
    
    channel_metrics = []

    for ch in range(num_channels):
        # TP: Count ripples detected by this channel
        TP = sum(1 for hits in ripple_hits.values() if ch in hits)
        
        # FN: Total ripples - TP
        FN = total_ripples - TP
        
        # FP: Count grouped FP events for this channel
        # fp_timing_to_channels maps a time window to a set of channels that had an FP in that window
        FP = sum(1 for chans in fp_timing_to_channels.values() if ch in chans)
        
        # Num Spikes
        num_spikes = len(filtered_spikes[ch]) if ch < len(filtered_spikes) else 0

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        channel_metrics.append({
            "TP": TP, 
            "FP": FP, 
            "FN": FN,
            "Precision": precision, 
            "Recall": recall, 
            "F1": f1,
            "Num_Spikes": num_spikes,
            "Num_Ripples": total_ripples,
            "Mean_Latency": np.nan # Cannot compute accurately from current info structure without re-processing
        })

    return channel_metrics



def compute_channel_wise_kappa_with_fp(info, num_channels=8):
    ripple_hits = info["ripple_hits"]
    undetected_ripples = info["undetected_ripples"]
    fp_spikes_by_channel = info["fp_spikes_by_channel"]

    kappa_per_channel = {}

    # Number of ripples (positive samples)
    n_pos = len(ripple_hits) + len(undetected_ripples)

    # Number of negatives - max number of FP spikes across all channels (to balance samples)
    n_neg = max(len(v) for v in fp_spikes_by_channel.values()) if fp_spikes_by_channel else 0

    for ch in range(num_channels):
        gt_vector = []
        pred_vector = []

        # Positive samples
        for ripple_id in sorted(ripple_hits.keys()):
            gt_vector.append(1)
            pred_vector.append(1 if ch in ripple_hits[ripple_id] else 0)

        # Negative samples
        fp_spikes = fp_spikes_by_channel.get(ch, [])
        for i in range(n_neg):
            gt_vector.append(0)
            pred_vector.append(1 if i < len(fp_spikes) else 0)

        # Calculate Cohen's kappa if there is variance
        if len(set(gt_vector)) > 1 and len(set(pred_vector)) > 1:
            kappa = cohen_kappa_score(gt_vector, pred_vector)
        else:
            kappa = np.nan

        kappa_per_channel[ch] = kappa

    return kappa_per_channel

def process_all_results(all_results):
    all_data = []
    # The structure from results_processor.py is { adapt: { network: { dataset: info } } }
    
    for adapt, networks_data in all_results.items():
        print(f"Processing adapt: {adapt}")
        if not isinstance(networks_data, dict):
            print(f"  Skipping unexpected data type for adapt {adapt}")
            continue
            
        for network_name, datasets in networks_data.items():
            for dataset_name, info in datasets.items():
                channel_metrics = compute_channel_metrics(info)
                channel_kappa=compute_channel_wise_kappa_with_fp(info)

                for ch_id, metrics in enumerate(channel_metrics):
                    entry = {
                        "Session": dataset_name,
                        "Channel": ch_id,
                        "Network": network_name,
                        "ADAPT": adapt,
                        "Kappa": channel_kappa.get(ch_id, np.nan),
                        **metrics
                    }
                    all_data.append(entry)
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    # Configuration matching results_processor.py
    tolerance = 20
    padding = 100
    max_detection_offset = 100
    jitter = 100
    
    # Path to the pickle file
    # This script is in snnTorch/eval/live_results/
    results_dir = os.path.dirname(os.path.abspath(__file__),"metrics")
    
    # Filename pattern from results_processor.py
    filename = f"all_network_results_tol{tolerance}_ma{max_detection_offset}_jit{jitter}_pa{padding}.pkl"
    path_file = os.path.join(results_dir, filename)
    
    print(f"Loading results from: {path_file}")
    
    if not os.path.exists(path_file):
        print(f"File not found: {path_file}")
        print("Please run snnTorch/eval/results_processor.py first to generate the pickle file.")
    else:
        with open(path_file, "rb") as f:
            all_results = pickle.load(f)

        # --- PROCESS METRICS ---
        metrics_df = process_all_results(all_results)

        # --- SAVE TO CSV ---
        output_csv = os.path.join(results_dir, "all_networks_metrics_from_processor.csv")
        metrics_df.to_csv(output_csv, index=False)
        print(f"Metrics saved to {output_csv}")
        
        # Print summary
        if not metrics_df.empty:
            print("\n--- Summary by Network and Adapt ---")
            summary = metrics_df.groupby(["Network", "ADAPT"])[["F1", "Precision", "Recall"]].mean()
            print(summary)
        else:
            print("No metrics calculated.")