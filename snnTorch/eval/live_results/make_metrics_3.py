import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

def compute_dataset_metrics(info, min_channels=3):
    ripple_hits = info["ripple_hits"]
    fp_timing_to_channels = info["fp_timing_to_channels"]

    # TP: events with ≥min_channels detected
    TP = sum(1 for hits in ripple_hits.values() if len(hits) >= min_channels)
    FN = sum(1 for hits in ripple_hits.values() if len(hits) < min_channels)

    # FP: false positive events with ≥min_channels detections
    FP = sum(1 for chans in fp_timing_to_channels.values() if len(chans) >= min_channels)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "Precision": precision, "Recall": recall, "F1": f1
    }

def compute_channel_wise_kappa_with_fp(info, num_channels=8, min_channels=3):
    ripple_hits = info["ripple_hits"]
    fp_spikes_by_channel = info["fp_spikes_by_channel"]

    # Number of ripples (positive samples)
    # n_pos = len(ripple_hits)

    # Number of negatives - max number of FP spikes across all channels (to balance samples)
    n_neg = sum(1 for v in fp_spikes_by_channel.values() if len(v) >= min_channels) if fp_spikes_by_channel else 0

    gt_vector = []
    pred_vector = []

    # Positive samples
    for ripple_id in sorted(ripple_hits.keys()):
        gt_vector.append(1)
        pred_vector.append(1 if len(ripple_hits[ripple_id]) >= min_channels else 0)

    # Negative samples
    for i in range(n_neg):
        gt_vector.append(0)
        pred_vector.append(1) # These are FPs, so Ground Truth is 0, Prediction is 1 (detected)

    # Calculate Cohen's kappa if there is variance
    if len(set(gt_vector)) > 1 and len(set(pred_vector)) > 1:
        kappa = cohen_kappa_score(gt_vector, pred_vector)
    else:
        kappa = np.nan

    return kappa

def process_all_results(all_results, min_channels=3):
    all_data = []
    
    # The structure is { adapt: { network: { dataset: info } } }
    
    for adapt, networks_data in all_results.items():
        print(f"Processing adapt: {adapt}")
        if not isinstance(networks_data, dict):
            continue
            
        for network_name, datasets in networks_data.items():
            for dataset_name, info in datasets.items():
                # Metrics
                metrics = compute_dataset_metrics(info, min_channels=min_channels)
                entry = {
                    "Session": dataset_name,
                    "Network": network_name,
                    "ADAPT": adapt,
                    "Kappa": compute_channel_wise_kappa_with_fp(info, min_channels=min_channels),
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
    results_dir = os.path.dirname(os.path.abspath(__file__))
    
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
        metrics_df = process_all_results(all_results, min_channels=3)

        # --- SAVE TO CSV ---
        metrics_dir = os.path.join(results_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        output_csv = os.path.join(metrics_dir, "parallel_metrics_all_adapts.csv")
        metrics_df.to_csv(output_csv, index=False)
        print(f"Metrics saved to {output_csv}")
        
        # Print summary
        if not metrics_df.empty:
            print("\n--- Summary by Network and Adapt (Parallel) ---")
            summary = metrics_df.groupby(["Network", "ADAPT"])[["F1", "Precision", "Recall"]].mean()
            print(summary)