import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_channel_metrics(info, num_channels=8):
    ripple_hits = info["ripple_hits"]
    fp_timing_to_channels = info["fp_timing_to_channels"]

    channel_metrics = []

    for ch in range(num_channels):
        TP = sum(1 for hits in ripple_hits.values() if ch in hits)
        FN = sum(1 for hits in ripple_hits.values() if ch not in hits)
        FP = sum(1 for chans in fp_timing_to_channels.values() if ch in chans)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        channel_metrics.append({
            "TP": TP, "FP": FP, "FN": FN,
            "Precision": precision, "Recall": recall, "F1": f1
        })

    return channel_metrics

def process_all_networks(metrics_dict_by_network,):
    all_data = []
    for network_name, datasets in metrics_dict_by_network.items():
        for dataset_name, info in datasets.items():
            channel_metrics = compute_channel_metrics(info,)
            for ch_id, metrics in enumerate(channel_metrics):
                entry = {
                    "Network": network_name,
                    "Dataset": dataset_name,
                    "Channel": ch_id,
                    **metrics
                }
                all_data.append(entry)
    return pd.DataFrame(all_data)

# --- LOAD THE .pkl FILE GENERATED PREVIOUSLY ---
adapt=0
test_ds_b4=False
if test_ds_b4:
    identifier=f"testing_dsb4_adaptable{adapt}" if adapt > 0 else "testing_dsb4"
else:
    identifier=f"30000_1000_100_adaptable{adapt}" if adapt > 0 else "30000_1000_100"
    # identifier="1000_200_median"
    
filename=f"all_network_results_{identifier}.pkl"
path_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

with open(path_file, "rb") as f:
    all_networks_info = pickle.load(f)

# --- PROCESS METRICS FOR EACH NETWORK ---
metrics_df = process_all_networks(all_networks_info)

# --- SAVE TO CSV FOR FUTURE USE ---
metrics_dir= os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")
os.makedirs(metrics_dir, exist_ok=True)
filename= f"channel_metrics_by_network_{identifier}.csv"
metrics_df.to_csv(os.path.join(metrics_dir, filename), index=False)