from process_results import *
import pickle
import os
import pandas as pd
import numpy as np

# Configuration
adapts = [0,2, 20, 120]
test_ds_b4 = False
networks = [
    "dsb4updn_median_200_15f",
    "dsb4updn_median_200_11b",
    "dsb4updn_median_200_13f",
    "dsb4updn_median_200_9f",
    "dsb4updn_median_200_12b",
    "updnb4ds_100_13b",
    "updnb4ds_100_7",
    "updnb4ds_100_13f",
    "adapt20_3b",
    "dsb4updn_median_200_11f",
    "dsb4updn_median_200_14b",
    "dsb4updn_median_200_16b"
]
# Parameters
tolerance = 20
padding = 100
refractory_period = 0
max_detection_offset = 100
jitter = 100

# Paths
parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_results")
os.makedirs(results_dir, exist_ok=True)

all_results = {}

for adapt in adapts:
    print(f"Processing adapt={adapt}")
    
    if test_ds_b4:
        identifier = f"testing_dsb4_adaptable{adapt}" if adapt > 0 else "testing_dsb4"
    else:
        identifier = f"30000_1000_100_adaptable{adapt}" if adapt > 0 else "30000_1000_100"
    
    dataset_path = os.path.join(parent_dir, "extract_Nripples", "train_pedro", "dataset_up_down", identifier)
    
    spikes_names = [f"{net}_adapt{adapt}" for net in networks if adapt > 0] if adapt > 0 else networks
    spikes_names = [f"{net}_dsb4" for net in networks] if test_ds_b4 else spikes_names
    
    list_nets = [
        {
            "name": net,
            "dataset_path": dataset_path,
            "spikes": os.path.join(os.path.dirname(os.path.abspath(__file__)), "spikes", f"{spike}_spikes.npy")
        }
        for net, spike in zip(networks, spikes_names)
    ]
    all_results[adapt]={}
    
    for net in list_nets:
        print(f"  Processing network: {net['name']}")
        try:
            info = process_network_results(
                net["dataset_path"],
                net["spikes"],
                tolerance=tolerance,
                padding=padding,
                refractory_period=refractory_period,
                max_detection_offset=max_detection_offset,
                jitter=jitter,
            )
            all_results[adapt][net["name"]] = info
              
        except Exception as e:
            print(f"    Error processing {net['name']}: {e}")

# Save intermediate pickle for this adapt
filename_pkl = f"all_network_results_tol{tolerance}_ma{max_detection_offset}_jit{jitter}_pa{padding}.pkl"
with open(os.path.join(results_dir, filename_pkl), "wb") as f:
    pickle.dump(all_results, f)
print(f"  Saved pickle to {filename_pkl}")
