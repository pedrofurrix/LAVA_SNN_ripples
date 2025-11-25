import os
import numpy as np
import torch
from collections import deque
from torch.utils.data import TensorDataset, DataLoader

import sys
curr_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir,os.pardir))
sys.path.append(parent_dir)  # Add parent directory to path for importing Net

from snnTorch.utils.start_net import Net
from liset_tk.read_data import read_data
from lists_sessions import *
from process_signal import load_experimental_data, spikify_signal

def run_inference(prefix, data_path, sessions,channel_sessions=None, adapt=0, test_dsb4=True, export_spikes=True, seed=None):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    curr_dir = os.getcwd()
    
    # Constants
    RIPPLE_DETECTION_OFFSET = [18, 45, 31, 20]
    PRED_CAUSALITY_WINDOW = 5
    TOLERANCE = 0
    MAX_DETECTION_OFFSET = int(RIPPLE_DETECTION_OFFSET[2] + RIPPLE_DETECTION_OFFSET[1] + PRED_CAUSALITY_WINDOW + TOLERANCE)
    dt = 1
    refrac_period = 200

    # Load network
    net = Net().to(device)

    net_path = f"out/{prefix}_trained_net_loss.pth"
    net.load_state_dict(torch.load(net_path, map_location=device))
    net.eval()

    print(f"Loading from: {net_path}")
    print(f"Dataset path: {dataset_path}")

    # Seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    datasets = os.listdir(dataset_path)
    datasets = [d for d in datasets if d != "config.json"]

    all_metrics = {}
    out_spikes = []

    for session in sessions:
        channel_session=channel_sessions[session]
        filtered_signal, ripples=load_experimental_data(data_path,session, channel = channel_session, load_data = True, downsample = False, normalize = True,offset = 0.16,)
        spk,threshold= spikify_signal(filtered_signal,
        fs=30000,
        time_max=20.0, # seconds
        overlap=0.5, # 50% overlap
        adapt_threshold=True, #  
        percentile=False,
        window_size=0.01,
        sample_ratio=0.2,
        scaling_factor=1.0,
        refractory=0,
        factor=1,
        initial_value=None,
        verbose=False,
        ripples=None,  
        # Warm-up call

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        data_path = os.path.join(dataset_path, dataset)
        data = np.load(os.path.join(data_path, "spike_data.npy"))
        ripples = np.load(os.path.join(data_path, "ripples.npy"))
        ripples_start = ripples[:, 0] - TOLERANCE

        input_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        gt_tensor = torch.tensor(ripples_start, dtype=torch.float32).to(device)
        all_metrics[dataset] = {}

        for channel in range(data.shape[1]):
            print(f"  Channel {channel} | HFOs: {len(ripples_start)}")
            curr_gt_idx = 0
            active_gts = deque()
            TP = FP = FN = TN = 0
            total_steps = input_tensor.shape[0]
            output_spikes_channel = []
            lif_out_refrac_times = torch.full(size=(1,), fill_value=-10000.0, device=device)

            net.reset_state()
            with torch.no_grad():
                for step in range(total_steps):
                    curr_input = input_tensor[step, channel, :].unsqueeze(0)

                    # Add GT if it's time
                    if curr_gt_idx < len(gt_tensor):
                        curr_gt = int(gt_tensor[curr_gt_idx].item())
                        if curr_gt == step:
                            active_gts.append((curr_gt, MAX_DETECTION_OFFSET))
                            curr_gt_idx += 1

                    # State update
                    spk, mem, syn = net(curr_input)
                    _, _, spk_out = spk

                    # Update active GTs
                    new_gts = deque()
                    for gt_time, ttl in active_gts:
                        if ttl < 0:
                            FN += 1
                        else:
                            new_gts.append((gt_time, ttl - 1))
                    active_gts = new_gts

                    # Output spikes
                    if torch.sum(spk_out) > 0:
                        spk_out_int = spk_out.squeeze(0).int()
                        output_spikes_channel.append(step * dt)
                        refrac_mask = lif_out_refrac_times > step * dt
                        valid_spk = torch.Tensor.bool(spk_out_int & (~refrac_mask))

                        if torch.sum(valid_spk) > 0:
                            lif_out_refrac_times[valid_spk] = float(step * dt + refrac_period)
                            if active_gts:
                                active_gts.popleft()
                                TP += 1
                            else:
                                FP += 1
                        else:
                            TN += 1
                    else:
                        TN += 1

                    if step % 100000 == 0 and step > 0:
                        print(f"    {step}/{total_steps} ({(step/total_steps)*100:.2f}%)")

            # Metrics
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            all_metrics[dataset][f"channel_{channel}"] = {
                "TP": int(TP),
                "FP": int(FP),
                "FN": int(FN),
                "TN": int(TN),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4)
            }
            out_spikes.append(output_spikes_channel)

    # Format output spikes
    max_len = max(len(spikes) for spikes in out_spikes)
    out_spikes_padded = np.array([
        np.pad(spikes, (0, max_len - len(spikes)), mode='constant')
        for spikes in out_spikes
    ])
    
    net_prefix = f"{prefix}_adapt{adapt}" if adapt else prefix
    if test_dsb4:
        net_prefix += "_dsb4"

    if export_spikes:
        out_dir = "eval/spikes"
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{net_prefix}_spikes.npy"), out_spikes_padded)
        print(f"Saved spikes to: {out_dir}/{net_prefix}_spikes.npy")

    return

