import os
import numpy as np
import torch
from collections import deque
from torch.utils.data import TensorDataset, DataLoader
import pickle as pkl

import sys
curr_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir,os.pardir))
sys.path.append(parent_dir)  # Add parent directory to path for importing Net

from snnTorch.utils.start_net import Net
from liset_tk.read_data import read_data
from lists_sessions import *
from process_signal import load_experimental_data, spikify_signal
import json
from tqdm import tqdm   
def run_inference(
        prefix,
        data_path,
        sessions,
        channel_sessions=None,
        adapt=0,
        export_spikes=True,
        seed=None):

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Constants
    RIPPLE_DETECTION_OFFSET = [18, 45, 31, 20]
    PRED_CAUSALITY_WINDOW = 5
    TOLERANCE = 20
    MAX_DETECTION_OFFSET = 120  # in ms
    dt = 1
    refrac_period = 100 # in ms

    # Load network
    net = Net().to(device)
    net_path = f"../out/{prefix}_trained_net_loss.pth"
    net.load_state_dict(torch.load(net_path, map_location=device))
    net.eval()

    print(f"Loading model: {net_path}")
    print(f"Data path: {data_path}")

    # Seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Output containers
    all_metrics = {}
    all_spikes = {}

    time_max = 120 if adapt == 0 else adapt
    
    for session in tqdm(sessions):

        print(f"\n--- Running session: {session} ---")

        channel = channel_sessions.get(session, None)
        if channel is None:
            raise ValueError(f"No channel mapping provided for session {session}")

        # Load signal + ripples
        filtered_signal, ripples = load_experimental_data(
            data_path,
            session,
            channel=channel,
            load_data=True,
            downsample=False,
            normalize=True,
            offset=0.16,
        )

        # Convert to spikes
        spk, threshold = spikify_signal(
            filtered_signal,
            fs=30000,
            time_max=time_max,
            overlap=0.5,
            adapt_threshold=True if adapt !=0 else False,
            percentile=False,
            window_size=0.1, # 100 ms windows...
            sample_ratio=0.25,
            scaling_factor=1.0,
            refractory=0.0003,
            factor=30,
            initial_value=None,
            verbose=False,
            ripples=None,
        )

        # Pass if no ripples...
        if ripples is None or len(ripples) == 0:
            print("No ripples detected in this session, skipping...")
            pass


        # GT ripple start in ms
        ripples_start = ripples[:, 0] // 30 - TOLERANCE

        # tensors
        input_tensor = torch.tensor(spk, dtype=torch.float32).to(device)
        gt_tensor = torch.tensor(ripples_start, dtype=torch.float32).to(device)

        curr_gt_idx = 0
        active_gts = deque()
        TP = FP = FN = TN = 0
        total_steps = input_tensor.shape[0]

        session_spikes = []
        lif_out_refrac_times = torch.full(size=(1,), fill_value=-10000.0, device=device)

        net.reset_state()

        # Run network
        with torch.no_grad():
            for step in range(total_steps):

                curr_input = input_tensor[step, :].unsqueeze(0)

                # Schedule GT event
                if curr_gt_idx < len(gt_tensor):
                    curr_gt = int(gt_tensor[curr_gt_idx].item())
                    if curr_gt == step:
                        active_gts.append((curr_gt, MAX_DETECTION_OFFSET))
                        curr_gt_idx += 1

                # Network
                spk_out_tuple, mem, syn = net(curr_input)
                _, _, spk_out = spk_out_tuple

                # GT countdown
                new_active = deque()
                for gt, ttl in active_gts:
                    if ttl < 0:
                        FN += 1
                    else:
                        new_active.append((gt, ttl - 1))
                active_gts = new_active

                # Output spike?
                if torch.sum(spk_out) > 0:
                    spk_out_int = spk_out.squeeze(0).int()
                    step_time = step * dt
                    session_spikes.append(step_time)
                    refrac_mask = lif_out_refrac_times > step_time
                    valid_spk = torch.Tensor.bool(spk_out_int & (~refrac_mask))

                    if torch.sum(valid_spk) > 0:
                        lif_out_refrac_times[valid_spk] = float(step_time + refrac_period)
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
                    print(f"    {step}/{total_steps} ({100*step/total_steps:.2f}%)")

        # Metrics
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        # Save
        all_metrics[session] = {
            "channel": channel,
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
            "TN": int(TN),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        }

        all_spikes[session] = {
            "channel": channel,
            "spikes": session_spikes,
        }
        print(f"Session {session} ended. F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, num spikes: {len(session_spikes)}")

    if export_spikes:
        curr_dir=os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(curr_dir, "spikes", prefix)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nExporting spikes to directory: {out_dir}")
        net_prefix = f"{prefix}_adapt{adapt}" if adapt else prefix

        pkl_path = os.path.join(out_dir, f"{net_prefix}_spikes.pkl")

        # Check if file exists and append if so
        if os.path.exists(pkl_path):
            print(f"Appending to existing file: {pkl_path}")
            try:
                with open(pkl_path, "rb") as f:
                    existing_data = pkl.load(f)
                existing_data.update(all_spikes)
                all_spikes = existing_data
            except Exception as e:
                print(f"Could not load existing file {pkl_path}, overwriting. Error: {e}")

        with open(pkl_path, "wb") as f:
            pkl.dump(all_spikes, f, protocol=pkl.HIGHEST_PROTOCOL)

        print(f"\nSaved spike data to: {pkl_path}")
    return
