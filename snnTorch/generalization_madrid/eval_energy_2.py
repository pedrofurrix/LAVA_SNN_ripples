import argparse
import gc
import os
import sys
import numpy as np
import torch
from collections import deque
from torch.utils.data import TensorDataset, DataLoader
import pickle as pkl
from codecarbon import EmissionsTracker
import json
from tqdm import tqdm  

ROOT_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in os.sys.path:
    sys.path.append(ROOT_DIR)

import liset_data_reader.lists_sessions as lists_sessions
from snnTorch.utils.minimal_net import Net
from liset_data_reader.read_data import read_data
from liset_data_reader.liset_tk_extra import liset_tk_extra

from process_signal import load_experimental_data, spikify_signal




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
    net_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "out", f"{prefix}_trained_net_loss.pth")
    print(f"Loading model from: {net_path}")
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
        if session in lists_sessions.extra_sessions:
            
            # Load signal + ripples
            filtered_signal, ripples = load_experimental_data(
                data_path,
                session,
                channel=channel,
                load_data=True,
                downsample=False,
                normalize=True,
                data_reader=liset_tk_extra,
            )
        else:
            # Load signal + ripples
            filtered_signal, ripples = load_experimental_data(
                data_path,
                session,
                channel=channel,
                load_data=True,
                downsample=False,
                normalize=True,
                data_reader=read_data,
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
        spk_unsqueezed=np.expand_dims(spk, axis=0)  # Add batch dimension
        input_tensor = torch.tensor(spk_unsqueezed, dtype=torch.float32).to(device)

        net.reset_state()

        # Track emissions for this session
        tracker=EmissionsTracker(project_name="snn_ripple_inference_minimal_2", log_level="error")
        tracker.start()
        # Run network
        spks_out=[]
        with torch.no_grad():
            for t in range(input_tensor.shape[1]):
                input_t = input_tensor[:,t,:]
                spk_out = net(input_t)
                spks_out.append(spk_out)    
        # Stop emissions tracker for this session and log results
        tracker.stop()
        print("Emissions for this session: {:.4f} kg CO2".format(tracker.final_emissions))
        print(f"{len(spks_out)} timesteps processed for session {session}.")
if __name__ == "__main__":
    
    # Unpack args
    prefix = "updnb4ds_100_7"
    adapt = 20

    data_path=r"C:\PedroFelix\extra_data\original_data"
    
    # sessions=lists_sessions.annotated_sessions

#     # Original Sessions 
#     session_set={"2025-09-22_17-55-26", #R
#                  "2025-09-23_15-50-26", #R
#                  "2025-09-24_10-24-40", #R
#                  "2025-09-25_16-41-14"} #R

#     # Extra
# #     session_set={"2025-09-24_16-29-07", #R   
# #             "2025-09-24_17-38-17",} #R 
#     session_set.update({ "2025-09-24_16-29-07", #R   
#             "2025-09-24_17-38-17",
#                 "2025-09-22_17-42-27", 
#                  "2025-09-23_16-17-52", 
#                  "2025-09-24_11-34-51",
#                  "2025-09-25_11-21-53",
#                  "2025-09-25_12-52-22",})
    session_set=    {
            "Calbai32FPGA_251003_144832",
            # "Calbai32FPGA_251003_150055",
            # "PV01ai32FPGA_250611_115923",
            # "PV01ai32FPGA_250611_122326",
            # "Calb_251209_160255",
            # "Calb_251210_115904",
            # "Calb_251210_121141",
            # "Calb_251210_122327",
            # "Calb_251210_162849",
            # "Calb_251210_164150",
            # "Calb_251210_165332",
            # "Calb_251211_104316",
            # "Calb_251211_105518",
            # "Calb_251211_110650",
                } 
    channel_sessions=lists_sessions.channel_sessions
    run_inference(
            prefix,
            data_path,
            session_set,
            channel_sessions=channel_sessions,
            adapt=adapt,
            export_spikes=False,
            seed=None)
    gc.collect()