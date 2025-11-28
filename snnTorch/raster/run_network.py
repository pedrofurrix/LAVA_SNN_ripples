import os
import numpy as np
import torch
from collections import deque
from torch.utils.data import TensorDataset, DataLoader
import pickle as pkl



from snnTorch.utils.start_net import Net
from liset_tk.read_data import read_data
from liset_tk.lists_sessions import *
from process_signal import load_experimental_data, spikify_signal
import json
from tqdm import tqdm   
def run_network(
        spikified,
        prefix,
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
    net_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,"out",f"{prefix}_trained_net_loss.pth")
    net.load_state_dict(torch.load(net_path, map_location=device))
    net.eval()

    print(f"Loading model: {net_path}")


    # tensors
    input_tensor = torch.tensor(spikified, dtype=torch.float32).to(device)

    # gt_tensor = torch.tensor(ripples_start, dtype=torch.float32).to(device)

    total_steps = input_tensor.shape[0]

    session_spikes = []
    net.reset_state()

    # Run network
    with torch.no_grad():
        for step in range(total_steps):

            curr_input = input_tensor[step, :].unsqueeze(0)

            # Network
            spk_out_tuple, mem, syn = net(curr_input)
            spk_lif1, spk_lif2, spk_out = spk_out_tuple
            # spk_* are tensors shaped (batch=1, n_neurons)
            # Collect spikes per population (convert to CPU numpy)
            spk1_np = spk_lif1.squeeze(0).cpu().numpy()
            spk2_np = spk_lif2.squeeze(0).cpu().numpy()
            spkout_np = spk_out.squeeze(0).cpu().numpy()

            session_spikes.append((spk1_np, spk2_np, spkout_np))

    # Convert collected spikes into arrays: shape (time_steps, n_neurons)
    if len(session_spikes) == 0:
        print("No spikes collected (empty input).")
        return None

    spk1_all = np.vstack([s[0] for s in session_spikes])
    spk2_all = np.vstack([s[1] for s in session_spikes])
    spkout_all = np.vstack([s[2] for s in session_spikes])

    # Export spikes if requested
    if export_spikes:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_spikes")
        os.makedirs(save_dir, exist_ok=True)

        # Dense arrays (time x neurons), dtype uint8 (0/1)
        np.save(os.path.join(save_dir, f"{prefix}_spikes_lif1.npy"), spk1_all.astype(np.uint8))
        np.save(os.path.join(save_dir, f"{prefix}_spikes_lif2.npy"), spk2_all.astype(np.uint8))
        np.save(os.path.join(save_dir, f"{prefix}_spikes_out.npy"), spkout_all.astype(np.uint8))

        # Also save per-neuron spike times (lists of time indices in ms)
        per_neuron_times = {
            'lif1': [np.where(spk1_all[:, n] > 0)[0].tolist() for n in range(spk1_all.shape[1])],
            'lif2': [np.where(spk2_all[:, n] > 0)[0].tolist() for n in range(spk2_all.shape[1])],
            'out': [np.where(spkout_all[:, n] > 0)[0].tolist() for n in range(spkout_all.shape[1])],
        }
        with open(os.path.join(save_dir, f"{prefix}_spikes_per_neuron.pkl"), 'wb') as f:
            pkl.dump(per_neuron_times, f)

        print(f"Saved spikes to {save_dir}: {prefix}_spikes_lif1.npy, {prefix}_spikes_lif2.npy, {prefix}_spikes_out.npy, and per-neuron pickle.")

    # Return dense arrays for programmatic use
    return {
        'lif1': spk1_all,
        'lif2': spk2_all,
        'out': spkout_all,
    }
