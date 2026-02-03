import os
import pickle
import numpy as np
import sys
import pandas as pd
from scipy.signal import hilbert
from tqdm import tqdm
import re

# Add path to project root
curr_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR= os.path.abspath(os.path.join(curr_dir, os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from liset_tk.read_data import read_data
from liset_tk.liset_tk_extra import liset_tk_extra
import liset_tk.lists_sessions as lists_sessions
from snnTorch.generalization_madrid.process_signal import load_experimental_data, ripple_band_power_trace


def extract_events(spikes_ms, ripples_samp, fs=30000, tolerance_ms=20, max_offset_ms=100,extra_tolerance_ms=20, fp_grouping_ms=50):
    """
    Classify events into TP, FP, FN and return their details.
    """
    # Convert spikes to samples
    spikes_samp = (spikes_ms * (fs / 1000)).astype(int)
    
    # Tolerance in samples
    tol_samp = int(tolerance_ms * fs / 1000)
    max_offset_samp = int(max_offset_ms * fs / 1000)
    fp_group_samp = int(fp_grouping_ms * fs / 1000)
    extra_tolerance_samp= int(extra_tolerance_ms * fs / 1000)
    # Ripple windows for detection
    ripple_windows = []
    for r in ripples_samp:
        start = int(r[0])
        end = int(r[1])
        # Detection window: [start - tol, start + max_offset]
        w_start = start - tol_samp
        w_end = start + max_offset_samp
        ripple_windows.append((w_start, w_end, start, end))
        
    tp_events = [] 
    fp_events = [] 
    fn_events = [] 
    
    detected_ripple_indices = set()
    used_spike_indices = set()
    
    # Sort spikes
    sorted_spike_indices = np.argsort(spikes_samp)
    sorted_spikes = spikes_samp[sorted_spike_indices]
    
    # 1. Find TPs and FNs
    for r_idx, (w_start, w_end, r_start, r_end) in enumerate(ripple_windows):
        
        # Find spikes in window
        idx_start = np.searchsorted(sorted_spikes, w_start)
        idx_end = np.searchsorted(sorted_spikes, w_end)
        
        in_window_indices = np.arange(idx_start, idx_end)
        
        if len(in_window_indices) > 0:
            # TP
            detected_ripple_indices.add(r_idx)
            
            # First spike is the detection
            first_spike_idx_in_sorted = in_window_indices[0]
            first_spike_samp = sorted_spikes[first_spike_idx_in_sorted]
            
            tp_events.append({
                'spike_samp': first_spike_samp,
                'ripple_idx': r_idx,
                'ripple_start': r_start,
                'ripple_end': r_end
            })
            
            # Mark all spikes in window as used
            for idx in in_window_indices:
                used_spike_indices.add(idx)
        else:
            # FN
            fn_events.append({
                'ripple_idx': r_idx,
                'ripple_start': r_start,
                'ripple_end': r_end
            })
            
    # 2. Find FPs
    current_fp_end = -np.inf
    
    for i in range(len(sorted_spikes)):
        if i in used_spike_indices:
            continue
            
        spk = sorted_spikes[i]
        
        # Check if this spike falls into ANY ripple window (even if that ripple was already detected)
        # This prevents counting extra spikes during a detected ripple as FPs
        is_in_ripple = False
        for w_start, w_end, _, _ in ripple_windows:
             # Using the same detection window logic for exclusion
             if w_start <= spk <= w_end + extra_tolerance_samp:
                 is_in_ripple = True
                 break
        
        if is_in_ripple:
            continue
            
        if spk > current_fp_end:
            fp_events.append({
                'spike_samp': spk
            })
            current_fp_end = spk + fp_group_samp
            
    return tp_events, fp_events, fn_events

def extract_power_snippets(power_trace, events, event_type, window_samp, look_around=3000, center="peak"):
    snippets = []
    max_idx = len(power_trace)
    
    for ev in events:
        center_idx = 0
        
        if center == "peak":
            if event_type == 'FP':
                # find peak power around the event
                center_idx = ev['spike_samp']
                snippet_around=power_trace[center_idx-look_around:center_idx+look_around] # 100 ms forward and back
                rel_idx=np.argmax(snippet_around)
                center_idx=center_idx+rel_idx-look_around
            else:
                # TP or FN: Find peak power in ripple interval
                r_start = int(ev['ripple_start'])
                r_end = int(ev['ripple_end'])
                
                # Clamp
                r_start = max(0, r_start)
                r_end = min(max_idx, r_end)
                
                if r_end > r_start:
                    ripple_power = power_trace[r_start:r_end]
                    if len(ripple_power) > 0:
                        peak_offset = np.argmax(ripple_power)
                        center_idx = r_start + peak_offset
                    else:
                        center_idx = r_start
                else:
                    center_idx = r_start
        else:
            # Center on event (spike or middle of ripple)
            if event_type == 'FN':
                r_start = int(ev['ripple_start'])
                r_end = int(ev['ripple_end'])
                center_idx = int((r_start + r_end) / 2)
            else:
                # TP or FP
                center_idx = ev['spike_samp']
                
        # Extract window
        start = center_idx - window_samp
        end = center_idx + window_samp
        
        if start >= 0 and end <= max_idx:
            snippets.append(power_trace[start:end])
            
    return snippets

def run_power_analysis():
    # Configuration
    SPIKES_ROOT = os.path.join(ROOT_DIR, "snnTorch", "generalization_madrid", "spikes")
    DATA_ROOT=os.path.join(ROOT_DIR, "snnTorch", "generalization_madrid", "data")
    os.makedirs(DATA_ROOT, exist_ok=True)
    DATA_PATH_ORIGINAL = r"C:\PedroFelix\Madrid_tests" # Update this if different
    DATA_PATH_EXTRA = r"C:\PedroFelix\extra_data\original_data" # Update this if different
    OUTPUT_FILE = os.path.join(SPIKES_ROOT, "power_analysis_results.pkl")
    center="peak"
    # Analysis parameters
    WINDOW_MS = 100 # +/- 100ms
    FS = 30000
    WINDOW_SAMP = int(WINDOW_MS * FS / 1000)
    SESSIONS_TO_EXCLUDE={        
        "2025-09-24_17-38-17", # Barely any ripples (14 in total...)
        "2025-09-25_12-52-22", # Not a good session to detect ripples
        }
    
    print(f"Searching for spike files in: {SPIKES_ROOT}")
    
    all_power_data = {} # Key: (Network, Adapt), Value: {'TP': [], 'FP': [], 'FN': []}
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(SPIKES_ROOT):
        for file in files:
            if file.endswith("_spikes.pkl"):
                pkl_path = os.path.join(root, file)
                print(f"\nProcessing: {file}")
                
                # Determine network name and adapt
                net_name = file.replace('_spikes.pkl', '')
                match = re.search(r'_adapt(\d+)$', net_name)
                adapt = int(match.group(1)) if match else 0
                if adapt==20:
                    net_name_clean = re.sub(r'_adapt\d+$', '', net_name) if match else net_name
                    
                    key = (net_name_clean, adapt)
                    if key not in all_power_data:
                        all_power_data[key] = {'TP': [], 'FP': [], 'FN': []}
                    
                    # Load spikes
                    with open(pkl_path, 'rb') as f:
                        all_spikes = pickle.load(f)
                    
                    # Iterate over sessions
                    for session, data in tqdm(all_spikes.items(), desc="Sessions"):
                        if session in SESSIONS_TO_EXCLUDE:
                            print(f"  Skipping excluded session: {session}")
                            continue

                        channel = data['channel']
                        spikes_ms = np.array(data['spikes'])
                        if session in lists_sessions.extra_sessions:
                            # Load Signal
                            try:
                                filtered_signal, ripples = load_experimental_data(
                                    DATA_PATH_EXTRA,
                                    session,
                                    channel=channel,
                                    load_data=True,
                                    verbose=False,
                                    normalize=False,
                                    data_reader=liset_tk_extra
                                )
                            except Exception as e:
                                print(f"  Error loading data for {session}: {e}")
                                continue
                        else:
                            # Load Signal
                            try:
                                filtered_signal, ripples = load_experimental_data(
                                    DATA_PATH_ORIGINAL,
                                    session,
                                    channel=channel,
                                    load_data=True,
                                    verbose=False,
                                    normalize=False,
                                    data_reader=read_data
                                )
                            except Exception as e:
                                print(f"  Error loading data for {session}: {e}")
                                continue
                            
                        if filtered_signal is None or ripples is None:
                            continue
                            
                        # Compute Power Trace
                        # filtered_signal is already 100-250Hz
                        power_trace = ripple_band_power_trace(filtered_signal, FS, smooth_ms=0, zscore=True)
                        
                        # Extract Events
                        # ripples from load_experimental_data are in samples (30kHz)
                        tp_ev, fp_ev, fn_ev = extract_events(spikes_ms, ripples, fs=FS)
                        
                        # Extract Snippets
                        tp_snips = extract_power_snippets(power_trace, tp_ev, 'TP', WINDOW_SAMP,center=center,look_around=WINDOW_SAMP)
                        fp_snips = extract_power_snippets(power_trace, fp_ev, 'FP', WINDOW_SAMP,center=center,look_around=WINDOW_SAMP)
                        fn_snips = extract_power_snippets(power_trace, fn_ev, 'FN', WINDOW_SAMP,center=center,look_around=WINDOW_SAMP)
                        
                        # Store (downsample to save space? maybe 1000Hz?)
                        # Let's downsample by factor of 3 (30kHz -> 10kHz)
                        factor = 3
                        
                        all_power_data[key]['TP'].extend([s[::factor] for s in tp_snips])
                        all_power_data[key]['FP'].extend([s[::factor] for s in fp_snips])
                        all_power_data[key]['FN'].extend([s[::factor] for s in fn_snips])

    # Save results
    filename=f"power_analysis_results_center_{center}.pkl"
    OUTPUT_FILE = os.path.join(DATA_ROOT, filename)

    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_power_data, f)
    print("Done.")

if __name__ == "__main__":
    run_power_analysis()
