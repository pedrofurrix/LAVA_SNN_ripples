from liset_data_reader.read_data import read_data
from liset_data_reader.liset_tk_extra import liset_tk_extra
import os
from liset_data_reader.signal_aid import bandpass_filter
from snnTorch.generalization_madrid.utils import *
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import butter, filtfilt, hilbert

def highpass_filter(signal, fs, highpass=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = highpass / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def load_experimental_data(path,name, downsample = False, normalize = True, numSamples = False, 
                           start = 0, verbose=True, channel=None,
                          load_data=True,data_reader=read_data):
    
    liset=data_reader(path,name, downsample = downsample, normalize = normalize, numSamples = numSamples, 
                           start = start, verbose=verbose,load_data=load_data,channels=[channel-1])

    filtered_signal = None
    if load_data:
        if liset.data.shape[1]>1:
            print("⚠️More than one channel loaded, please select a single channel.")
        channel_data= liset.data[:].reshape(-1)
        filtered_signal=bandpass_filter(channel_data, bandpass=[100,250], fs=liset.fs, order=4)
    
    if hasattr(liset,'annotated') and hasattr(liset.annotated,'ripples_GT'):
        ripples=liset.annotated.ripples_GT #original frequency - 30000 Hz
        if len(ripples)==0:
            ripples = None
        if verbose:
            print(f"Loaded {len(ripples)} ground truth ripples.")
    elif hasattr(liset,'ripples_GT'):
        ripples=liset.ripples_GT
        if len(ripples)==0:
            ripples = None
        if verbose:
            print(f"Loaded {len(ripples)} ground truth ripples.")
    return filtered_signal, ripples

# Double Checked - should work okay and return a spikified signal in the shape [n_samples(ms), 2 (UP/DN)] 

def spikify_signal(
    signal,
    fs,
    time_max=20.0,
    overlap=0.5,
    adapt_threshold=True,
    percentile=False,
    window_size=0.10,
    sample_ratio=0.25,
    scaling_factor=1.0,
    refractory=0,
    factor=30,
    initial_value=None,
    verbose=False,
    ripples=None,     # kept for compatibility but unused
):

    N = len(signal)
    out_len = N // factor if factor > 1 else N
    spikified = np.zeros((out_len, 2))

    win = int(fs * time_max)
    step = int(fs * overlap * time_max)

    # -------- THRESHOLD HELPERS --------
    def compute_threshold(x):
        return calculate_threshold(x, fs, window_size, sample_ratio, scaling_factor)

    def get_threshold_window(signal, t):
        """Sliding window for adaptive threshold."""
        if t < win:
            return signal[:win]
        else:
            return signal[t - win:t]

    if verbose:
        print(f"[spikify] N={N}, out_len={out_len}, win={win}, step={step}, factor={factor}")

    # =============================
    # ADAPTIVE THRESHOLD MODE
    # =============================
    if adapt_threshold:

        thresholds = []

        for t in range(0, N, step):

            # --- sliding threshold window ---
            tw = get_threshold_window(signal, t)

            thr = compute_threshold(tw)
            thresholds.append(thr)

            # --- extract chunk ---
            r_edge = min(t + step, N)
            chunk = signal[t:r_edge]

            # --- spiking ---
            spk, initial_value = up_down_channel(
                chunk, thr, fs, refractory,
                initial_value=initial_value, return_value=True
            )

            # Downsample spikes if using factor
            if factor > 1:
                spk, _ = extract_spikes_downsample(spk, factor)

            L = t // factor
            R = r_edge // factor
            spikified[L:R] = spk

        if verbose:
            print("Spikification complete.")
            print(f"UP spikes:   {np.sum(spikified[:,0])}")
            print(f"DOWN spikes: {np.sum(spikified[:,1])}")

        return spikified, thresholds

    # =============================
    # FIXED THRESHOLD MODE
    # =============================
    else:
        thr = compute_threshold(signal[:win])

        spk = up_down_channel(signal, thr, fs, refractory,
                              initial_value=None, return_value=False)

        if factor > 1:
            spk, _ = extract_spikes_downsample(spk, factor)

        return spk, thr

def reconstruct_signal_from_spikes(spikes, thresholds, adapt_threshold, fs_spikes, 
                                   scaling_factor=1.0, # Not actually used if factor is passed, but kept for signature
                                   factor=30, 
                                    time_max=20,
                                overlap=0.5,         # Downsampling factor used in spikify
                                   original_fs=30000, 
                                   highpass=100):
   
    N = len(spikes)
    reconstructed = np.zeros(N)

    # Convert to array if it isn't one (upfront)
    thresholds_arr = np.array(thresholds)
    step= int(original_fs * overlap * time_max) 

    if adapt_threshold:

        # Ensure thresholds is a list or array
        thresholds_arr = np.array(thresholds)
        
        # original indices
        orig_indices = np.arange(N) * 30 # since factor=30 in spikify - 30kHz -> 1kHz
        
        if step is None:
            # For safety, use 0 if fail.
            thr_indices = np.zeros(N, dtype=int)
        else:
            thr_indices = (orig_indices // step).astype(int)
        
        # Clip to valid range
        thr_indices = np.clip(thr_indices, 0, len(thresholds_arr) - 1)
        
        # Now expand thresholds to size N
        current_thresholds = thresholds_arr[thr_indices]
        
    else:
        # Fixed threshold (single value broadcasted)
        # Handle case where thresholds might be a single-element list or scalar
        if thresholds_arr.ndim > 0 and thresholds_arr.size > 0:
            val = thresholds_arr[0]
        else:
            # Scalar fallback
            val = float(thresholds_arr) if thresholds_arr.size==1 else 0.0
            
        current_thresholds = np.full(N, val)

    # -----------------------------------------------
    # Vectorized reconstruction using cumsum
    # -----------------------------------------------
    # reconstructed[t] = reconstructed[t-1] + (UP[t] - DOWN[t]) * thr[t]
    # => reconstructed = cumsum( (UP - DOWN) * thr )
    
    ups = spikes[:, 0]
    dns = spikes[:, 1]
    
    changes = (ups - dns) * current_thresholds
    reconstructed = np.cumsum(changes)

    if highpass:
        # Avoid filter artifacts if signal is too short
        if len(reconstructed) > 3 * fs_spikes / highpass:
             reconstructed = highpass_filter(reconstructed, fs=fs_spikes, highpass=highpass)
             
    return reconstructed

def plot_signal_spikes(
    signal,
    spikes,
    fs_signal=30000,
    fs_spikes=1000,
    ripples=None,
    window=None,
    ripple_color="yellow",
    ripple_alpha=0.25,
    figsize=(15,5)
):
    """
    signal:      filtered LFP already aligned to window (shape N)
    fs_signal:   sampling rate of signal (e.g., 30000)
    spikes:      spikified data already aligned (N_spikes, 2)
    fs_spikes:   sampling rate of spikes (e.g., 1000)
    ripples:     ripple timestamps in *seconds* relative to whole session
    window:      (start_s, end_s) used only to determine which ripples appear
    """

    w_start, w_end = window

    # ------------------------------------------------------
    # Build time axes — these are already correct
    # ------------------------------------------------------
    t_sig = np.arange(len(signal)) / fs_signal        # in seconds
    t_spk = np.arange(len(spikes)) / fs_spikes        # in seconds

    # Convert both to milliseconds for nicer plotting
    t_sig_ms = t_sig * 1000
    t_spk_ms = t_spk * 1000

    plt.figure(figsize=figsize)

    # ------------------------------------------------------
    # Plot the LFP signal
    # ------------------------------------------------------
    plt.plot(t_sig_ms, signal, color="black", lw=0.8, label="Filtered LFP")

    # ------------------------------------------------------
    # Plot UP/DOWN spikes (already aligned)
    # ------------------------------------------------------
    up_times  = t_spk_ms[spikes[:,0] > 0]
    dn_times  = t_spk_ms[spikes[:,1] > 0]

    # Place spikes above the signal amplitude
    ymax = np.max(signal)
    ymin= np.min(signal)

    plt.vlines(up_times, ymax*0.75, ymax*1, color="red", lw=1, label="UP")

    plt.vlines(dn_times, ymin, ymin*0.75, color="blue", lw=1,label="DOWN")

    # ------------------------------------------------------
    # Plot ripples (now relative to window start)
    # ------------------------------------------------------
    for idx, r in enumerate(ripples):

        # Single timestamp or start/end?
        if isinstance(r, (int, float)):
            r_start = r
            r_end   = r
        else:
            r_start, r_end = r

        # Skip outside the window
        if r_end < w_start or r_start > w_end:
            continue

        # Convert to milliseconds relative to window start
        rs_ms = (r_start - w_start) * 1000
        re_ms = (r_end   - w_start) * 1000

        plt.axvspan(
            rs_ms, re_ms,
            color=ripple_color,
            alpha=ripple_alpha,
            label="Ripple" if idx == 0 else None
        )

    # ------------------------------------------------------
    # Formatting
    # ------------------------------------------------------
    plt.xlabel("Time (ms, relative to window start)")
    plt.ylabel("Filtered LFP amplitude")
    plt.title("Signal + UP/DN Spikes + Ripples")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.xlim(0, (w_end - w_start)*1000)  # in ms
