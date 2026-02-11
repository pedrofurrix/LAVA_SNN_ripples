from liset_data_reader.read_data import read_data
# from liset_paper import liset_paper
from liset_data_reader.signal_aid import bandpass_filter
from snnTorch.raster.utils import calculate_threshold, up_down_channel, extract_spikes_downsample
from snnTorch.raster.visualization import plot_offline_detections
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def load_experimental_data(path,name, downsample = False, normalize = True, numSamples = False, 
                           start = 0, verbose=True, original_fs=30000,channel=None,
                           invert=False,offset=0.16,load_data=True,visualize=False,):
    
    try:
        session_date = datetime.strptime(name.split('_')[0], "%Y-%m-%d").day
    except Exception as e:
        print(f"⚠️ Skipping {name} - invalid format: {e}")
        return None

    # determine invert rule
    invert = session_date >= 24

    liset=read_data(path,name, downsample = downsample, normalize = normalize, numSamples = numSamples, 
                           start = start, verbose=verbose, original_fs=original_fs,channel_num=None,
                           invert=invert,offset=offset,load_data=load_data)
    
    channel_data=liset.data[:,channel-1]
    filtered_signal=bandpass_filter(channel_data, bandpass=[100,250], fs=liset.fs, order=4)
    ripples=liset.annotated.ripples_GT
    if visualize:
        plot_offline_detections(liset,ch=channel, filtered=[100,250], title="Filtered Signal with Ground Truth Ripples",
                                 window=None, plot_offline=False,plot_light=False,plot_predicted=False)
    return filtered_signal, ripples

# Double Checked - should work okay and return a spikified signal in the shape [n_samples(ms), 2 (UP/DN)] 

def spikify_signal(
    signal,
    fs,
    time_max=20.0,
    overlap=0.5,
    adapt_threshold=True,
    percentile=False,
    window_size=0.01,
    sample_ratio=0.2,
    scaling_factor=1.0,
    refractory=0,
    factor=1,
    initial_value=None,
    external_window=None,
    verbose=False,
    ripples=None,           # for plotting
):

    N = len(signal)
    out_len = N // factor if factor > 1 else N
    spikified = np.zeros((out_len, 2))

    win = int(fs * time_max)
    step = int(fs * overlap*time_max)

    # ---------------------------
    # Helper functions
    # ---------------------------
    def compute_threshold(x):
        return calculate_threshold(x, fs, window_size, sample_ratio, scaling_factor)

    def get_threshold_window(signal, t):
        """Standard sliding window used when no external_window is given."""
        if t < win:
            return signal[:win]
        else:
            return signal[t - win:t]

    # ---------------------------
    # If user gives external window → find needed t-values
    # ---------------------------
    if external_window is not None:

        ext_start = int(external_window[0] * fs)
        ext_end   = int(external_window[1] * fs)

        needed_t = []
        for t in range(0, N, step):
            # processing region for this t
            r_edge = min(t + step, N)
            if r_edge >= ext_start and t <= ext_end:
                needed_t.append(t)

        # gather all threshold windows required
        thr_windows = []
        for t in needed_t:
            if t < win:
                thr_windows.append((0, win))
            else:
                thr_windows.append((t - win, t))

        if thr_windows:
            g_start = min(w[0] for w in thr_windows)
            g_end   = max(w[1] for w in thr_windows)
            threshold_base = signal[g_start:g_end]
            offset = g_start
        else:
            threshold_base = signal
            offset = 0

    else:
        needed_t = None
        threshold_base = signal
        offset = 0
    
    if verbose:
        print(f"spikify_signal: N={N}, out_len={out_len}, win={win}, step={step}, factor={factor}")
        threshold_base_len = len(threshold_base)
        print(f"threshold_base length: {threshold_base_len}, offset={offset}")
        print("Spikifying...")
    # ============================
    # ADAPTIVE THRESHOLD MODE
    # ============================
    if adapt_threshold:

        thresholds = []

        for t in range(0, N, step):

            # skip irrelevant t's when external window is used
            if needed_t is not None and t not in needed_t:
                continue

            # --- select the correct threshold window ---
            if needed_t is not None:
                # compute global window boundaries
                if t < win:
                    ws, we = 0, win
                else:
                    ws, we = t - win, t
                # slice from reduced threshold_base
                tw = threshold_base[max(0,(ws - offset)):(we - offset)]
            else:
                tw = get_threshold_window(signal, t)

            # --- threshold computation ---
            thr = compute_threshold(tw)
            thresholds.append(thr)

            # --- spiking ---
            r_edge = min(t + step, N)
            chunk = signal[t:r_edge]

            spk, initial_value = up_down_channel(
                chunk, thr, fs, refractory,
                initial_value=initial_value, return_value=True
            )

            if factor > 1:
                spk, _ = extract_spikes_downsample(spk, factor)

            L = t // factor
            R = r_edge // factor
            spikified[L:R] = spk

        if external_window is not None:
            left= ext_start // factor
            right= ext_end // factor
            spikified = spikified[left:right]
            
        # plot_signal_spikes(
        #         signal[ext_start:ext_end] if external_window is not None else signal,
        #         spikified,
        #         fs_signal=30000,
        #         fs_spikes=1000,
        #         ripples=ripples,
        #         window=external_window,
        #         ripple_color="yellow",
        #         ripple_alpha=0.5,
        #         figsize=(15,5)
        #     )
        
        if verbose:
            print("Spikification complete.")
            print(f"Number of UP spikes: {np.sum(spikified[:,0])} ,\nNumber of DOWN spikes: {np.sum(spikified[:,1])}")
        return spikified, thresholds

        # ============================
        # FIXED THRESHOLD MODE
        # ============================
    else:
        thr = compute_threshold(signal[:win])

        spk = up_down_channel(signal, thr, fs, refractory,
                              initial_value=None, return_value=False)

        if factor > 1:
            spk, _ = extract_spikes_downsample(spk, factor)
        
        if external_window is not None:
            left= ext_start // factor
            right= ext_end // factor
            spk = spk[left:right]

        return spk,thr


def plot_signal_spikes(
    signal,
    spikes,
    fs_signal=30000,
    fs_spikes=1000,
    ripples=None,
    window=None,
    ripple_color="yellow",
    ripple_alpha=0.25,
    figsize=(15,5),
    ax=None,
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

    # If an axis was provided, use it; otherwise create a figure/axis
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True

    # ------------------------------------------------------
    # Plot the LFP signal
    # ------------------------------------------------------
    ax.plot(t_sig_ms, signal, color="black", lw=0.8, label="Filtered LFP")

    # ------------------------------------------------------
    # Plot UP/DOWN spikes (already aligned)
    # ------------------------------------------------------
    up_times  = t_spk_ms[spikes[:,0] > 0]
    dn_times  = t_spk_ms[spikes[:,1] > 0]

    # Place spikes above the signal amplitude
    ymax = np.max(signal)
    ymin= np.min(signal)

    ax.vlines(up_times, ymax*0.75, ymax*1, color="red", lw=1, label="UP")

    ax.vlines(dn_times, ymin, ymin*0.75, color="blue", lw=1,label="DOWN")

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

        ax.axvspan(
            rs_ms, re_ms,
            color=ripple_color,
            alpha=ripple_alpha,
            label="Ripple" if idx == 0 else None
        )

    # ------------------------------------------------------
    # Formatting
    # ------------------------------------------------------
    ax.set_xlabel("Time (ms, relative to window start)")
    ax.set_ylabel("Filtered LFP amplitude")
    ax.set_title("Signal + UP/DN Spikes + Ripples")
    ax.legend()
    if created_fig:
        ax.figure.tight_layout()
        plt.show(block=False)
    ax.set_xlim(0, (w_end - w_start)*1000)  # in ms
