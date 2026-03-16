import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.patches as mpatches
import liset_tk.signal_aid as signal_aid
def plot_offline_detections(liset,ch=None, offset=0, filtered=None, extend=0.5, title='Overview', window=None, plot_offline=True,plot_light=True,plot_predicted=True):
    """
    Plot data with overlays for:
    - Ground truth ripples (yellow)
    - Model predicted ripples (blue)
    - Light stimulation TTLs (red)

    Parameters
    ----------
    ch : int or list of ints, optional
        Channel(s) to plot. If None, plots the first channel.
    offset : float, optional
        Vertical offset between channels.
    filtered : tuple (low, high), optional
        Bandpass filter range (Hz).
    extend : float, optional
        Extra context (in seconds) around each event.
    title : str, optional
        Title for the plot.
    window : tuple (start, end), optional
        Time window in seconds to plot (e.g., (10, 20)).
    """

    # ---------------------
    # Select channels
    # ---------------------
    if ch is None:
        ch = [0]
    elif isinstance(ch, int):
        ch = [ch]

    n_samples = liset.data.shape[0]
    time = np.arange(n_samples) / liset.fs

    # ---------------------
    # Apply time window
    # ---------------------
    if window is not None:
        start_s, end_s = window
        start_idx = int(start_s * liset.fs)
        end_idx = int(end_s * liset.fs)
        time = time[start_idx:end_idx]
        data_slice = liset.data[start_idx:end_idx, :]
    else:
        data_slice = liset.data

    fig, ax = plt.subplots(figsize=(15, 6))
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (normalized units)")

    # ---------------------
    # Plot data (optionally filtered)
    # ---------------------
    for i, ch_idx in enumerate(ch):
        sig = deepcopy(data_slice[:, ch_idx])
        if filtered:
            sig = signal_aid.bandpass_filter(sig, filtered, liset.fs)

        ax.plot(time, sig + i * offset, color="black", lw=0.8, label=f"Ch {ch_idx}" if i == 0 else "")

    min_y, max_y = ax.get_ylim()

    # ---------------------
    # Helper function to plot intervals safely within window
    # ---------------------
    def plot_intervals(intervals, color, label):
        if window is not None:
            mask = (intervals[:, 1] / liset.fs > start_s) & (intervals[:, 0] / liset.fs < end_s)
            intervals = intervals[mask]
        for i, r in enumerate(intervals):
            ax.fill_between(r / liset.fs, min_y, max_y, color=color, alpha=0.3, label=label if i == 0 else "")

    # ---------------------
    # Ground truth ripples (yellow)
    # ---------------------
    if hasattr(liset.annotated, "ripples_GT") and liset.annotated.ripples_GT is not None:
        plot_intervals(liset.annotated.ripples_GT, "yellow", "Ground truth")

    # ---------------------
    # Predicted ripples (blue)
    # ---------------------
    if plot_predicted:
        if hasattr(liset.from_data, "snn_predicts") and liset.from_data.snn_predicts is not None:
            # plot_intervals(self.from_data.snn_predicts, "tab:blue", "Predicted")
            starts = liset.from_data.snn_predicts[:, 0] / liset.fs
            if window is not None:
                starts = starts[(starts >= start_s) & (starts <= end_s)]
            ax.vlines(starts, min_y, max_y, color="tab:blue", alpha=0.5, lw=0.5,label="Predicted")

        if hasattr(liset.annotated, "snn_predicts") and liset.annotated.snn_predicts is not None:
            starts = liset.annotated.snn_predicts[:, 0] / liset.fs
            if window is not None:
                starts = starts[(starts >= start_s) & (starts <= end_s)]
            ax.vlines(starts, min_y, max_y, color="green", alpha=0.5, lw=0.5,label="Annotated Predicted")

    # ---------------------
    # Light stimulation (red)
    # ---------------------
    if plot_light:
        if hasattr(liset.from_data, "light_stim") and liset.from_data.light_stim is not None:
            plot_intervals(liset.from_data.light_stim, "red", "Light stim")
            starts = liset.from_data.light_stim[:, 0] / liset.fs
            if window is not None:
                starts = starts[(starts >= start_s) & (starts <= end_s)]
            ax.vlines(starts, min_y, max_y, color="red", alpha=0.5, lw=0.5)

        if hasattr(liset.annotated, "light_stim") and liset.annotated.light_stim is not None:
            plot_intervals(liset.annotated.light_stim, "orange", "Annotated light stim")
            starts = liset.annotated.light_stim[:, 0] / liset.fs
            if window is not None:
                starts = starts[(starts >= start_s) & (starts <= end_s)]
            ax.vlines(starts, min_y, max_y, color="orange", alpha=0.5, lw=0.5)

    # ---------------------
    # Offline detections (purple dashed)
    # ---------------------
    if plot_offline:
        if hasattr(liset, "offline_detections") and liset.offline_detections.size > 0:
            offline_starts = liset.offline_detections / liset.fs
            if window is not None:
                offline_starts = offline_starts[(offline_starts >= start_s) & (offline_starts <= end_s)]
            ax.vlines(offline_starts, min_y, max_y, color="purple", alpha=0.5, lw=0.5, ls='--', label="Offline detections")
    # ---------------------
    # Beautify
    # ---------------------
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())
    ax.grid(False, alpha=0.3)
    ax.spines[['top', 'right','bottom', 'left']].set_visible(False)
    plt.tight_layout()
    plt.show(block=False)
    