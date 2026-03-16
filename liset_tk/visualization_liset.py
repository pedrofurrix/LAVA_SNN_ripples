import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.patches as mpatches
import liset_data_reader.signal_aid as signal_aid
from liset_data_reader.liset_paper import liset_paper
import os

def plot_offline_detections(liset,ch=None, offset=0, filtered=None, extend=0.5, title='Overview', window=None, ):
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
        ch = range(liset.data.shape[1])  # Plot all channels by default
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

        ax.plot(time, sig + i * offset, 
                lw=1.2, label=f"Ch {ch_idx}" if i == 0 else "")

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
    if hasattr(liset, "ripples_GT") and liset.ripples_GT is not None:
        plot_intervals(liset.ripples_GT, "yellow", "Ground truth")

    
    # ---------------------
    # Beautify
    # ---------------------
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    # ax.legend(unique.values(), unique.keys())
    ax.grid(False)
    ax.spines[['top', 'right','bottom', 'left']].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    plt.show(block=True)
    return fig, ax

if __name__ == "__main__": 
    DATA_PATH=r"E:\NCN\neurospark_mat\Download_from_paper"
    datasets=os.listdir(DATA_PATH)
    dataset_to_read=datasets[2]
    liset=liset_paper(os.path.join(DATA_PATH,dataset_to_read),shank=1, normalize=True,downsample=10000)
    fig,ax=plot_offline_detections(liset, offset=0.8, filtered=False, extend=0.5, title='Overview', window=(89.5,90), )
    file_path=os.path.join(os.path.dirname(__file__), "figures","pngs","ripple_example.png")
    fig.savefig(file_path, dpi=400)