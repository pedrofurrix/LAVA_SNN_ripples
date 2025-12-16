import sys
import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir,os.pardir))
sys.path.append(parent_dir)  # Add parent directory to path for importing Net
import liset_tk.lists_sessions as lists_sessions
from process_signal import *
from plot_raster import *
from run_network import run_network
import matplotlib.pyplot as plt

sessions_channels=lists_sessions.channel_sessions
if __name__ == "__main__":

    path=r"D:\Madrid_tests"
    name="2025-09-24_15-13-10"
    downsample=False
    normalize=False
    numSamples=False
    channel=sessions_channels[name]-1
    fs=30000
    # Example usage
    signal, ripples = load_experimental_data(
        path=path,
        name=name,
        downsample=downsample,
        normalize=normalize,
        numSamples=numSamples,
        start=0,
        verbose=True,
        original_fs=30000,
        channel=channel,
        invert=False,
        offset=0,
        load_data=True,
        visualize=True
    )
    
    ripple_seconds=ripples/30000  # convert to seconds

    raw = input("Enter time window to spikify (e.g., 20.0 or 10,20,30; leave empty for none): ").strip()
    if raw == "":
        window = None
    else:
        # split by commas if present
        parts = [p.strip() for p in raw.split(",")]
        # convert to float
        window = [float(p) for p in parts if p != ""]

    plt.close('all')
    spk,thr=spikify_signal( # Something really weird here
    signal,
    fs=fs,
    time_max=20.0,
    overlap=0.5,
    adapt_threshold=True,
    percentile=False,
    window_size=0.10,
    sample_ratio=0.25,
    scaling_factor=1.0,
    refractory=0.0003,
    factor=30,
    initial_value=None,
    external_window=window,
    ripples=ripple_seconds,
    )

    print(f"window: {window}")
    print(f"spk.shape: {spk.shape}")
    # spk=np.zeros((100,2))
    # spk[10,0]=1
    # spk[20,1]=1

    prefix = "updnb4ds_100_13b"
    res = run_network(spk, prefix, export_spikes=False)

    # Create a shared-x figure: top = filtered signal + up/dn spikes, bottom = raster
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 1.2]})

    # Top: signal + up/down spikes. Use the same window and ripples.
    # If `external_window` was supplied to spikify, `spk` is already aligned to that window.
    plot_signal_spikes(
        signal[ (int(window[0]*fs) if window is not None else 0):(int(window[1]*fs) if window is not None else len(signal)) ] if window is not None else signal,
        spk,
        fs_signal=fs,
        fs_spikes=1000,
        ripples=ripple_seconds,
        window=window,
        ripple_color="yellow",
        ripple_alpha=0.5,
        figsize=(15, 5),
        ax=axes[0]
    )

    # Bottom: raster. Pass up_dn_spikes as the spikified input so UP/DN appear too.
    ax=plot_raster(lif1_spikes=res['lif1'], lif2_spikes=res['lif2'], lif_out_spikes=res['out'], up_dn_spikes=spk,
                labels=("UP_DN", "LIF1", "LIF2", "LIF_OUT"), window=window, ripples=ripple_seconds,
                colors=None, figsize=(10,5), ax=axes[1])

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(curr_dir, f"raster_{name}.svg"), dpi=300)

# WINDOW USED - 162,166
# 100-150 is good - but change titles and stuff