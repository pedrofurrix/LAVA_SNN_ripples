import matplotlib.pyplot as plt
import numpy as np

def plot_raster(
    lif1_spikes=None, lif2_spikes=None, lif_out_spikes=None, up_dn_spikes=None,
    labels=("UP_DN", "LIF1", "LIF2", "LIF_OUT"),
    colors=None,
    figsize=(10,5),
    ripples=None,              # list of ripple times or ripple intervals (seconds)
    window=None,               # (start_s, end_s) in seconds
    ripple_color="yellow",
    ripple_alpha=0.4,
    ax=None,
):
    """
    Raster plot for neuron populations + ripple intervals.

    IMPORTANT:
    Spike times passed to this function
        MUST already be aligned so that 0 ms = window start.

    Ripples are given in seconds and will be shifted to the same reference.
    """

    # -------------------------------------------
    # Default colors
    # -------------------------------------------
    if colors is None:
        colors = {
            "UP_DN": ["red", "blue"],   # up & down units
            "LIF1": "black",
            "LIF2": "gray",
            "LIF_OUT": "purple",
        }

    groups = [up_dn_spikes, lif1_spikes, lif2_spikes, lif_out_spikes]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    y_offset = 0

    # -------------------------------------------
    # Plot spike rasters
    # -------------------------------------------
    for g_idx, group in enumerate(groups):
        if group is None:
            continue

        label = labels[g_idx]
        color_setting = colors.get(label, "black")

        # Normalize group to an iterable of per-neuron spike-time lists
        per_neuron_lists = None

        # Case A: numpy array of shape (T, N) -> convert to per-neuron spike times
        if isinstance(group, np.ndarray):
            if group.ndim == 2:
                T, N = group.shape
                per_neuron_lists = [np.where(group[:, n] > 0)[0] for n in range(N)]
            else:
                # 1D array: treat it as a list of spike times
                per_neuron_lists = [group]

        # Case B: python list/tuple of per-neuron iterables -> use directly
        elif isinstance(group, (list, tuple)):
            per_neuron_lists = [np.asarray(s) for s in group]

        else:
            # Unknown type — attempt to iterate
            try:
                per_neuron_lists = [np.asarray(s) for s in group]
            except Exception:
                # Skip if we cannot interpret the group
                continue

        for neuron_idx, spikes in enumerate(per_neuron_lists):
            times = np.asarray(spikes)
            if times.size == 0:
                continue

            # Assign color
            if isinstance(color_setting, list):
                color = color_setting[neuron_idx % len(color_setting)]
            else:
                color = color_setting

            y_val = y_offset + neuron_idx

            ax.scatter(
                times,
                np.full_like(times, y_val),
                s=4,
                color=color,
                label=label if neuron_idx == 0 else None
            )
           

        y_offset += len(per_neuron_lists) + 1  # spacing

    # -------------------------------------------
    # Plot ripple intervals (relative to window)
    # -------------------------------------------
    if ripples is not None and window is not None:
        # print(f"Plotting ripples: {ripples} within window: {window}")
        w_start, w_end = window  # in seconds

        for idx, r in enumerate(ripples):   

            # Case 1: ripple is a single timestamp in seconds
            if isinstance(r, (int, float)):
                r_start = r
                r_end = r
            else:
                # Case 2: ripple is (start, end)
                r_start, r_end = r

            if r_start<=w_end and r_end>=w_start:
                # print(f"  Ripple {idx}: {r_start}-{r_end} s overlaps window.")
                # convert ripples to milliseconds relative to window start
                rs = (r_start - w_start) * 1000
                re = (r_end   - w_start) * 1000

                ax.axvspan(
                    rs, re,
                    ymin=0, ymax=1,
                    color=ripple_color,
                    alpha=ripple_alpha,
                    zorder=0   # put behind spikes
                )
    # -------------------------------------------
    # Labels & cleanup
    # -------------------------------------------
    ax.set_xlabel("Time (ms relative to window start)")
    ax.set_ylabel("Neuron index")
    if window is not None:
        ax.set_xlim(0, (window[1] - window[0]) * 1000)
    ax.set_title("Raster + Ripples")
    ax.legend(loc="upper right")
    if ax.figure:
        ax.figure.tight_layout()
        if ax is None:
            plt.show()

    return ax
