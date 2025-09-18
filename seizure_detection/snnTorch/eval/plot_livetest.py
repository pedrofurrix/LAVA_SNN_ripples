import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict

def plot_iis_live_test(prefix, adapt, id, shank=2, window=None,
                      title='Live Test IIS Detection', xlabel='Time (s)', ylabel='Amplitude (Z-score)',
                      input_spikes=True, filename=None,
                      tolerance_ms=50, refractory_period_ms=100, jitter_ms=100, max_detection_offset=80):
    """
    Visualizes the performance of an IIS detector on live test data.

    Args:
        prefix (str): Prefix for the output spike file (e.g., 'LIF_sTorch').
        adapt (int): Adaptation parameter, used for file naming.
        id (int): Dataset index to load.
        shank (int): The shank number being analyzed.
        window (tuple, optional): (start_sec, end_sec) for a time window to plot.
        title (str): The title of the plot.
        xlabel (str), ylabel (str): Axis labels.
        input_spikes (bool): If True, plot the UP/DOWN input spikes.
        filename (str, optional): If provided, save the figure here.
        tolerance_ms, refractory_period_ms, jitter_ms, max_detection_offset (int): Detection parameters.
    """

    # --- 1. Data Loading ---
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(curr_dir, os.pardir, os.pardir, "up_dn_spikes", "live_validation", f"adapt_{adapt}")
    datasets = os.listdir(data_dir)
    data_dir = os.path.join(data_dir, datasets[id])

    print(f"Loading data from: {data_dir}")

    try:
        spikes = np.load(os.path.join(data_dir, f'spikified_{shank}.npy'))   # shape: (time, channels, 2)
        gt_iis = np.load(os.path.join(data_dir, f'IISs_{shank}.npy'))       # 1D array of GT IIS times
        data = np.load(os.path.join(data_dir, f'filtered_{shank}.npy'))     # shape: (time, channels)
        seizures = np.load(os.path.join(data_dir, f'seizures_{shank}.npy')) # (N, 2) intervals
        
        spikes_file = f'{prefix}_adapt{adapt}_spikes.npy' if adapt > 0 else f'{prefix}_spikes.npy'
        output_spikes_all = np.load(os.path.join(os.path.dirname(__file__),"live_results2","spikes", spikes_file))

        # Select only the dataset’s output spikes → 1D array
        output_spikes = output_spikes_all[id]
        # print(output_spikes_all.shape)
        print(f"Output spikes shape: {output_spikes.shape}")
    except FileNotFoundError as e:
        print(f"Error: A required data file was not found. {e}")
        return

    # --- 2. Setup Plotting and Time Window ---
    fs = 1000  # Hz (assumption: 1 kHz downsampled data)

    if window:
        start_sample, end_sample = int(window[0] * fs), int(window[1] * fs)
        end_sample = min(end_sample, data.shape[0])
    else:
        start_sample, end_sample = 0, data.shape[0]

    # Slice into window
    data_win = data[start_sample:end_sample, :]
    spikes_win = spikes[start_sample:end_sample, :, :]
    gt_iis_win = gt_iis[(gt_iis >= start_sample) & (gt_iis < end_sample)]
    seizures_win = seizures[(seizures[:, 1] >= start_sample) & (seizures[:, 0] < end_sample)]

    # Filter output spikes for the window
    output_spikes_win = output_spikes[(output_spikes >= start_sample) & (output_spikes < end_sample)]
    output_spikes_win_sec = output_spikes_win / fs

    print(f"Windowed output spikes: {len(output_spikes_win_sec)} events")

    time_sec = np.arange(start_sample, end_sample) / fs

    # --- 3. Classify Output Spikes (TP/FP) ---
    classified_spikes = []
    tolerance_sec = tolerance_ms / 1000.0
    refractory_sec = refractory_period_ms / 1000.0
    jitter_sec = jitter_ms / 1000.0
    max_detection_offset_sec = max_detection_offset / 1000.0

    gt_iis_times_sec = gt_iis_win / fs
    used_gt_indices = set()
    last_fp_time, last_tp_time = -1e10, -1e10

    for spike_time in output_spikes_win_sec:
        is_tp = False
        if len(gt_iis_times_sec) > 0:
            time_diffs = np.abs(gt_iis_times_sec - spike_time)
            closest_gt_idx = np.argmin(time_diffs)
            if time_diffs[closest_gt_idx] <= max_detection_offset_sec and closest_gt_idx not in used_gt_indices:
                if spike_time - last_tp_time >= refractory_sec:
                    classified_spikes.append(('TP', spike_time))
                    used_gt_indices.add(closest_gt_idx)
                    last_tp_time = spike_time
                    is_tp = True

        if not is_tp:
            is_near_any_gt = np.any(np.abs(gt_iis_times_sec - spike_time) <= tolerance_sec)
            if not is_near_any_gt and spike_time - last_fp_time >= jitter_sec:
                classified_spikes.append(('FP', spike_time))
                last_fp_time = spike_time

    # --- 4. Plotting ---
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.plot(time_sec, data_win[:, 3], color='black', alpha=0.7, label='Signal (Ch 3)')
    y_max = np.max(data_win[:, 3]) * 1.5
    y_min = np.min(data_win[:, 3]) * 1.5

    # Plot seizures as shaded regions
    for start, end in seizures_win:
        ax.fill_between([start/fs, end/fs], y_min, y_max, color='purple', alpha=0.1, label='Seizure')

    # Plot input UP/DOWN spikes
    if input_spikes:
        up_spikes_sec = time_sec[spikes_win[:, 3, 0] == 1]
        down_spikes_sec = time_sec[spikes_win[:, 3, 1] == 1]
        spike_height = (y_max - y_min) * 0.1
        ax.vlines(up_spikes_sec, ymin=y_max - spike_height, ymax=y_max, color='red', alpha=0.3, label='Up Spikes')
        ax.vlines(down_spikes_sec, ymin=y_min, ymax=y_min + spike_height, color='blue', alpha=0.3, label='Down Spikes')

    # Plot GT IISs + tolerance windows
    for gt_time in gt_iis_times_sec:
        ax.axvline(gt_time, color='orange', linestyle='-', linewidth=1.5, label='GT IIS')
        ax.fill_between([gt_time - max_detection_offset_sec, gt_time + max_detection_offset_sec],
                        y_min, y_max, color='orange', alpha=0.15, label='Tolerance')

    # Scatter output spikes
    output_y = y_min + (y_max - y_min) * 0.2
    ax.scatter(output_spikes_win_sec, [output_y] * len(output_spikes_win_sec),
               color='purple', marker='o', s=30, label='Output Spikes')

    # Classified spikes (TP green, FP red)
    for idx, (label, spike_time) in enumerate(classified_spikes):
        color = 'green' if label == 'TP' else 'red'
        ax.axvline(spike_time, color=color, linestyle='--', linewidth=1.5, label=label if idx == 0 else None)

    # False Negatives
    detected_gt_indices = {idx for idx, gt_time in enumerate(gt_iis_times_sec)
                           if any(abs(gt_time - spike_time) <= tolerance_sec for _, spike_time in classified_spikes if _ == 'TP')}
    fn_indices = set(range(len(gt_iis_times_sec))) - detected_gt_indices
    for fn_idx in fn_indices:
        ax.axvline(gt_iis_times_sec[fn_idx], color='blue', linestyle=':', linewidth=2, label='FN')

    # Formatting
    ax.set_ylim([y_min, y_max])
    ax.set_xlim(time_sec[0], time_sec[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=':', alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if filename:
        save_path = os.path.join(os.path.dirname(__file__), "live_plots")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, filename), dpi=300)
        print(f"Plot saved to {os.path.join(save_path, filename)}")

    plt.show()
    return fig

prefix="iiss_4b"
adapt=60
id=1
shank=2
# SHIT
plot_iis_live_test(prefix, adapt, id, shank=shank, window=None,
                      title='Live Test IIS Detection', xlabel='Time (s)', ylabel='Amplitude (Z-score)',
                      input_spikes=False, filename=None,
                      tolerance_ms=100, refractory_period_ms=0, jitter_ms=100, max_detection_offset=100)