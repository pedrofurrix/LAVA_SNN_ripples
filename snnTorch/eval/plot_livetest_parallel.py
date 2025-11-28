import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
import json
import pickle
from typing import List, Dict, Any


def plot_livetest_channels_with_clusters(prefix, parent_dir, identifier, spikes_path, window=None,
                                         title='Live Test Data', xlabel='Time (s)', ylabel='Value',
                                         dataset=0, input=True, filename=None, tolerance: int = None,
                                         max_detection_offset: int = 80, information_dataset=None,
                                         save: bool = False):
    """
    Plot spikes and detection clusters per channel.

    Notes
    -----
    - `tolerance` and `max_detection_offset` are in milliseconds (ms). The
      plotting function expects ms values because `process_network_results`
      operates in ms. The default `max_detection_offset` is 80 ms.
    """

    # ----------------------------------------------------------
    # Load raw data
    # ----------------------------------------------------------
    data_dir = os.path.join(parent_dir, "extract_Nripples", "train_pedro",
                            "dataset_up_down", str(identifier))

    datasets = os.listdir(data_dir)
    if "config.json" in datasets:
        datasets.remove("config.json")

    dataset_name = datasets[dataset]

    spikes = np.load(os.path.join(data_dir, dataset_name, 'spike_data.npy'))
    gt = np.load(os.path.join(data_dir, dataset_name, 'ripples.npy'))
    data = np.load(os.path.join(data_dir, dataset_name, 'filtered_data.npy'))

    # ----------------------------------------------------------
    # Compute information_dataset (clusters etc.) if not provided
    # ----------------------------------------------------------
    if information_dataset is None:
        information_dataset = process_network_results(
            data_dir,
            spikes_path,
            tolerance=20,
            padding=100,
            refractory_period=0,
            max_detection_offset=max_detection_offset,
            jitter=100,
            seed=None,
            min_channels=3,
            num_channels=8,
            save=True
        )

    info = information_dataset[dataset_name]
    clusters = info["clusters"]
    per_channel_classified = info["per_channel_classified_spikes"]
    undetected_ripples = info["undetected_ripples"]
    ripple_hits = info["ripple_hits"]
    filtered_spikes = info["filtered_spikes"]

    num_channels = len(filtered_spikes)

    # ----------------------------------------------------------
    # Windowing
    # ----------------------------------------------------------
    if window is not None:
        start_sec, end_sec = window
        start = int(start_sec * 1000)
        end = int(end_sec * 1000)
        end = min(end, data.shape[0])
    else:
        start, end = 0, data.shape[0]

    t_axis = np.arange(start, end) / 1000.0
    data = data[start:end, :]
    spikes = spikes[start:end, :]

    # Filter GT ripples into window (in seconds)
    gt_sec = gt / 1000.0
    gt_sec = gt_sec[(gt_sec[:, 1] >= start/1000.0) & (gt_sec[:, 0] < end/1000.0)]

    # Prepare per-channel spike lists (convert ms->s)
    all_spikes_sec = []
    for ch in range(num_channels):
        arr = np.array(filtered_spikes[ch]) if ch < len(filtered_spikes) else np.array([])
        arr = arr[(arr >= start) & (arr < end)]
        all_spikes_sec.append(arr / 1000.0)

    # Precompute cluster centers in seconds and their labels
    cluster_times_tp = []
    cluster_times_fp = []
    for cl in clusters:
        center_s = cl['center'] / 1000.0
        if cl['label'] == 'TP':
            cluster_times_tp.append(center_s)
        elif cl['label'] == 'FP':
            cluster_times_fp.append(center_s)

    # FN ripples — convert to center times (seconds)
    fn_centers = []
    for ridx in undetected_ripples:
        if ridx < len(gt_sec):
            r0, r1 = gt_sec[ridx]
            fn_centers.append((r0 + r1) / 2.0)

    # Plotting
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 1.5 * num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]

    first_axis = True
    for ch in range(num_channels):
        ax = axes[ch]
        ax.plot(t_axis, data[:, ch], color="black", alpha=0.4)

        y_max = np.max(data[:, ch]) * 1.2
        y_min = np.min(data[:, ch]) * 1.2
        spike_y = y_min + 0.08 * (y_max - y_min)

        ch_spks = all_spikes_sec[ch]
        if len(ch_spks) > 0:
            ax.scatter(ch_spks, [spike_y] * len(ch_spks), color='purple', s=18, marker='o',
                       label='All spikes' if first_axis else None)

        for i, t in enumerate(cluster_times_tp):
            ax.axvline(x=t, color='green', linestyle='--', linewidth=1.0,
                       label='TP cluster' if first_axis and i == 0 else None)

        for i, t in enumerate(cluster_times_fp):
            ax.axvline(x=t, color='red', linestyle='--', linewidth=1.0,
                       label='FP cluster' if first_axis and i == 0 else None)

        for i, t in enumerate(fn_centers):
            ax.axvline(x=t, color='blue', linestyle='--', linewidth=1.0,
                       label='FN ripple' if first_axis and i == 0 else None)

        for i, (r0, r1) in enumerate(gt_sec):
            ax.add_patch(Rectangle((r0, y_min), r1 - r0, y_max - y_min,
                                   alpha=0.12, color='yellow',
                                   label='GT Ripple' if first_axis and i == 0 else None))

        if input:
            up = np.where(spikes[:, ch, 0] == 1)[0] / 1000.0 + start/1000.0
            down = np.where(spikes[:, ch, 1] == 1)[0] / 1000.0 + start/1000.0
            if up.size:
                ax.vlines(up, y_max - 0.05 * (y_max - y_min), y_max, color='red', alpha=0.3,
                          label='Up spikes' if first_axis else None)
            if down.size:
                ax.vlines(down, y_min, y_min + 0.05 * (y_max - y_min), color='blue', alpha=0.3,
                          label='Down spikes' if first_axis else None)

        ax.set_ylabel(f"Ch {ch+1}", fontsize=12)
        ax.set_ylim([y_min, y_max])
        first_axis = False

    axes[-1].set_xlabel(xlabel, fontsize=14)
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc='upper right')

    if filename:
        save_path = os.path.join(os.path.dirname(__file__), "live_plots")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, filename), dpi=300)

    plt.show()
    return fig


def process_network_results(
    dataset_path: str,
    spikes_path: str,
    tolerance: int = 20,
    padding: int = 100,
    refractory_period: int = 0,
    max_detection_offset: int = 100,
    jitter: int = 100,
    seed: int = None,
    min_channels: int = 3,
    num_channels: int = 8,
    save: bool = False
) -> Dict[str, Any]:
    """
    Process network spikes and compute multi-channel cluster-based TP/FP metrics.

    Returns a dictionary mapping dataset folder name -> computed info (clusters,
    per-channel classifications, ripple hits, undetected ripples, stats, etc.).
    All time units in this function are milliseconds (ms).
    """

    spikes = np.load(spikes_path, allow_pickle=True)
    datasets = sorted(os.listdir(dataset_path))
    if "config.json" in datasets:
        datasets.remove("config.json")

    information_dataset = {}

    for dataset_id, dataset in enumerate(datasets):
        print(f"[process_network_results] Processing dataset: {dataset}")
        data_path = os.path.join(dataset_path, dataset)

        spike_data_file = os.path.join(data_path, "spike_data.npy")
        ripples_file = os.path.join(data_path, "ripples.npy")

        if not os.path.isfile(spike_data_file) or not os.path.isfile(ripples_file):
            print(f"  ⚠️ Missing files in {data_path}. Skipping.")
            continue

        data = np.load(spike_data_file)
        ripples = np.load(ripples_file)  # shape [N,2] in ms [start, end]

        # Optional seed window cropping
        if seed is not None:
            time_duration = 60  # seconds
            window = np.arange(seed * time_duration * 1000,
                               (time_duration + seed * time_duration) * 1000, 1)
            data = data[window, :]
            ripples_window = []
            for ripple in ripples:
                if ripple[1] >= window[0] and ripple[0] <= window[-1]:
                    ripples_window.append(ripple)
            ripples = np.array(ripples_window) - window[0]
        else:
            ripples = ripples

        dataset_spikes = spikes[dataset_id * num_channels:(dataset_id + 1) * num_channels, :]

        # Per-channel refractory filtering (ms)
        filtered_spikes = []
        for ch_spikes in dataset_spikes:
            ch_spikes = np.asarray(ch_spikes)
            if ch_spikes.size == 0:
                filtered_spikes.append(np.array([], dtype=int))
                continue
            ch_spikes = np.sort(ch_spikes[~np.isnan(ch_spikes)])
            if ch_spikes.size == 0:
                filtered_spikes.append(np.array([], dtype=int))
                continue
            if refractory_period <= 0:
                filtered_spikes.append(ch_spikes.astype(int))
            else:
                valid = [int(ch_spikes[0])]
                for sp in ch_spikes[1:]:
                    if int(sp) - valid[-1] >= refractory_period:
                        valid.append(int(sp))
                filtered_spikes.append(np.array(valid, dtype=int))

        # Build all_spikes list (time_ms, channel)
        all_spikes = []
        for ch_idx, ch_spk in enumerate(filtered_spikes):
            for t in ch_spk:
                all_spikes.append((int(t), ch_idx))
        all_spikes.sort(key=lambda x: x[0])

        # Cluster spikes across channels using jitter (ms)
        spike_groups = []
        for t, ch in all_spikes:
            placed = False
            for g in spike_groups:
                if abs(t - g['time']) <= jitter:
                    g['channels'].add(ch)
                    g['spike_times'].append((t, ch))
                    g['time'] = int(round(np.mean([st for st, _ in g['spike_times']])))
                    placed = True
                    break
            if not placed:
                spike_groups.append({'time': int(t), 'channels': set([ch]), 'spike_times': [(int(t), ch)]})

        valid_detection_ranges = [
            (int(r[0]) - tolerance, int(r[0]) + max_detection_offset + tolerance + padding)
            for r in ripples
        ]

        def cluster_is_in_any_ripple(center_time: int) -> List[int]:
            matches = []
            for ridx, rstart_end in enumerate(ripples):
                window_start = int(rstart_end[0]) - tolerance
                window_end = int(rstart_end[0]) + max_detection_offset + tolerance
                if window_start <= center_time <= window_end:
                    matches.append(ridx)
            return matches

        clusters = []
        for gi, g in enumerate(spike_groups):
            cluster_dict = {
                'center': int(g['time']),
                'channels': sorted(list(g['channels'])),
                'spike_times': list(g['spike_times']),
                'n_channels': len(g['channels']),
                'label': 'IGNORE',
                'matched_ripples': [],
            }
            if cluster_dict['n_channels'] < min_channels:
                cluster_dict['label'] = 'IGNORE'
            else:
                matched = cluster_is_in_any_ripple(cluster_dict['center'])
                if len(matched) > 0:
                    cluster_dict['label'] = 'TP'
                    cluster_dict['matched_ripples'] = matched
                else:
                    if any(valid_range[0] <= cluster_dict['center'] <= valid_range[1] for valid_range in valid_detection_ranges):
                        cluster_dict['label'] = 'IGNORE'
                    else:
                        cluster_dict['label'] = 'FP'
            clusters.append(cluster_dict)

        ripple_hits = defaultdict(list)
        latencies = []
        for cidx, c in enumerate(clusters):
            if c['label'] != 'TP':
                continue
            for ridx in c['matched_ripples']:
                ripple_hits[ridx].append(cidx)
                latency = c['center'] - int(ripples[ridx, 0])
                latencies.append(latency)

        undetected_ripples = [ridx for ridx in range(len(ripples)) if len(ripple_hits.get(ridx, [])) == 0]

        TP_count = sum(1 for c in clusters if c['label'] == 'TP')
        FP_count = sum(1 for c in clusters if c['label'] == 'FP')
        FN_count = len(undetected_ripples)

        per_channel_classified_spikes = [[] for _ in range(num_channels)]
        for cidx, c in enumerate(clusters):
            if c['label'] == 'IGNORE':
                continue
            for t, ch in c['spike_times']:
                per_channel_classified_spikes[ch].append((c['label'], int(t), cidx))

        stats = {
            'TP': int(TP_count),
            'FP': int(FP_count),
            'FN': int(FN_count),
            'num_clusters': int(len(clusters)),
            'num_ripples': int(len(ripples)),
        }

        information_dataset[dataset] = {
            'clusters': clusters,
            'per_channel_classified_spikes': per_channel_classified_spikes,
            'ripple_hits': dict(ripple_hits),
            'undetected_ripples': undetected_ripples,
            'latencies': latencies,
            'filtered_spikes': filtered_spikes,
            'fp_spikes_by_channel': {ch: list(v) for ch, v in ((i, [t for (lab, t, idx) in per_channel_classified_spikes[i] if lab == 'FP']) for i in range(num_channels))},
            'stats': stats,
        }

    if save:
        save_path = os.path.join(os.path.dirname(__file__), "live_plots")
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "information_dataset.pkl")
        with open(save_file, 'wb') as f:
            pickle.dump(information_dataset, f)
        print(f"[process_network_results] Saved information_dataset to {save_file}")

    return information_dataset


if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,os.pardir))
    prefix="updnb4ds_100_7"
    identifier="30000_1000_100"
    adapt=20
    test_ds_b4=False
    spikes_name=f"{prefix}_adapt{adapt}" if adapt > 0 else prefix
    spikes_name = f"{spikes_name}_dsb4" if test_ds_b4 else spikes_name
    spikes_path= os.path.join(os.path.dirname(os.path.abspath(__file__)),"spikes",f"{spikes_name}_spikes.npy")


    save_path = os.path.join(os.path.dirname(__file__), "live_plots")
    save_file = os.path.join(save_path, "information_dataset.pkl")
    with open(save_file, 'rb') as f:
            information_dataset = pickle.load(f) 
    plot_livetest_channels_with_clusters(prefix, parent_dir, identifier, 
                                         spikes_path,
                                         window=None,information_dataset=information_dataset,
                                         title='Live Test Data',
                                         xlabel='Time (s)', ylabel='Value',
                                         dataset=2, input=False,
                                         filename=None,
                                         tolerance=20,
                                         max_detection_offset=80)