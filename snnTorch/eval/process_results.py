import os
import json
import numpy as np
from collections import defaultdict




def process_network_results(
    dataset_path,
    spikes_path,
    tolerance=20,
    padding=100,
    refractory_period=0,
    max_detection_offset=100,
    jitter=100,
    seed=None,
):

    spikes = np.load(spikes_path, allow_pickle=True)

    datasets = os.listdir(dataset_path)
    if "config.json" in datasets:
        datasets.remove("config.json")

    information_dataset = {}

    for dataset_id, dataset in enumerate(datasets):
        print(f"Processing dataset: {dataset}")
        data_path = os.path.join(dataset_path, dataset)
        data = np.load(os.path.join(data_path, "spike_data.npy"))
        ripples = np.load(os.path.join(data_path, "ripples.npy"))

        if seed is not None:
            time_duration = 60
            window = np.arange(seed * time_duration * 1000, (time_duration + seed * time_duration) * 1000, 1)
            data = data[window, :]
            ripples_window = []
            for ripple in ripples:
                if ripple[1] >= window[0] and ripple[0] <= window[-1]:
                    ripples_window.append(ripple)
            ripples = np.array(ripples_window) - window[0] - tolerance
        else:
            ripples = ripples 

        num_channels = 8
        dataset_spikes = spikes[dataset_id * num_channels:(dataset_id + 1) * num_channels, :]

        # Remove refractory period violations
        filtered_spikes = []
        for ch_spikes in dataset_spikes:
            sorted_spikes = np.sort(ch_spikes)
            if len(sorted_spikes) == 0:
                filtered_spikes.append(np.array([]))
                continue
            valid = [sorted_spikes[0]]
            for spike in sorted_spikes[1:]:
                if spike - valid[-1] >= refractory_period:
                    valid.append(spike)
            filtered_spikes.append(np.array(valid))

        ripples_start = ripples[:, 0] - tolerance
        ripple_hits = {}
        undetected_ripples = []
        latencies = []

        for idx, ripple_time in enumerate(ripples_start):
            detected_channels = set()
            for ch in range(num_channels):
                valid_window_start = ripple_time - tolerance
                valid_window_end = ripple_time + max_detection_offset + tolerance
                ch_spikes = filtered_spikes[ch]
                in_window = np.any((ch_spikes >= valid_window_start) & (ch_spikes <= valid_window_end))
                if in_window:
                    first_spike = ch_spikes[ch_spikes >= valid_window_start][0]
                    latency = first_spike - ripple_time
                    latencies.append(latency)
                    detected_channels.add(ch)
            ripple_hits[idx] = detected_channels
            if not detected_channels:
                undetected_ripples.append(idx)

        # Collect false positives
        valid_detection_ranges = [
            (r[0] - tolerance, r[0] + max_detection_offset + tolerance + padding) for r in ripples
        ]

        def is_tp(spike_time, valid_ranges):
            return any(start <= spike_time <= end for start, end in valid_ranges)

        fp_spikes_by_channel = defaultdict(list)
        for ch in range(num_channels):
            for spike_time in filtered_spikes[ch]:
                if not is_tp(spike_time, valid_detection_ranges):
                    fp_spikes_by_channel[ch].append(spike_time)

        jitter = jitter if jitter is not None else (max_detection_offset + tolerance)
        fp_timing_to_channels = defaultdict(set)
        for ch, spike_list in fp_spikes_by_channel.items():
            for spike_time in spike_list:
                matched_time = None
                for t in range(int(spike_time - jitter), int(spike_time + jitter + 1)):
                    if t in fp_timing_to_channels:
                        matched_time = t
                        break
                if matched_time is not None:
                    fp_timing_to_channels[matched_time].add(ch)
                else:
                    fp_timing_to_channels[spike_time].add(ch)

        information_dataset[dataset] = {
            "ripple_hits": ripple_hits,
            "undetected_ripples": undetected_ripples,
            "filtered_spikes": filtered_spikes,
            "fp_spikes_by_channel": fp_spikes_by_channel,
            "fp_timing_to_channels": dict(fp_timing_to_channels),
            "latencies": latencies,
        }

    return information_dataset