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
    seizures=False,
):

    spikes = np.load(spikes_path, allow_pickle=True)

    datasets = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    information_dataset = {}

    for dataset_id, dataset in enumerate(datasets):
        print(f"Processing dataset: {dataset}")
        data_path = os.path.join(dataset_path, dataset)
        shank_information = {}

        for shank in [1]:
            # Load ground truth Interictal Spikes (IIS) and network-detected spikes for the shank
            try:
                iiss = np.load(os.path.join(data_path, f"IISs_{shank}.npy"))
                seizures = np.load(os.path.join(data_path, f"seizures_{shank}.npy"))
                # The file contains spikes for each shank. The structure is assumed to be flat.
                shank_spikes_all_channels = spikes[dataset_id, :]
            except FileNotFoundError:
                print(f"  - Warning: Data for shank {shank} not found in {dataset}. Skipping.")
                continue

            # Since you use one channel per shank, we extract the first (and only) channel's spikes.
            if len(shank_spikes_all_channels) == 0:
                 spike_times_for_shank = np.array([])
            else:
                 spike_times_for_shank = shank_spikes_all_channels
            
            # 1. Apply refractory period to filter out rapid, successive detections
            sorted_spikes = np.sort(spike_times_for_shank)
            if len(sorted_spikes) == 0:
                filtered_shank_spikes = np.array([])
            else:
                valid_spikes = [sorted_spikes[0]]
                for spike in sorted_spikes[1:]:
                    if spike - valid_spikes[-1] >= refractory_period:
                        valid_spikes.append(spike)
                filtered_shank_spikes = np.array(valid_spikes)

            # 2. Identify True Positives (TPs) and calculate latencies
            detected_iiss_indices = set()
            latencies = []
            for idx, iis_time in enumerate(iiss):
                valid_window_start = iis_time - max_detection_offset
                valid_window_end = iis_time + max_detection_offset +tolerance
                
                # Find all spikes within the valid window for this IIS event
                spikes_in_window = filtered_shank_spikes[
                    (filtered_shank_spikes >= valid_window_start) & (filtered_shank_spikes <= valid_window_end)
                ]

                if len(spikes_in_window) > 0:
                    first_spike = spikes_in_window[0]
                    latency = first_spike - iis_time
                    latencies.append(latency)
                    detected_iiss_indices.add(idx)

            # 3. Identify False Positives (FPs)
            # An FP is a spike that does not fall within any valid detection window of any true IIS event.
            valid_detection_ranges = [
                (iis_time - max_detection_offset, iis_time + max_detection_offset + tolerance + padding) for iis_time in iiss
            ]

            def is_tp(spike_time, valid_ranges):
                return any(start <= spike_time <= end for start, end in valid_ranges)

            fp_spikes = [
                spike_time for spike_time in filtered_shank_spikes if not is_tp(spike_time, valid_detection_ranges)
            ]

            # 3b. Merge raw FPs into FP events using 'jitter'
            fp_event_groups = {}
            for spike_time in sorted(fp_spikes): # Sorting is more efficient here
                matched_time = None
                # Only need to look backwards for a match due to sorting
                for t in range(int(spike_time), int(spike_time - jitter - 1), -1):
                    if t in fp_event_groups:
                        matched_time = t
                        break
                if matched_time is not None:
                    # This spike belongs to an existing FP event group
                    pass # No action needed, it's just part of the cluster
                else:
                    # This is the first spike of a new FP event group
                    fp_event_groups[int(spike_time)] = True

            merged_fp_spikes = sorted(fp_event_groups.keys())

            # 4. Store all calculated metrics for the shank
            shank_results = {
                "num_iiss": len(iiss),
                "TP": len(detected_iiss_indices),
                "num_filtered_spikes": len(filtered_shank_spikes),
                "num_fps": len(fp_spikes),
                "fp_spikes": fp_spikes,
                "latencies": latencies,
                # --- METRICS YOU ASKED FOR ---
                "FP": len(merged_fp_spikes),
                "merged_fp_spikes": merged_fp_spikes,
                "FN": len(iiss) - len(detected_iiss_indices),
                # --- -------------------- ---
                "detected_iiss_indices": sorted(list(detected_iiss_indices)),
                "undetected_iiss_indices": sorted([i for i in range(len(iiss)) if i not in detected_iiss_indices]),
            }
            shank_information[f"shank_{shank}"] = shank_results
            print(f"  - Shank {shank}: TP={shank_results['TP']}, FP={shank_results['FP']}, FN={shank_results['FN']}, Num IIS={shank_results['num_iiss']}, Num Filtered Spikes={shank_results['num_filtered_spikes']}")
        information_dataset[dataset] = shank_information

    return information_dataset