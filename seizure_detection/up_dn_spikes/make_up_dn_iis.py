import os
import numpy as np
import sys
dir= os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir, '..',))
# sys.path.append(os.path.join(dir, '..', '..',"liset_tk"))
# from liset_tk import liset_tk
# from liset_paper import liset_paper as liset_tk
from liset_seizures import liset_seizures
# from signal_aid import most_active_channel, bandpass_filter
from up_dn_spikes.utils_encoding import *
import random
import json

def make_up_dn_dataset(parent, ids, time_max, downsampled_fs, bandpass, window_size, sample_ratio, scaling_factor, percentile, refractory, overlap,adapt_threshold=False,window=None):
    for id in ids:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_validation")
        if adapt_threshold:
            save_dir = os.path.join(save_dir, f"adapt_{time_max}")
        else:
            save_dir = os.path.join(save_dir, f"adapt_0")
        os.makedirs(save_dir, exist_ok=True)
        dataset_list=os.listdir(parent)
        data_path = os.path.join(parent, dataset_list[id])
        thresholds_all={}
        if window is not None:
            start=window[0]
            numSamples=(window[1]-window[0])*30000 if window[1] != 0 else False
        else:
            start=0
            numSamples=False
        for shank in [1,2,3,4]:   
            liset=liset_seizures(data_path, shank=shank, downsample=downsampled_fs, normalize=True, start=start, verbose=False,numSamples=numSamples)
            liset_threshold=liset_seizures(data_path, shank=shank, downsample=downsampled_fs, normalize=True, start=0, verbose=False,numSamples=
                                                time_max*30000)
            spikified=np.zeros((liset.data.shape[0], liset.data.shape[1], 2))
            filtered=np.zeros((liset.data.shape[0], liset.data.shape[1]))
            IISs_times=np.array(liset.IISs_times)
            seizures=np.array(liset.seizure_times)
            thresholds=[]
            for channel in range(liset.data.shape[1]):
                initial_value=None
                channel_signal = liset.data[:, channel]
                filtered_channel = bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
                filtered[:, channel] = filtered_channel
                if adapt_threshold:
                    thresholds = []
                    step = int(downsampled_fs * overlap)
                    win = int(downsampled_fs * time_max)
                    total_len = len(filtered_channel)
                    for time in range(0,total_len,step):
                        if time<win:
                            threshold_window = filtered_channel[:win]
                        else:
                            threshold_window = filtered_channel[time-win:time]
                        right_edge = min(time + step, total_len)
                        current_window = filtered_channel[time:right_edge]
                        # Compute threshold
                        if percentile:
                            threshold = threshold_percentile(threshold_window, downsampled_fs, window_size, sample_ratio * 100, scaling_factor)
                        else:
                            threshold = calculate_threshold(threshold_window, downsampled_fs, window_size, sample_ratio, scaling_factor)
                        # Spikify
                        threshold=max(threshold,1.2)
                        spikified_window,initial_value = up_down_channel(current_window, threshold, downsampled_fs, refractory,initial_value=initial_value,return_value=True)
                        spikified[time:right_edge, channel, :] = spikified_window
                        thresholds.append(round(threshold,4))
                        
                    # config[dataset]["thresholds"][channel] = thresholds
                else:
                    # Compute threshold
                    threshold_channel_signal = liset_threshold.data[:, channel]
                    threshold_filtered_channel = bandpass_filter(threshold_channel_signal, bandpass=bandpass, fs=liset.fs)
                    threshold_window = threshold_filtered_channel[:int(downsampled_fs * time_max)]
                    if percentile:
                        threshold = threshold_percentile(threshold_window, downsampled_fs, window_size, sample_ratio * 100, scaling_factor)
                    else:
                        threshold = calculate_threshold(threshold_window, downsampled_fs, window_size, sample_ratio, scaling_factor)
                    # Spikify
                    spikified_window = up_down_channel(filtered_channel, threshold, downsampled_fs, refractory,initial_value=None,return_value=False)
                    # if factor > 1:
                    #     downsampled_window, _ = extract_spikes_downsample(spikified_window, factor)
                    # else:
                    #     downsampled_window = spikified_window      
                    spikified[:,channel,:]= spikified_window
                    # config[dataset]["thresholds"][channel] = round(threshold, 4)
                    print(f"Channel {channel} - Threshold: {round(threshold,4)}")
                thresholds.append(round(threshold,4))
            thresholds_all[shank]=thresholds
             # Save Spikified Data            
            save_dataset=os.path.join(save_dir, dataset_list[id])
            os.makedirs(save_dataset, exist_ok=True)
            np.save(os.path.join(save_dataset, f"spikified_{shank}.npy"), spikified)
            np.save(os.path.join(save_dataset, f"filtered_{shank}.npy"), filtered)
            np.save(os.path.join(save_dataset, f"IISs_{shank}.npy"), IISs_times)
            np.save(os.path.join(save_dataset, f"seizures_{shank}.npy"), seizures)
        with open(os.path.join(save_dataset, f"thresholds.json"), 'w') as f:
            json.dump(thresholds_all, f, indent=4)


def make_windows(ids, WINDOW_SIZE,WINDOW_SHIFT,MARGINS,min_spikes=0,fraction=(0,1),min_threshold=1,adapt_threshold=0):
    # Lists to store the final windowed data
    windowed_input_data = []
    windowed_gt = []
    filtered_windows = []
    iiss_ids = []
    
    # --- Statistics Counters ---
    total_windows_count = 0
    skipped_iiss_count = 0
    total_iiss = 0
    would_be_non_iiss = 0
    iiss_too_late = 0
    dataset_iiss_offset = 0
    current_iiss_id = 0

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "up_dn_data", f"adapt_{adapt_threshold}")
    dataset_list=os.listdir(data_dir)

    for id in ids:
        data_path = os.path.join(data_dir, dataset_list[id])
        thresholds_path=os.path.join(data_path, f"thresholds.json")
        with open(thresholds_path, 'r') as f:
            thresholds_all = json.load(f)
        for shank in [1,2,3,4]:   
            spikified=np.load(os.path.join(data_path, f"spikified_{shank}.npy"))
            filtered = np.load(os.path.join(data_path, f"filtered_{shank}.npy"))
            iiss = np.load(os.path.join(data_path, f"IISs_{shank}.npy"))
            thresholds_shank=thresholds_all[str(shank)]
            print("data shape: ", spikified.shape)
            print("iiss shape: ", iiss.shape)
            # EXTRACT WINDOWS
            beginning_step=int(fraction[0]*spikified.shape[0])
            end_step=int(fraction[1]*spikified.shape[0])
            
            iiss_in_fraction=[]
            for iiss_time in iiss:
                if iiss_time <= int(end_step) and iiss_time >= int(beginning_step):
                    adjusted_time = iiss_time - beginning_step
                    iiss_in_fraction.append(adjusted_time)
            iiss_in_fraction = np.array(iiss_in_fraction, dtype=np.int32)

        
            for channel in range(spikified.shape[1]):
                # if thresholds_shank[channel]<min_threshold:
                #     print(f"[WARNING] Channel {channel} in Shank {shank} has a very low threshold ({thresholds_shank[channel]}). Skipping...")
                #     continue
                curr_iiss_id = 0
                spikified_channel = spikified[beginning_step:end_step, channel, :]
                filtered_liset= filtered[beginning_step:end_step, channel]

                for i in range(0, spikified_channel.shape[0], WINDOW_SHIFT):
                    left, right = i, i+WINDOW_SIZE
                    # Get the current input window
                    spikified_window = spikified_channel[left:right, :]
                    filtered_window = filtered_liset[left:right]
                
                    # Increment the total windows count
                    total_windows_count += 1

                
                    # Check if the current window is smaller than the expected size
                    if filtered_window.shape[0] < WINDOW_SIZE or spikified_window.shape[0] < WINDOW_SIZE:
                        # If the current window is smaller than the expected size, break the loop
                        print(f"[WARNING] Current window [{left}, {right}] is smaller than the expected size. Breaking the loop...")
                        break

                    # OPTIMIZATION STEP: Skip windows with no activations - The gradient will be zero 
                    if np.sum(spikified_window) < min_spikes:
                        # print(f"Window [{left}:{right}] has no Input activations. Skipping...")
                        cur_gt_time=[-1, -1]    # Default value for Spike Time (no HFO)
                        if curr_iiss_id < len(iiss_in_fraction):
                            cur_gt_time =  iiss_in_fraction[curr_iiss_id]  
                        if (cur_gt_time >= left) and (cur_gt_time <= right):
                            if cur_gt_time <= right:
                                print(f"[WARNING] Window [{left}:{right}] has a GT event at {cur_gt_time} and only {np.sum(spikified_window)} input activations. Skipping...")
                                # Update the curr_gt_idx to the next GT event
                                skipped_iiss_count += 1
                            # curr_ripple_id += 1
                        continue   
                    
                    curr_gt = -1    # Default value for Spike Time (no HFO)
                    curr_iiss=-1

                    # Check if the current GT event is within the current window
                    while curr_iiss_id < len(iiss_in_fraction) - 1 and iiss_in_fraction[curr_iiss_id] < left:
                        # Ripple ends before the window starts → skip it
                        curr_iiss_id += 1

                    if curr_iiss_id >= len(iiss_in_fraction):
                        curr_iiss_id = len(iiss_in_fraction) - 1

                    cur_gt_time =  iiss_in_fraction[curr_iiss_id]
                    if (cur_gt_time >= left) and (cur_gt_time <= right):
                        '''
                            Check if the current window overlaps with the current GT event
                            The Network may spike in the interval [GT_time[0], GT_time[0] + MEAN_HFO_DURATION + PRED_GT_TOLERANCE]
                            However, we are using an upper limit for the HFO Duration of WINDOW_SIZE.
                            This way, the Ground Truth Timestamps will be clamped uppwards by WINDOW_SIZE - MAX_HFO_DURATION + MEAN_HFO_DURATION
                        '''
                        if cur_gt_time-MARGINS[0]<left:
                            #TODO: Check if the GT event starts before the window starts 
                            # THIS IS DEBUGGING
                            print(f"[WARNING] GT event {cur_gt_time} starts before the window [{left}:{right}]. Skipping...")
                            would_be_non_iiss+=1
                            continue
                        if cur_gt_time + MARGINS[1] <right: # If the GT event is completely within the current window
                            '''The Network should predict the HFO -> Calculate the spike time
                            Let's assume the network should spike at the end of the relevant event. We have no way of knowing
                            the exact end time, so we use the mean duration of the event to calculate the spike time.
                            '''
                            avg_spike_time = cur_gt_time # The network should spike at the end of the relevant event

                            # Subtract the left offset to get the spike time in the current window
                            relative_spike_time = avg_spike_time - left

                            # relative_spike_time//=factor
                            curr_gt = int(relative_spike_time)   # Update the curr_gt value

                            curr_iiss=curr_iiss_id+dataset_iiss_offset
                        else: # Added now - if it does not work remove later
                            iiss_too_late+=1
                            continue    
                    # Append the current window    
                    iiss_ids.append(curr_iiss)
                    windowed_input_data.append(spikified_window)            
                    # Append the current GT Spike Time to the windowed GT
                    windowed_gt.append(curr_gt)
                    filtered_windows.append(filtered_window)
                total_iiss += len(iiss_in_fraction)
            dataset_iiss_offset += len(iiss_in_fraction)
      # Convert to numpy array
    iiss_ids=np.array(iiss_ids,dtype=np.int32)
    filtered_windows=np.array(filtered_windows, dtype=np.float32)
    windowed_input_data = np.array(windowed_input_data)
    windowed_gt = np.array(windowed_gt, dtype=np.float32)
    removed_windows = total_windows_count - windowed_input_data.shape[0]
    # config["adapt_threshold"] = adapt_threshold
    # config["overlap"] = overlap
    print(f"Removed {removed_windows}/{total_windows_count} ({round((removed_windows / total_windows_count)*100, 2)}%) windows with no input activations")
    print(f"Skipped {skipped_iiss_count} IISSs due to no input activations")
    print(f"Total IISSs (theoretical): {total_iiss}")
    print("Windowed Input Data Shape: ", windowed_input_data.shape)
    print("Windowed GT Shape: ", windowed_gt.shape)
    print("Filtered Windows Shape: ", filtered_windows.shape)
    print("TOTAL WINDOWS SKIPPED DUE TO IISS STARTING BEFORE WINDOW: ", would_be_non_iiss)
    print("TOTAL IISSs SKIPPED DUE TO IISS STARTING TOO LATE: ", iiss_too_late)    

    return  windowed_input_data, windowed_gt, filtered_windows, iiss_ids

from collections import defaultdict
def only_some_channels_per_ripple(windows, gt, ripple_ids, top_channels):
    """
    Retains the top N most active windows (based on spike count) per ripple event,
    while keeping all non-HFO windows.

    Args:
        windows (np.ndarray): shape (N, T, 2)
        gt (np.ndarray): shape (N,)
        ripple_ids (np.ndarray): shape (N,)
        top_channels (int): number of most spiking channels to keep per ripple

    Returns:
        filtered_windows, filtered_gt, filtered_ripple_ids
    """
    ripple_groups = defaultdict(list)
    filtered_windows = []
    filtered_gt = []
    filtered_ripple_ids = []

    for idx, (window, label, ripple_id) in enumerate(zip(windows, gt, ripple_ids)):
        if label == -1:
            # Keep non-HFO windows
            filtered_windows.append(window)
            filtered_gt.append(label)
            filtered_ripple_ids.append(ripple_id)
        else:
            # Group HFO windows by ripple ID
            ripple_groups[ripple_id].append((idx, window, label))

    for group in ripple_groups.values():
        # Rank by total spike count
        group_sorted = sorted(group, key=lambda x: np.sum(x[1]), reverse=True)
        top = group_sorted[:top_channels]
        for idx, window, label in top:
            filtered_windows.append(window)
            filtered_gt.append(label)
            filtered_ripple_ids.append(ripple_ids[idx])

    return (
        np.array(filtered_windows),
        np.array(filtered_gt, dtype=np.float32),
        np.array(filtered_ripple_ids)
    )


def drop_linear(score, threshold, inverse=False, max_prob=1.0, multiplier=1.0,decay=3.0):
    # Compute directional difference
    diff = (threshold - score) if inverse else (score - threshold)
    if diff <= 0:
        return 0.0
    prob = min((diff * multiplier) / threshold, 1.0) * max_prob
    return prob

def drop_quadratic(score, threshold, inverse=False, max_prob=1.0,decay=3.0,multiplier=1.0):
    diff = (threshold - score) if inverse else (score - threshold)
    if diff <= 0:
        return 0.0
    x = diff / threshold
    prob = min(x**2, 1.0) * max_prob
    return prob

def drop_exponential(score, threshold, inverse=False, max_prob=1.0, multiplier=1, decay=3.0):
    diff = (threshold - score) if inverse else (score - threshold)
    if diff <= 0:
        return 0.0
    x = diff / threshold
    prob = (1 - np.exp(-decay * x))
    return min(prob, 1.0) * max_prob

def drop_logistic(score, threshold, inverse=False, max_prob=1.0, multiplier=1,decay=10.0):
    diff = (threshold - score) if inverse else (score - threshold)
    # Normalize to [−∞, +∞]
    rel = diff / threshold
    # Apply logistic growth
    prob = max_prob / (1 + np.exp(-decay * rel))
    # Ensure zero when diff ≤ 0
    return prob if diff > 0 else 0.0

def drop_all(score, threshold, inverse=False, max_prob=1.0, multiplier=1,decay=10.0):
    return max_prob


def min_max_spike_threshold_prob(windows, gt, ripple_ids, thresholds, max_prob=1.0,multiplier=1,decay=3.0,drop_fn=drop_linear):
    """
    Filters spike windows based on spike activity thresholds with a probability-based approach.

    Args:
        windows (np.ndarray): shape (N, T, 2)
        gt (np.ndarray): shape (N,)
        ripple_ids (list or np.ndarray): IDs for ripple tracking
        MEAN_DETECTION_OFFSET (int): frames before GT spike to check for activity
        thresholds (tuple): (non_hfo_threshold, hfo_activity_threshold)
        max_prob (float): maximum probability of dropping a window (0.0 to 1.0)

    Returns:
        filtered_windows, filtered_gt, filtered_ripple_ids
    """
    cleaned_windows = []
    cleaned_gt = []
    cleaned_ripple_ids = []
    false_pos = 0
    false_neg = 0
    def drop_prob(score, threshold, inverse=False):
        """Probability increases the more 'wrong' the value is."""
        diff = abs(score - threshold)*multiplier
        ratio = diff / threshold if threshold != 0 else 1
        prob = min(ratio, 1.0) * max_prob
        return 1.0 - prob if inverse else prob
    
    def exp_drop_prob(score,threshold,inverse=False,steepness=5.0):
        if threshold == 0:
            return max_prob  # Avoid division by zero

        # Compute relative difference from threshold
        rel_diff = (score - threshold) / threshold

        # Invert direction if needed (e.g., for HFOs with too few spikes)
        if inverse:
            rel_diff *= -1

        # Compute logistic drop probability
        prob = max_prob / (1 + np.exp(-steepness * rel_diff))
        return prob

    for window, label, id in zip(windows, gt, ripple_ids):
        total_spikes = np.sum(window)

        if label == -1:  # Non-HFO (False Negatives)
            if total_spikes>thresholds[0]:
                prob = drop_fn(total_spikes,thresholds[0],inverse=False,max_prob=max_prob,multiplier=multiplier,decay=decay)
                if random.random() < prob:
                    print(f"False Negative Removed (Prob {prob:.2f}) - Spikes: {total_spikes}")
                    false_neg+=1
                    continue

        else:  # True or False Positive
            spike_time = int(label)
            # pre_spikes = np.sum(window[spike_time - MEAN_DETECTION_OFFSET:])
            if total_spikes  < thresholds[1]:
                prob = drop_fn(total_spikes,thresholds[1],inverse=True,max_prob=max_prob,multiplier=multiplier,decay=decay)
                if random.random() < prob:
                    print(f"False Positive Removed (Prob {prob:.2f}) - Spikes: {total_spikes}")
                    false_pos+=1
                    continue

        cleaned_windows.append(window)
        cleaned_gt.append(label)
        cleaned_ripple_ids.append(id)

    removed = len(windows) - len(cleaned_windows)
    print(f"Removed {removed} windows probabilistically!")
    print("False Positives Removed: ", false_pos)
    print("False Negatives Removed: ", false_neg)
    return np.array(cleaned_windows), np.array(cleaned_gt), np.array(cleaned_ripple_ids)


def iis_window_plot(window_signal, window_spikes, gt, downsampled_fs=1000, detection_margins=[0,0]):
    """
    Visualizes a single window of LFP data with its corresponding 'spikified' input
    and the ground truth label for an Interictal Spike (IIS).

    Args:
        window_signal (np.ndarray): 1D array of the filtered signal for the window.
        window_spikes (np.ndarray): 2D array of shape (n_samples, 2) where column 0
                                    is for UP spikes and column 1 is for DOWN spikes.
        gt (int): The ground truth label. If >= 0, it's the sample index of the
                  IIS peak within the window. If -1, it's a "No IIS" window.
        downsampled_fs (int, optional): The sampling rate of the data. Defaults to 1000.
        detection_margins (list, optional): A [before, after] margin in ms to shade
                                            around the ground truth peak.
                                            Defaults to IIS_DETECTION_MARGINS.
    """
    # Create a time vector for the x-axis in seconds
    time_sec = np.arange(len(window_signal)) / downsampled_fs
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot the filtered LFP signal
    ax.plot(time_sec, window_signal, label='Filtered Signal', color='black', zorder=1)

    # --- Plot the UP and DOWN spikes ---
    
    # Find the y-range for the spike lines to make them look nice
    peak = max(np.max(window_signal), 0.1)
    trough = min(np.min(window_signal), -0.1)
    mean = np.mean(window_signal)
    
    # Find the times where UP spikes occur and plot them
    up_spike_times = time_sec[window_spikes[:, 0] == 1]
    ax.vlines(up_spike_times, ymin=mean, ymax=peak, color='red', 
              label='UP Spikes', linewidth=0.8, alpha=0.8, zorder=2)

    # Find the times where DOWN spikes occur and plot them
    down_spike_times = time_sec[window_spikes[:, 1] == 1]
    ax.vlines(down_spike_times, ymin=trough, ymax=mean, color='blue',
              label='DOWN Spikes', linewidth=0.8, alpha=0.8, zorder=2)

    # --- Plot the ground truth (gt) if an IIS exists in this window ---
    if gt >= 0:
        # Convert the ground truth sample index to time in seconds
        gt_time_sec = gt / downsampled_fs
        
        # Plot a prominent dashed line at the exact GT peak
        ax.axvline(gt_time_sec, ymin=trough * 1.2, ymax=peak * 1.2, color='green', 
                   linestyle="--", label='Ground Truth Peak (IIS)', zorder=3)
        
        # Shade the detection margin around the GT peak
        margin_start_sec = gt_time_sec - (detection_margins[0] / 1000.0)
        margin_end_sec = gt_time_sec + (detection_margins[1] / 1000.0)
        ax.fill_between([margin_start_sec, margin_end_sec], trough * 1.2, peak * 1.2, 
                        color='green', alpha=0.15, label='Detection Margin', zorder=0)

    # --- Formatting ---
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (Z-score)')
    ax.set_title(f"Window Visualization - {'IIS Event' if gt >= 0 else 'No IIS'}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    plt.show()
