import os
import numpy as np
# from liset_tk import liset_tk
from liset_paper import liset_paper as liset_tk
from signal_aid import most_active_channel, bandpass_filter
from extract_Nripples.utils_encoding import *

def make_windows(parent,config,time_max,downsampled_fs,bandpass,window_size,sample_ratio, scaling_factor, 
                     refractory,WINDOW_SHIFT, WINDOW_SIZE,MEAN_DETECTION_OFFSET,MAX_DETECTION_OFFSET,factor):
# Split the Input Data and Ground Truth into Windows
    windowed_input_data = []    # Input Data Windows
    windowed_gt = []        # Ground Truth Windows (spike time if HFO, -1 if no HFO)
    filtered_windows=[] 
    ripple_ids=[]

    total_windows_count = 0
    skipped_hfo_count = 0   # Counts the nº of skipped HFOs due to no input activations
    total_hfos=0
    
    # curr_ripple_times = ripples_concat[curr_ripple_id]    # Get the GT times for the current sEEG source

    # LOAD THE DATA
    # Iterate over the datasets
    dataset_id = 0
    for k,dataset in enumerate(os.listdir(parent)):
        dataset_path = os.path.join(parent, dataset)
        liset= liset_tk(dataset_path, shank=1, downsample=downsampled_fs, verbose=False)

        ripples=np.array(liset.ripples_GT)
        spikified=np.zeros((liset.data.shape[0], liset.data.shape[1], 2))
        filtered=np.zeros((liset.data.shape[0], liset.data.shape[1]))
        thresholds = []
        downsampled=np.zeros((liset.data.shape[0]//factor, liset.data.shape[1], 2))
        print("data shape: ", liset.data.shape)
        print("ripples shape: ", ripples.shape)
        # print("Head of data_concat: ", data[:10][:])
        # print("Head of ripples_concat: ", ripples[:10])
        ripples = ripples[np.argsort(ripples[:, 0])]
        # print(ripples[:10][:])
        config[dataset] = {}
        config[dataset]["thresholds"] = {}
        for channel in range(liset.data.shape[1]):
            channel_signal = liset.data[:time_max*liset.fs, channel]
            filtered_signal=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
            thresholds.append(round(calculate_threshold(filtered_signal,liset.fs,window_size,sample_ratio,scaling_factor),4))
            config[dataset]["thresholds"][channel]=thresholds[channel]
            if thresholds[channel] > 0.1:
                channel_signal = liset.data[:, channel]
                curr_ripple_id = 0  # Keep track of the current GT event index since it is monotonically increasing the timestep
                filtered_liset=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
                spikified[:, channel, :]=up_down_channel(filtered_liset,thresholds[channel],liset.fs,refractory)
                downsampled[:,channel,:]=extract_spikes_downsample(spikified[:,channel,:],factor )
                
                for i in range(0, liset.data.shape[0], WINDOW_SHIFT*factor):
                    left, right = i, i+WINDOW_SIZE*factor
                    # Get the current input window
                    curr_window = spikified[left:right, channel, :]
                    downsampled_window= downsampled[left//factor:right//factor, channel, :]
                    filtered_window = filtered_liset[left:right]
                    
                    # Increment the total windows count
                    total_windows_count += 1
                    # Check if the current window is smaller than the expected size
                    if downsampled_window.shape[0] < WINDOW_SIZE:
                        # If the current window is smaller than the expected size, break the loop
                        print(f"[WARNING] Current window [{left}, {right}] is smaller than the expected size. Breaking the loop...")
                        break

                    # OPTIMIZATION STEP: Skip windows with no activations - The gradient will be zero 
                    if np.sum(downsampled_window) == 0:
                        # print(f"Window [{left}:{right}] has no Input activations. Skipping...")
                        cur_gt_time=[-1, -1]    # Default value for Spike Time (no HFO)
                        if curr_ripple_id < ripples.shape[0]:
                            cur_gt_time = ripples[curr_ripple_id]  
                        if (cur_gt_time[1] >= left) and (cur_gt_time[0] <= right):
                            if cur_gt_time[1] <= right:
                                print(f"[WARNING] Window [{left}:{right}] has a GT event at {cur_gt_time} and NO Input activations. Skipping...")
                                # Update the curr_gt_idx to the next GT event
                                skipped_hfo_count += 1
                            curr_ripple_id += 1
                        continue   
                    
                    '''
                    Check if there is a GT event in the current window
                    '''

                    curr_gt = -1    # Default value for Spike Time (no HFO)
                    curr_ripple=-1
                    # Check if the current GT event is within the current window
                    while curr_ripple_id<ripples.shape[0]-1 and ripples[curr_ripple_id][1] < left:
                        # Ripple ends before the window starts → skip it
                        curr_ripple_id += 1
                    
                    if curr_ripple_id >= ripples.shape[0]:
                        curr_ripple_id=ripples.shape[0]-1
                
                    cur_gt_time = ripples[curr_ripple_id]
                    

                    if (cur_gt_time[1] >= left) and (cur_gt_time[0] <= right):
                        '''
                            Check if the current window overlaps with the current GT event
                            The Network may spike in the interval [GT_time[0], GT_time[0] + MEAN_HFO_DURATION + PRED_GT_TOLERANCE]
                            However, we are using an upper limit for the HFO Duration of WINDOW_SIZE.
                            This way, the Ground Truth Timestamps will be clamped uppwards by WINDOW_SIZE - MAX_HFO_DURATION + MEAN_HFO_DURATION
                        '''
                        if cur_gt_time[1] <= right and cur_gt_time[0]>=left: # If the GT event is completely within the current window
                            '''The Network should predict the HFO -> Calculate the spike time
                            Let's assume the network should spike at the end of the relevant event. We have no way of knowing
                            the exact end time, so we use the mean duration of the event to calculate the spike time.
                            '''
                            avg_spike_time = cur_gt_time[0] +  MEAN_DETECTION_OFFSET*factor # The network should spike at the end of the relevant event
                            
                            # Subtract the left offset to get the spike time in the current window
                            relative_spike_time = avg_spike_time - left
                            
                            if relative_spike_time//factor > WINDOW_SIZE:
                                # If the spike time is greater than the window size, we want to skip the window
                                print(f"[WARNING] Spike time {relative_spike_time//factor} is greater than the window size {WINDOW_SIZE}. Adjusting...")
                                relative_spike_time= cur_gt_time[1]-left

                            relative_spike_time//=factor
                            curr_gt = relative_spike_time   # Update the curr_gt value
                            curr_ripple=curr_ripple_id+dataset_id
                        elif cur_gt_time[1] > right or cur_gt_time[0] < left:
                            continue
                            # If the GT event is not completely within the current window, we want to skip the window
                    
                    # Append the current window    
                    ripple_ids.append(curr_ripple)
                    windowed_input_data.append(downsampled_window)            
                    # Append the current GT Spike Time to the windowed GT
                    windowed_gt.append(curr_gt)
                    filtered_windows.append(filtered_window)
                total_hfos+=ripples.shape[0]
            else:
                print(f"[WARNING] Channel {channel} has a very low threshold. Skipping...")
        dataset_id+=liset.ripples_GT.shape[0]
    # Convert to numpy array
    ripple_ids=np.array(ripple_ids,dtype=np.int32)
    filtered_windows=np.array(filtered_windows, dtype=np.float32)
    windowed_input_data = np.array(windowed_input_data)
    windowed_gt = np.array(windowed_gt, dtype=np.float32)
    removed_windows = total_windows_count - windowed_input_data.shape[0]
    print(f"Removed {removed_windows}/{total_windows_count} ({round((removed_windows / total_windows_count)*100, 2)}%) windows with no input activations")
    print(f"Skipped {skipped_hfo_count} HFOs due to no input activations")
    print(f"Total HFOs (theoretical): {total_hfos}")
    print("Windowed Input Data Shape: ", windowed_input_data.shape)
    print("Windowed GT Shape: ", windowed_gt.shape)
    print("Filtered Windows Shape: ", filtered_windows.shape)
    return  windowed_input_data, windowed_gt, filtered_windows, ripple_ids, config


def make_windows_mesquita(parent,config,time_max,downsampled_fs,bandpass,window_size,sample_ratio, scaling_factor, 
                     refractory,WINDOW_SHIFT, WINDOW_SIZE,MEAN_DETECTION_OFFSET,MAX_DETECTION_OFFSET,factor):
# Split the Input Data and Ground Truth into Windows
    windowed_input_data = []    # Input Data Windows
    windowed_gt = []        # Ground Truth Windows (spike time if HFO, -1 if no HFO)
    filtered_windows=[] 
    ripple_ids=[]

    total_windows_count = 0
    skipped_hfo_count = 0   # Counts the nº of skipped HFOs due to no input activations
    total_hfos=0
    # curr_ripple_times = ripples_concat[curr_ripple_id]    # Get the GT times for the current sEEG source

    # LOAD THE DATA
    # Iterate over the datasets
    dataset_id = 0
    for dataset in os.listdir(parent):
        dataset_path = os.path.join(parent, dataset)
        liset= liset_tk(dataset_path, shank=3, downsample=downsampled_fs, verbose=False)

        ripples=np.array(liset.ripples_GT)
        spikified=np.zeros((liset.data.shape[0], liset.data.shape[1], 2))
        filtered=np.zeros((liset.data.shape[0], liset.data.shape[1]))
        thresholds = []
        downsampled=np.zeros((liset.data.shape[0]//factor, liset.data.shape[1], 2))
        print("data shape: ", liset.data.shape)
        print("ripples shape: ", ripples.shape)
        # print("Head of data_concat: ", data[:10][:])
        # print("Head of ripples_concat: ", ripples[:10])
        ripples = ripples[np.argsort(ripples[:, 0])]
        # print(ripples[:10][:])
        config[dataset] = {}
        config[dataset]["thresholds"] = {}
        for channel in range(liset.data.shape[1]):
            channel_signal = liset.data[:time_max*liset.fs, channel]
            filtered_signal=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
            thresholds.append(round(calculate_threshold(filtered_signal,liset.fs,window_size,sample_ratio,scaling_factor),4))
            config[dataset]["thresholds"][channel]=thresholds[channel]
            if thresholds[channel] > 0.1:
                channel_signal = liset.data[:, channel]
                curr_ripple_id = 0     # Keep track of the current GT event index since it is monotonically increasing the timestep
                filtered_liset=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
                spikified[:, channel, :]=up_down_channel(filtered_liset,thresholds[channel],liset.fs,refractory)
                downsampled[:,channel,:]=extract_spikes_downsample(spikified[:,channel,:],factor )
                
                for i in range(0, liset.data.shape[0], WINDOW_SHIFT*factor):
                    left, right = i, i+WINDOW_SIZE*factor
                    # Get the current input window
                    curr_window = spikified[left:right, channel, :]
                    downsampled_window= downsampled[left//factor:right//factor, channel, :]
                    filtered_window = filtered_liset[left:right]
                    
                    # Increment the total windows count
                    total_windows_count += 1
                    # Check if the current window is smaller than the expected size
                    if downsampled_window.shape[0] < WINDOW_SIZE:
                        # If the current window is smaller than the expected size, break the loop
                        print(f"[WARNING] Current window [{left}, {right}] is smaller than the expected size. Breaking the loop...")
                        break

                    # OPTIMIZATION STEP: Skip windows with no activations - The gradient will be zero 
                    if np.sum(downsampled_window) == 0:
                        # print(f"Window [{left}:{right}] has no Input activations. Skipping...")
                        cur_gt_time=[-1, -1]    # Default value for Spike Time (no HFO)
                        if curr_ripple_id < ripples.shape[0]:
                            cur_gt_time = ripples[curr_ripple_id]  
                        if (cur_gt_time[1] >= left) and (cur_gt_time[0] <= right):
                            if cur_gt_time[1] <= right:
                                print(f"[WARNING] Window [{left}:{right}] has a GT event at {cur_gt_time} and NO Input activations. Skipping...")
                                # Update the curr_gt_idx to the next GT event
                                skipped_hfo_count += 1
                            curr_ripple_id += 1
                        continue   
                    
                    '''
                    Check if there is a GT event in the current window
                    '''

                    curr_gt = -1    # Default value for Spike Time (no HFO)
                    curr_ripple=-1
                    # Check if the current GT event is within the current window
                    while curr_ripple_id<ripples.shape[0]-1 and ripples[curr_ripple_id][1] < left:
                        # Ripple ends before the window starts → skip it
                        curr_ripple_id += 1
                    
                    if curr_ripple_id >= ripples.shape[0]:
                        curr_ripple_id=ripples.shape[0]-1
                
                    cur_gt_time = ripples[curr_ripple_id]      
                    if (cur_gt_time[0] >= left) and (cur_gt_time[0] <= right):
                        '''
                            Check if the current window overlaps with the current GT event
                            The Network may spike in the interval [GT_time[0], GT_time[0] + MEAN_HFO_DURATION + PRED_GT_TOLERANCE]
                            However, we are using an upper limit for the HFO Duration of WINDOW_SIZE.
                            This way, the Ground Truth Timestamps will be clamped uppwards by WINDOW_SIZE - MAX_HFO_DURATION + MEAN_HFO_DURATION
                        '''
                        if  cur_gt_time[0] + MAX_DETECTION_OFFSET*factor<=right: # If the GT event is completely within the current window
                            '''The Network should predict the HFO -> Calculate the spike time
                            Let's assume the network should spike at the end of the relevant event. We have no way of knowing
                            the exact end time, so we use the mean duration of the event to calculate the spike time.
                            '''
                            avg_spike_time = cur_gt_time[0] +  MEAN_DETECTION_OFFSET*factor # The network should spike at the end of the relevant event
                            
                            # Subtract the left offset to get the spike time in the current window
                            relative_spike_time = avg_spike_time - left

                            relative_spike_time/=factor
                            curr_gt = relative_spike_time   # Update the curr_gt value

                            curr_ripple=curr_ripple_id+dataset_id
                    # Append the current window    
                    ripple_ids.append(curr_ripple)
                    windowed_input_data.append(downsampled_window)            
                    # Append the current GT Spike Time to the windowed GT
                    windowed_gt.append(curr_gt)
                    filtered_windows.append(filtered_window)
                total_hfos+=ripples.shape[0]
            else:
                print(f"[WARNING] Channel {channel} has a very low threshold. Skipping...")
        dataset_id+=liset.ripples_GT.shape[0]
        
    # Convert to numpy array
    ripple_ids=np.array(ripple_ids,dtype=np.int32)
    filtered_windows=np.array(filtered_windows, dtype=np.float32)
    windowed_input_data = np.array(windowed_input_data)
    windowed_gt = np.array(windowed_gt, dtype=np.float32)
    removed_windows = total_windows_count - windowed_input_data.shape[0]
    print(f"Removed {removed_windows}/{total_windows_count} ({round((removed_windows / total_windows_count)*100, 2)}%) windows with no input activations")
    print(f"Skipped {skipped_hfo_count} HFOs due to no input activations")
    print(f"Total HFOs (theoretical): {total_hfos}")
    print("Windowed Input Data Shape: ", windowed_input_data.shape)
    print("Windowed GT Shape: ", windowed_gt.shape)
    print("Filtered Windows Shape: ", filtered_windows.shape)
    return  windowed_input_data, windowed_gt, filtered_windows, ripple_ids, config


def min_max_spike_threshold(windows,gt,ripple_ids,MEAN_DETECTION_OFFSET,thresholds):
    """
    Filters spike windows based on spike activity thresholds.

    Args:
        windows (np.ndarray): shape (N, T, 2)
        gt (np.ndarray): shape (N,)
        MEAN_DETECTION_OFFSET (int): frames before GT spike to check for activity
        thresholds (tuple): (non_hfo_threshold, hfo_activity_threshold)

    Returns:
        filtered_windows, filtered_gt
    """
    cleaned_windows=[]
    cleaned_gt=[]
    cleaned_ripple_ids=[]

    for window, label,id in zip(windows, gt,ripple_ids):
        if label == -1:
            if np.sum(window) > thresholds[0]:
                print("False Negative Removed - Number of Spikes: ", np.sum(window))
                continue
        else:
            # Optional: you could add checks for valid spike positions
            spike_time = int(label)
            if np.sum(window[spike_time-MEAN_DETECTION_OFFSET:spike_time]) < thresholds[1]:
                print("False Positive Removed - Number of Spikes: ", np.sum(window))
                continue
        
        cleaned_windows.append(window)
        cleaned_gt.append(label)
        cleaned_ripple_ids.append(id)
    removed=len(windows)-len(cleaned_windows)
    print("Removed ", removed, " windows!")
    return np.array(cleaned_windows), np.array(cleaned_gt),np.array(cleaned_ripple_ids)


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