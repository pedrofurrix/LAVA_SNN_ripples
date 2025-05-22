import os
import numpy as np
# from liset_tk import liset_tk
from liset_paper import liset_paper as liset_tk
from signal_aid import most_active_channel, bandpass_filter
from extract_Nripples.utils_encoding import *
import random

def make_windows(parent,config,time_max,downsampled_fs,bandpass,window_size,sample_ratio, scaling_factor, 
                     refractory,WINDOW_SHIFT, WINDOW_SIZE,MEAN_DETECTION_OFFSET,MAX_DETECTION_OFFSET,factor,fraction=1):
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
        liset= liset_tk(dataset_path, shank=1, downsample=False, verbose=False)
        downsample_factor=liset.fs//downsampled_fs
        liset=TrainData(liset,fraction)

        ripples=np.array(liset.ripples_GT)//downsample_factor
        spikified=np.zeros((liset.data.shape[0]//downsample_factor, liset.data.shape[1], 2))
        filtered=np.zeros((liset.data.shape[0]//downsample_factor, liset.data.shape[1]))
        thresholds = []
        downsampled=np.zeros((spikified.shape[0]//factor, liset.data.shape[1], 2))
        print("Dataset: ", dataset)
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
                filtered[:,channel]=decimation_downsampling(filtered_liset,downsample_factor)
                spikified[:, channel, :]=up_down_channel(filtered[:,channel],thresholds[channel],downsampled_fs,refractory)
                downsampled[:,channel,:]=extract_spikes_downsample(spikified[:,channel,:],factor)
                
                for i in range(0, downsampled.shape[0], WINDOW_SHIFT*factor):
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
        print("Thresholds: ", thresholds)
        print("Dataset Processed: ", dataset)
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
                     refractory,WINDOW_SHIFT, WINDOW_SIZE,MEAN_DETECTION_OFFSET,MAX_DETECTION_OFFSET,factor,fraction=1):
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
        liset= liset_tk(dataset_path, shank=1, downsample=False, verbose=False)
        liset=TrainData(liset,fraction)
        downsample_factor=liset.fs//downsampled_fs
        ripples=np.array(liset.ripples_GT)//downsample_factor
        spikified=np.zeros((liset.data.shape[0]//downsample_factor, liset.data.shape[1], 2))
        thresholds = []
        downsampled=np.zeros((liset.data.shape[0]//(downsample_factor*factor), liset.data.shape[1], 2))

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
            if downsample_factor>1:
                filtered_signal=decimation_downsampling(filtered_signal,downsample_factor)
            thresholds.append(round(calculate_threshold(filtered_signal,downsampled_fs,window_size,sample_ratio,scaling_factor),4))
            config[dataset]["thresholds"][channel]=thresholds[channel]  
            if thresholds[channel] > 0.1:
                channel_signal = liset.data[:, channel]
                curr_ripple_id = 0     # Keep track of the current GT event index since it is monotonically increasing the timestep
                filtered_liset=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
                if downsample_factor>1:
                    filtered_liset=decimation_downsampling(filtered_liset,downsample_factor)
                    # filtered_liset=average_downsampling(filtered_liset,downsample_factor)
                spikified[:, channel, :]=up_down_channel(filtered_liset,thresholds[channel],downsampled_fs,refractory)
                # spikified[:, channel, :]=up_down_channel_SF(filtered_liset,thresholds[channel],downsampled_fs,refractory)
                if factor>1:
                    downsampled[:,channel,:],spikes_lost=extract_spikes_downsample(spikified[:,channel,:],factor)
                else:
                    downsampled[:,channel,:]=spikified[:,channel,:]
                
                for i in range(0, spikified.shape[0], WINDOW_SHIFT*factor):
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
                            # curr_ripple_id += 1
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

                            relative_spike_time//=factor
                            curr_gt = int(relative_spike_time)   # Update the curr_gt value

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


def min_max_spike_threshold_prob(windows, gt, ripple_ids, MEAN_DETECTION_OFFSET, thresholds, max_prob=1.0,multiplier=1,decay=3.0,drop_fn=drop_linear):
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



class TrainData:
    def __init__(self, liset,fraction,beginning=True):
        self.id_train=int(liset.data.shape[0]*fraction)
        self.fs=liset.fs
        self.get_data(liset,beginning)
        
    
    def get_data(self,liset,beginning):

        if beginning:
            self.data=liset.data[:self.id_train,:]
              # Keep only ripples that start within the training data range
            self.ripples_GT = liset.ripples_GT[
                (liset.ripples_GT[:, 0] < self.id_train)
            ]
        else:
            self.data = liset.data[self.id_train:, :]
            # Ripples that start after id_train
            mask = liset.ripples_GT[:, 1] >= self.id_train
            filtered_ripples = liset.ripples_GT[mask]
            # Shift indices to match new data segment
            self.ripples_GT = filtered_ripples - self.id_train

      