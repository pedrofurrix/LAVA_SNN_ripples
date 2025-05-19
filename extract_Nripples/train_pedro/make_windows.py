from liset_tk import liset_tk
import os
import numpy as np
from signal_aid import most_active_channel, bandpass_filter
from liset_aux import ripples_std, middle
from extract_Nripples.utils_encoding import *
def make_windows(parent,config,time_max,downsampled_fs,bandpass,window_size,sample_ratio, scaling_factor,
                 refractory,WINDOW_SHIFT, WINDOW_SIZE,MEAN_DETECTION_OFFSET,MAX_DETECTION_OFFSET,factor):
    """
    Function to create the windows for the training of the model
    """

    # Create the directory to save the data
    os.makedirs(os.path.join(parent), exist_ok=True)

    # Initialize variables
    ripples_concat = []
    data_concat = []
    ripples = []
    spikified = []
    filtered = []
    # Split the Input Data and Ground Truth into Windows
    windowed_input_data = []    # Input Data Windows
    windowed_gt = []        # Ground Truth Windows (spike time if HFO, -1 if no HFO)
    filtered_windows=[] 
    total_windows_count = 0
    skipped_hfo_count = 0   # Counts the nº of skipped HFOs due to no input activations
    total_hfos=0
    # curr_ripple_times = ripples_concat[curr_ripple_id]    # Get the GT times for the current sEEG source

    # LOAD THE DATA
    # Iterate over the datasets
    for dataset in os.listdir(parent):
        config[str(dataset)]={}
        dataset_path = os.path.join(parent, dataset)
        liset= liset_tk(dataset_path, shank=3, downsample=downsampled_fs, verbose=False)

        ripples=np.array(liset.ripples_GT)
        spikified=np.zeros((liset.data.shape[0], liset.data.shape[1], 2))
        filtered=np.zeros((liset.data.shape[0], liset.data.shape[1]))
        thresholds = []

        print("data shape: ", liset.data.shape)
        print("ripples shape: ", ripples.shape)
        # print("Head of data_concat: ", data[:10][:])
        # print("Head of ripples_concat: ", ripples[:10])
        ripples = ripples[np.argsort(ripples[:, 0])]
        # print(ripples[:10][:])
        
        for channel in range(liset.data.shape[1]):
            channel_signal = liset.data[:time_max*liset.fs, channel]
            filtered_signal=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
            thresholds.append(round(calculate_threshold(filtered_signal,liset.fs,window_size,sample_ratio,scaling_factor),4))

            if thresholds[channel] > 0.1:
                channel_signal = liset.data[:, channel]
                curr_ripple_id = 0     # Keep track of the current GT event index since it is monotonically increasing the timestep
                filtered_liset=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
                spikified[:, channel, :]=up_down_channel(filtered_liset,thresholds[channel],liset.fs,refractory)
                
                for i in range(0, liset.data.shape[0], WINDOW_SHIFT*factor):
                    left, right = i, i+WINDOW_SIZE*factor
                    # Get the current input window
                    curr_window = spikified[left:right, channel, :]
                    downsampled_window=extract_spikes_downsample(curr_window,factor)
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
                        cur_gt_time=[-1,-1]   # Default value for Spike Time (no HFO)
                        if curr_ripple_id < ripples.shape[0]:
                            cur_gt_time = ripples[curr_ripple_id]  
                        if (cur_gt_time[1] >= left) and (cur_gt_time[0] <= right):
                            if cur_gt_time[0] + MAX_DETECTION_OFFSET <= right:
                                print(f"[WARNING] Window [{left}:{right}] has a GT event at {cur_gt_time} and NO Input activations. Skipping...")
                                # Update the curr_gt_idx to the next GT event
                                skipped_hfo_count += 1
                            curr_ripple_id += 1
                        continue   
                    
                    '''
                    Check if there is a GT event in the current window
                    '''

                    curr_gt = -1    # Default value for Spike Time (no HFO)
                    
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
                            avg_spike_time = cur_gt_time[0] +  MEAN_DETECTION_OFFSET* factor # The network should spike at the end of the relevant event
                            
                            # Subtract the left offset to get the spike time in the current window
                            relative_spike_time = (avg_spike_time - left)//factor
                            if relative_spike_time > WINDOW_SIZE:
                                # If the spike time is greater than the window size, we want to skip the window
                                print(f"[WARNING] Spike time {relative_spike_time} is greater than the window size {WINDOW_SIZE}. Adjusting...")
                                relative_spike_time= cur_gt_time[1]-left

                            curr_gt = int(relative_spike_time)   # Update the curr_gt value

                            # Update the curr_gt_idx to the next GT event
                            curr_ripple_id += 1
                            
                        elif cur_gt_time[1] > right or cur_gt_time[0] < left:
                            continue
                            # If the GT event is not completely within the current window, we want to skip the window
                    
                    # Append the current window    
                    windowed_input_data.append(downsampled_window)            
                    # Append the current GT Spike Time to the windowed GT
                    windowed_gt.append(curr_gt)
                    filtered_windows.append(filtered_window)
                total_hfos+=ripples.shape[0]
            else:
                print(f"[WARNING] Channel {channel} has a very low threshold. Skipping...")
        config[str(dataset)]["thresholds"]=thresholds
    # Convert to numpy array
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
    return windowed_input_data, windowed_gt, filtered_windows, config

def make_windows_mesquita(parent,config,time_max,downsampled_fs,bandpass,window_size,sample_ratio, scaling_factor,
                 refractory,WINDOW_SHIFT, WINDOW_SIZE,MEAN_DETECTION_OFFSET,MAX_DETECTION_OFFSET,factor):
    """
    Function to create the windows for the training of the model
    """
    # Split the Input Data and Ground Truth into Windows
    windowed_input_data = []    # Input Data Windows
    windowed_gt = []        # Ground Truth Windows (spike time if HFO, -1 if no HFO)
    filtered_windows=[] 
    total_windows_count = 0
    skipped_hfo_count = 0   # Counts the nº of skipped HFOs due to no input activations
    total_hfos=0
    # curr_ripple_times = ripples_concat[curr_ripple_id]    # Get the GT times for the current sEEG source

    # LOAD THE DATA
    # Iterate over the datasets
    for dataset in os.listdir(parent):
        config[str(dataset)]={}
        dataset_path = os.path.join(parent, dataset)
        liset= liset_tk(dataset_path, shank=3, downsample=downsampled_fs, verbose=False)

        ripples=np.array(liset.ripples_GT)
        spikified=np.zeros((liset.data.shape[0], liset.data.shape[1], 2))
        filtered=np.zeros((liset.data.shape[0], liset.data.shape[1]))
        thresholds = []

        print("data shape: ", liset.data.shape)
        print("ripples shape: ", ripples.shape)
        # print("Head of data_concat: ", data[:10][:])
        # print("Head of ripples_concat: ", ripples[:10])
        ripples = ripples[np.argsort(ripples[:, 0])]
        # print(ripples[:10][:])
        
        for channel in range(liset.data.shape[1]):
            channel_signal = liset.data[:time_max*liset.fs, channel]
            filtered_signal=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
            thresholds.append(round(calculate_threshold(filtered_signal,liset.fs,window_size,sample_ratio,scaling_factor),4))

            if thresholds[channel] > 0.1:
                channel_signal = liset.data[:, channel]
                curr_ripple_id = 0     # Keep track of the current GT event index since it is monotonically increasing the timestep
                filtered_liset=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
                spikified[:, channel, :]=up_down_channel(filtered_liset,thresholds[channel],liset.fs,refractory)
                
                for i in range(0, liset.data.shape[0], WINDOW_SHIFT*factor):
                    left, right = i, i+WINDOW_SIZE*factor
                    # Get the current input window
                    curr_window = spikified[left:right, channel, :]
                    downsampled_window=extract_spikes_downsample(curr_window, liset.fs,1000)
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
                        cur_gt_time=[-1,-1]   # Default value for Spike Time (no HFO)
                        if curr_ripple_id < ripples.shape[0]:
                            cur_gt_time = ripples[curr_ripple_id]  
                        if (cur_gt_time[1] >= left) and (cur_gt_time[0] <= right):
                            if cur_gt_time[0] + MAX_DETECTION_OFFSET <= right:
                                print(f"[WARNING] Window [{left}:{right}] has a GT event at {cur_gt_time} and NO Input activations. Skipping...")
                                # Update the curr_gt_idx to the next GT event
                                skipped_hfo_count += 1
                            curr_ripple_id += 1
                        continue   
                    
                    '''
                    Check if there is a GT event in the current window
                    '''

                    curr_gt = -1    # Default value for Spike Time (no HFO)
                    
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
                        if cur_gt_time + MAX_DETECTION_OFFSET <= right: 
                            '''The Network should predict the HFO -> Calculate the spike time
                            Let's assume the network should spike at the end of the relevant event. We have no way of knowing
                            the exact end time, so we use the mean duration of the event to calculate the spike time.
                            '''
                            avg_spike_time = cur_gt_time[0]+  MEAN_DETECTION_OFFSET*factor # The network should spike at the end of the relevant event
                            
                            # Subtract the left offset to get the spike time in the current window
                            relative_spike_time = (avg_spike_time - left)//factor
                            curr_gt = int(relative_spike_time)   # Update the curr_gt value
                            # Update the curr_gt_idx to the next GT event
                            curr_ripple_id += 1
                    # Append the current window    
                    windowed_input_data.append(downsampled_window)            
                    # Append the current GT Spike Time to the windowed GT
                    windowed_gt.append(curr_gt)
                    filtered_windows.append(filtered_window)
                total_hfos+=ripples.shape[0]
            else:
                print(f"[WARNING] Channel {channel} has a very low threshold. Skipping...")
        config[str(dataset)]["thresholds"]=thresholds
    # Convert to numpy array
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
    return windowed_input_data, windowed_gt, filtered_windows, config