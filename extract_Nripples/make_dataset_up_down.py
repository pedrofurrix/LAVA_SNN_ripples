import sys
import os
liset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../liset_tk'))

sys.path.insert(0, liset_path)

from liset_aux import ripples_std, middle
from signal_aid import most_active_channel, bandpass_filter
from different_methods_conversion import *
import matplotlib.pyplot as plt
from liset_tk import liset_tk
import os
import numpy as np
from copy import deepcopy
import time
import json

parent = r"C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\CNN_TRAINING_SESSIONS" # Modify this to your data path folder
downsampled_fs= 4000
save_dir = os.path.join(os.path.dirname(__file__),"train_pedro","dataset_up_down")
time_max=10 # seconds
window_size=0.05 # seconds
sample_ratio=0.25 # ratio of max amplitudes to use
scaling_factor=0.5 # scale factor for the threshold
refractory=0 # seconds
bandpass=[100,250]
min_threshold=0.1

def make_up_down(parent=parent,downsampled_fs=downsampled_fs,save_dir=save_dir,
                 time_max=time_max,window_size=window_size,sample_ratio=sample_ratio,scaling_factor=scaling_factor,
                 refractory=refractory,bandpass=bandpass,min_threshold=min_threshold):
    # Define general variables
    
    chunk_size  = 10000000
    threshold=np.zeros(8)
    # Define saving directory
    print('Extracting True Positive events ...')
    for i in os.listdir(parent):
        print(i)
        # Restart loop variables
        dataset_path = os.path.join(parent, i)

        liset = liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=0, verbose=False)
        spikified_chunks=[]

        for channel in range(liset.data.shape[1]):
            channel_signal = liset.data[:time_max*downsampled_fs, channel]
            filtered_signal=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
            threshold[channel]=max(min_threshold,calculate_threshold(filtered_signal,downsampled_fs,window_size,sample_ratio,scaling_factor))

        print("Thresholds:",threshold)
        start = 0
        keep_looping = True
        # Create the save directory for each dataset
        sub_save_dir=os.path.join(save_dir, f"{i}",f"{downsampled_fs}")
        os.makedirs(sub_save_dir, exist_ok=True)  # <-- creates directory if it doesn't exist

        # Loop until all ripples are saved in the list
        while keep_looping:
            print(f'Start: {start}', end='\r', flush=True)
            liset = liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=start, numSamples=chunk_size, verbose=False)

            if hasattr(liset, 'data'):
                chunk_spikes = np.zeros((liset.data.shape[0], liset.data.shape[1], 2))
                # Loop throough the ripples found in liset class (the ones in the range of the selected samples)
                for channel in range(liset.data.shape[1]):
                    # Find the peaks above the threshold
                    channel_signal = liset.data[:, channel]
                    filtered_signal=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
                    chunk_spikes[:, channel, :]=up_down_channel(filtered_signal,threshold[channel],downsampled_fs,refractory)
                # Update the reading start for the next loop
                spikified_chunks.append(chunk_spikes)
                start += chunk_size
            else:
                start = 0
                keep_looping = False
                print("End of file reached.")

        spikified=np.concatenate(spikified_chunks, axis=0)
        # Save the spikified data	
        save_data=os.path.join(sub_save_dir, f'data_up_down_{bandpass[0]}_{bandpass[1]}.npy')
        np.save(save_data, arr=spikified, allow_pickle=True)
        save_params(sub_save_dir,time_max,window_size,sample_ratio,scaling_factor,refractory,bandpass,threshold,downsampled_fs,chunk_size)
        print(f'Saved UP-DOWN DataSet - {i}')

def save_params(sub_save_dir,time_max,window_size,sample_ratio,scaling_factor,refractory,bandpass,threshold,downsampled_fs,chunk_size):
    # Save the parameters used for the conversion
    params = {
        'time_max': time_max,
        'window_size': window_size,
        'sample_ratio': sample_ratio,
        'scaling_factor': scaling_factor,
        'refractory': refractory,
        'bandpass': bandpass,
        "threshold": threshold.tolist(),
        "downsampled_fs": downsampled_fs,
        "chunk_size": chunk_size,
    }
    save_path = os.path.join(sub_save_dir, f'params_{bandpass[0]}_{bandpass[1]}.json')  # use .json instead of .npy
    with open(save_path, 'w') as f:
        json.dump(params, f, indent=4)  # optional: indent=4 for readability

# make_up_down()

def plot_ripple_no_ripple(save_dir=save_dir,bandpass=bandpass,downsampled_fs=downsampled_fs,parent=parent,id=0,ripple=7,channels=[],diff_plots=True):
    datasets=os.listdir(parent)
    up_down= np.load(os.path.join(save_dir,datasets[id],f"{downsampled_fs}", f'data_up_down_{bandpass[0]}_{bandpass[1]}.npy'))
    dataset_path=os.path.join(parent,datasets[id])
    liset= liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=0, verbose=False)
    
    if not ripple:
        chunk_length=int(0.04*downsampled_fs)    
        max_start = liset.data.shape[0] - chunk_length
        while True:
            candidate_start = np.random.randint(0, max_start)
            candidate_end = candidate_start + chunk_length
            if not overlaps_with_any_ripple(candidate_start, candidate_end, liset.ripples_GT):
                ripple_ids = [candidate_start, candidate_end]
                print(f"Candidate start: {candidate_start}, Candidate end: {candidate_end}")
                break  # found a valid, non-overlapping chunk
        title="No Ripple"
    else:
        ripple_ids=liset.ripples_GT[ripple]
        title=f"Ripple {ripple} "

    signal=liset.data[ripple_ids[0]:ripple_ids[1],:]
    up_down_ripple=up_down[ripple_ids[0]:ripple_ids[1],:,:]
    time=np.arange(0,up_down_ripple.shape[0])/downsampled_fs
    for channel in channels:
        filtered_signal=bandpass_filter(signal[:,channel], bandpass=bandpass, fs=liset.fs)
        if diff_plots:
            # Plot
            plt.subplots(2,1,figsize=(12, 8),sharex=True)
            plt.suptitle(f"{title} - Channel {channel+1}")
            plt.subplot(211)
            plt.title("Filtered Signal")
            plt.plot(time, filtered_signal, label='Original Signal', color='black')

            # Overlay positive spikes
            plt.subplot(212)
            plt.title("UP/Down Spikes")

            plt.vlines(time[up_down_ripple[:,channel,0] == 1], 0.5,1.5, alpha=0.5,
                    color='red', label='Positive Spikes' ,lw=0.5)
            # Overlay negative spikes
            plt.vlines(time[up_down_ripple[:,channel,1] == 1], -0.5,0.5, alpha=0.5,
                    color='blue', label='Negative Spikes',lw=0.5)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude") 
            plt.show()
        
        else:
            plt.suptitle(f"{title} - Channel {channel+1}")
            plt.figure(figsize=(12, 4))
            plt.plot(time, filtered_signal, label='Original Signal', color='black')
            peak=np.max(filtered_signal)
            trough=np.min(filtered_signal)
            mean=np.mean(filtered_signal)
            plt.vlines(time[up_down_ripple[:,channel,0] == 1],mean,peak, alpha=0.5,
                    color='red', label='Positive Spikes' ,lw=0.5)
            plt.vlines(time[up_down_ripple[:,channel,1] == 1], trough,mean, alpha=0.5,
                    color='blue', label='Negative Spikes',lw=0.5)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude") 
            plt.show()



def overlaps_with_any_ripple(candidate_start, candidate_end, ripples_GT):
    """
    Returns True if the candidate interval overlaps with any ripple in ripples_GT.
    """
    for ripple_start, ripple_end in ripples_GT:
        # Check for overlap: A < D and C < B
        if candidate_start < ripple_end and ripple_start < candidate_end:
            return True
    return False


def plot_channels(save_dir=save_dir,bandpass=bandpass,downsampled_fs=downsampled_fs,parent=parent,id=0,ripple=7,channels=[],diff_plots=False):
    datasets=os.listdir(parent)
    up_down= np.load(os.path.join(save_dir,datasets[id],f"{downsampled_fs}", f'data_up_down_{bandpass[0]}_{bandpass[1]}.npy'))
    dataset_path=os.path.join(parent,datasets[id])
    liset= liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=0, verbose=False)
    
    if not ripple:
        chunk_length=int(0.04*downsampled_fs)    
        max_start = liset.data.shape[0] - chunk_length
        while True:
            candidate_start = np.random.randint(0, max_start)
            candidate_end = candidate_start + chunk_length
            if not overlaps_with_any_ripple(candidate_start, candidate_end, liset.ripples_GT):
                ripple_ids = [candidate_start, candidate_end]
                print(f"Candidate start: {candidate_start}, Candidate end: {candidate_end}")
                break  # found a valid, non-overlapping chunk
        title="No Ripple"
    else:
        ripple_ids=liset.ripples_GT[ripple]
        title=f"Ripple {ripple} "

    signal=liset.data[ripple_ids[0]:ripple_ids[1],:]
    up_down_ripple=up_down[ripple_ids[0]:ripple_ids[1],:,:]
    time=np.arange(0,up_down_ripple.shape[0])/downsampled_fs

    if diff_plots:
        fig,axes=plt.subplots(len(channels),2,figsize=(10,4*len(channels)),sharex=True,sharey=True,constrained_layout=True)
        fig.suptitle(f"{title}")
    else:
        fig,axes=plt.subplots(int(len(channels)/2),2,figsize=(int(8*2),int(4*len(channels)/2)),sharex=True,sharey=True,constrained_layout=True)
        fig.suptitle(f"{title}")   


    for channel in channels:
        filtered_signal=bandpass_filter(signal[:,channel], bandpass=bandpass, fs=liset.fs)
        if diff_plots:
            ax=axes[channel,0]
            # Plot
            ax.set_title("Filtered Signal")
            ax.plot(time, filtered_signal, label='Original Signal', color='black')
            ax=axes[channel,1]
            ax.set_title("UP/Down Spikes")
            ax.vlines(time[up_down_ripple[:,channel,0] == 1], 0.5,1.5, alpha=0.5,
                    color='red', label='Positive Spikes' ,lw=0.5)
            # Overlay negative spikes
            ax.vlines(time[up_down_ripple[:,channel,1] == 1], -0.5,0.5, alpha=0.5,
                    color='blue', label='Negative Spikes',lw=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")         
        else:
            ax=axes[int(channel // 2),int(channel % 2)]
            ax.set_title(f"Channel {channel+1}")
            ax.plot(time, filtered_signal, label='Original Signal', color='black')
            if not ripple:
                peak=max(np.max(filtered_signal),0.1)
                trough=min(np.min(filtered_signal),-0.1)
            else:
                peak=max(np.max(filtered_signal),0.5)
                trough=min(np.min(filtered_signal),-0.5)
            mean=np.mean(filtered_signal)
            ax.vlines(time[up_down_ripple[:,channel,0] == 1],mean,peak, alpha=0.5,
                    color='red', label='Positive Spikes' ,lw=0.5)
            ax.vlines(time[up_down_ripple[:,channel,1] == 1], trough,mean, alpha=0.5,
                    color='blue', label='Negative Spikes',lw=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude") 
    plt.show()