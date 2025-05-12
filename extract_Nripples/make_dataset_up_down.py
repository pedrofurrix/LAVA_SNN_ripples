import sys
import os
liset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../liset_tk'))

sys.path.insert(0, liset_path)

from liset_aux import ripples_std, middle
from signal_aid import most_active_channel, bandpass_filter

from utils_encoding import *
import matplotlib.pyplot as plt
from liset_tk import liset_tk
import os
import numpy as np
from copy import deepcopy
import time
import json
from matplotlib.lines import Line2D

#### LAB PC
parent = r"C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\CNN_TRAINING_SESSIONS" # Modify this to your data path folder

### HOME PC
# parent=r"E:\neurospark_mat\CNN_TRAINING_SESSIONS"
downsampled_fs= 4000
save_dir = os.path.join(os.path.dirname(__file__),"train_pedro","dataset_up_down")
time_max=10 # seconds
window_size=0.05 # seconds # 50 ms
sample_ratio=0.5 # ratio of max amplitudes to use
scaling_factor=0.5 # scale factor for the threshold
refractory=0 # seconds
bandpass=[100,250]
min_threshold=0.1 # minimum threshold for the spike detection
save=True
chunk_size=10000000
threshold=None
def make_up_down(parent=parent,downsampled_fs=downsampled_fs,save_dir=save_dir,
                 time_max=time_max,window_size=window_size,sample_ratio=sample_ratio,scaling_factor=scaling_factor,
                 refractory=refractory,bandpass=bandpass,min_threshold=min_threshold,save=save,chunk_size=chunk_size,threshold=threshold):
   
    # Define saving directory
    print('Extracting UP/Down Spikes ...')
    dirs=os.listdir(parent)
    # dirs=[dirs[0]] # test
    for i in dirs:
        print(i)
        # Restart loop variables
        dataset_path = os.path.join(parent, i)

        # Load data from Liset and initialize threshold
        liset = liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=0, verbose=False)
       
        ripples=np.array(liset.ripples_GT)
        print("Ripples - shape:",ripples.shape)
        spikified_chunks=[]
        filtered_chunks=[]
        # Calculate the threshold for each channel
        if threshold is None:
            threshold=np.zeros(liset.data.shape[1])
            for channel in range(liset.data.shape[1]):
                channel_signal = liset.data[:time_max*downsampled_fs, channel]
                filtered_signal=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
                threshold[channel]=max(min_threshold,calculate_threshold(filtered_signal,downsampled_fs,window_size,sample_ratio,scaling_factor))
            print("Thresholds:",threshold)

        # Initialize looping variables
        start = 0
        keep_looping = True
        # Create the save directory for each dataset
        sub_save_dir=os.path.join(save_dir, f"{i}",f"{downsampled_fs}")

        # Loop until all ripples are saved in the list
        while keep_looping:
            print(f'Start: {start}', end='\r', flush=True)
            liset = liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=start, numSamples=chunk_size, verbose=False)

            if hasattr(liset, 'data'):
                print(f'Shape of the loaded data: {liset.data.shape}')
                chunk_spikes = np.zeros((liset.data.shape[0], liset.data.shape[1], 2))
                chunk_filtered = np.zeros((liset.data.shape[0], liset.data.shape[1]))

                # Loop throough the ripples found in liset class (the ones in the range of the selected samples)
                for channel in range(liset.data.shape[1]):
                    print("Channel:", channel+1)
                    # Find the peaks above the threshold, extract channel data, filter and get the up/down spikes
                    channel_signal = liset.data[:, channel]
                    filtered_signal=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
                    chunk_spikes[:, channel, :]=up_down_channel(filtered_signal,threshold[channel],liset.fs,refractory)
                    chunk_filtered[:,channel]=filtered_signal
                # Add the Update the reading start for the next loop
                spikified_chunks.append(chunk_spikes)
                filtered_chunks.append(chunk_filtered)
                start += chunk_size
            else:
                keep_looping = False
                print("End of file reached.")

        spikified=np.concatenate(spikified_chunks, axis=0)
        filtered=np.concatenate(filtered_chunks, axis=0)
        if save:
            # Save the spikified data	
            os.makedirs(sub_save_dir, exist_ok=True)  # <-- creates directory if it doesn't exist
            save_data=os.path.join(sub_save_dir, f'data_up_down_{bandpass[0]}_{bandpass[1]}.npy')
            np.save(save_data, arr=spikified, allow_pickle=True)
            save_params(sub_save_dir,time_max,window_size,sample_ratio,scaling_factor,refractory,bandpass,threshold,downsampled_fs,chunk_size)
            print(f'Saved UP-DOWN DataSet - {i}')
            # Save Ripples
            ripples_path=os.path.join(sub_save_dir, f'ripples.npy')
            np.save(ripples_path, arr=ripples, allow_pickle=True)
            print(f'Saved Ripples - {i}')
        else:
            return spikified,filtered

def save_params(sub_save_dir,time_max,window_size,sample_ratio,scaling_factor,refractory,bandpass,threshold,downsampled_fs,chunk_size=chunk_size):
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

def plot_ripple_no_ripple(spikified,filtered,save_dir=save_dir,bandpass=bandpass,downsampled_fs=downsampled_fs,parent=parent,id=0,ripple=7,channels=[],diff_plots=True):
    datasets=os.listdir(parent)

    dataset_path=os.path.join(parent,datasets[id])
    up_down_path=os.path.join(save_dir,datasets[id],f"{downsampled_fs}")
    liset= liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=0, verbose=False)
    print("Loaded LFPs:",dataset_path)

    if filtered is None:
        filtered_liset=np.zeros((int(liset.data.shape[0]),int(liset.data.shape[1])))
        for channel in channels:
            print("Channel:", channel+1)
            filtered_liset[:,channel]=bandpass_filter(liset.data[:,channel], bandpass=bandpass, fs=liset.fs)
            # print("Max:",max(filtered_liset[:,channel]),"\n Min:",min(filtered_liset[:,channel]))
    else:
        filtered_liset=filtered
        print("Filtered Loaded")

    if spikified is None:
        path=os.path.join(up_down_path, f'data_up_down_{bandpass[0]}_{bandpass[1]}.npy')
        up_down= np.load(path)
        print("Loaded UP/DN SPikes:", path)
    else:
        up_down=spikified
        print("Spikified Loaded")
    
    with open(os.path.join(up_down_path, f'params_{bandpass[0]}_{bandpass[1]}.json'), 'r') as f:
        parameters=json.load(f)

    thresholds=parameters["threshold"]
    print(f'Shape of the filtered data: {filtered_liset.shape}')
    print(f'Shape of the UP/DN data: {up_down.shape}')
   
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

    signal=filtered_liset[ripple_ids[0]:ripple_ids[1],:]
    up_down_ripple=up_down[ripple_ids[0]:ripple_ids[1],:,:]
    time=np.arange(ripple_ids[0],ripple_ids[1])/downsampled_fs

    for channel in channels:
        for channel in channels:
            reconstructed_signal=np.zeros(up_down_ripple.shape[0])
            reconstructed_signal[0]=signal[0,channel]
            for t in range(1, up_down_ripple.shape[0]):
                spike_plus = up_down_ripple[t, channel,0]
                spike_minus = up_down_ripple[t, channel,1]
                if spike_plus == 1:
                    reconstructed_signal[t] = reconstructed_signal[t - 1] + thresholds[channel]
                elif spike_minus == 1:
                    reconstructed_signal[t] = reconstructed_signal[t - 1] - thresholds[channel]
                else:
                    reconstructed_signal[t] = reconstructed_signal[t - 1] 
        if diff_plots:
            # Plot
            plt.subplots(2,1,figsize=(12, 8),sharex=True)
            plt.suptitle(f"{title} - Channel {channel+1}")
            plt.subplot(211)
            plt.title("Filtered Signal")
            plt.plot(time, signal[:,channel], label='Original Signal', color='black')
            plt.plot(time, reconstructed_signal, label='Reconstructed Signal', linestyle="--",alpha=0.6, color='green')

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
            plt.plot(time, signal[:,channel], label='Original Signal', color='black')
            plt.plot(time, reconstructed_signal, label='Reconstructed Signal', linestyle="--",alpha=0.6, color='green')
            
            if not ripple:
                peak=max(np.max(signal[:,channel]),0.1)
                trough=min(np.min(signal[:,channel]),-0.1)
            else:
                peak=max(np.max(signal[:,channel]),0.5)
                trough=min(np.min(signal[:,channel]),-0.5)

            mean=np.mean(signal[:,channel])
     
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




def make_up_down_nochunks(parent=parent,downsampled_fs=downsampled_fs,save_dir=save_dir,
                 time_max=time_max,window_size=window_size,sample_ratio=sample_ratio,scaling_factor=scaling_factor,
                 refractory=refractory,bandpass=bandpass,min_threshold=min_threshold,save=save,threshold=threshold):
    
    # Define saving directory
    print('Extracting UP/Down Spikes ...')
    dirs=os.listdir(parent)
    # dirs=[dirs[0]] # test
    for i in dirs:
        print(i)
        # Restart loop variables
        dataset_path = os.path.join(parent, i)
    
        # Load data from Liset and initialize threshold
        liset = liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=0, verbose=False)
        ripples=np.array(liset.ripples_GT)
        print("Ripples - shape:",ripples.shape)        
        spikified=np.zeros((liset.data.shape[0], liset.data.shape[1], 2))
        filtered=np.zeros((liset.data.shape[0], liset.data.shape[1]))

        # Calculate the threshold for each channel if not given
        if threshold is None:
            thresholds=np.zeros(liset.data.shape[1])
            for channel in range(liset.data.shape[1]):
                channel_signal = liset.data[:time_max*downsampled_fs, channel]
                filtered_signal=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
                thresholds[channel]=max(min_threshold,round(calculate_threshold(filtered_signal,downsampled_fs,window_size,sample_ratio,scaling_factor),4))
            print("Thresholds:",thresholds)
        else:
            thresholds=np.ones(liset.data.shape[1])*threshold

        sub_save_dir=os.path.join(save_dir, f"{i}",f"{downsampled_fs}")

        if hasattr(liset, 'data'):
            print(f'Shape of the loaded data: {liset.data.shape}')
            # Loop throough the ripples found in liset class (the ones in the range of the selected samples)
            for channel in range(liset.data.shape[1]):
                print("Channel:", channel+1)
                # Find the peaks above the threshold, extract channel data, filter and get the up/down spikes
                channel_signal = liset.data[:, channel]
                filtered_signal=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
                filtered[:,channel]=filtered_signal
                spikified[:, channel, :]=up_down_channel(filtered_signal,thresholds[channel],liset.fs,refractory)
        else:
            print("There is no data :(")
            return

        if save:
            # Save the spikified data	
            os.makedirs(sub_save_dir, exist_ok=True)  # <-- creates directory if it doesn't exist
            save_data=os.path.join(sub_save_dir, f'data_up_down_{bandpass[0]}_{bandpass[1]}.npy')
            np.save(save_data, arr=spikified, allow_pickle=True)
            save_params(sub_save_dir,time_max,window_size,sample_ratio,scaling_factor,refractory,bandpass,thresholds,downsampled_fs,chunk_size)
            print(f'Saved UP-DOWN DataSet - {i}')
            # Save ripples
            ripples_path=os.path.join(sub_save_dir, f'ripples.npy')
            np.save(ripples_path, arr=ripples, allow_pickle=True)
            print(f'Saved Ripples - {i}')
        else:
            return spikified,filtered
    
def plot_channels(spikified=None,filtered=None,save_dir=save_dir,bandpass=bandpass,downsampled_fs=downsampled_fs,parent=parent,id=0,ripple=7,channels=[],diff_plots=False):
    
    datasets=os.listdir(parent)
    dataset_path=os.path.join(parent,datasets[id])
    up_down_path=os.path.join(save_dir,datasets[id],f"{downsampled_fs}")
    liset= liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=0, verbose=False)
    print("Loaded LFPs:",dataset_path)
    if filtered is None:
        filtered_liset=np.zeros((int(liset.data.shape[0]),int(liset.data.shape[1])))
        for channel in channels:
            print("Channel:", channel+1)
            filtered_liset[:,channel]=bandpass_filter(liset.data[:,channel], bandpass=bandpass, fs=liset.fs)
            # print("Max:",max(filtered_liset[:,channel]),"\n Min:",min(filtered_liset[:,channel]))
    else:
        filtered_liset=filtered
        print("Filtered Loaded")

    if spikified is None:
        path=os.path.join(up_down_path, f'data_up_down_{bandpass[0]}_{bandpass[1]}.npy')
        up_down= np.load(path)
        print("Loaded UP/DN SPikes:", path)
    else:
        up_down=spikified
        print("Spikified Loaded")
    
    with open(os.path.join(up_down_path, f'params_{bandpass[0]}_{bandpass[1]}.json'), 'r') as f:
        parameters=json.load(f)

    thresholds=parameters["threshold"]
    print(f'Shape of the filtered data: {filtered_liset.shape}')
    print(f'Shape of the UP/DN data: {up_down.shape}')
   


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
        print(f"Ripple start: {ripple_ids[0]}, Ripple end: {ripple_ids[1]}")
    
    
    signal=filtered_liset[ripple_ids[0]:ripple_ids[1],:]
    up_down_ripple=up_down[ripple_ids[0]:ripple_ids[1],:,:]
    time=np.arange(ripple_ids[0],ripple_ids[1])/downsampled_fs

    if diff_plots:
        fig,axes=plt.subplots(len(channels),2,figsize=(10,4*len(channels)),sharex=True,sharey=True,constrained_layout=True)
        fig.suptitle(f"{title}")
    else:
        fig,axes=plt.subplots(int(len(channels)/2),2,figsize=(int(8*2),int(4*len(channels)/2)),sharex=True,sharey=True,constrained_layout=True)
        fig.suptitle(f"{title}")   


    for channel in channels:
        reconstructed_signal=np.zeros(up_down_ripple.shape[0])
        reconstructed_signal[0]=signal[0,channel]
        for t in range(1, up_down_ripple.shape[0]):
            spike_plus = up_down_ripple[t, channel,0]
            spike_minus = up_down_ripple[t, channel,1]
            if spike_plus == 1:
                reconstructed_signal[t] = reconstructed_signal[t - 1] + thresholds[channel]
            elif spike_minus == 1:
                reconstructed_signal[t] = reconstructed_signal[t - 1] - thresholds[channel]
            else:
                reconstructed_signal[t] = reconstructed_signal[t - 1] 
                
        if diff_plots:
            ax=axes[channel,0]
            # Plot
            ax.set_title("Filtered Signal")
            ax.plot(time, signal[:,channel], label='Original Signal', color='black')
            ax.plot(time, reconstructed_signal, label='Reconstructed Signal', linestyle="--",alpha=0.6, color='green')
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
            ax.plot(time, signal[:,channel], label='Original Signal', color='black')
            ax.plot(time, reconstructed_signal, label='Reconstructed Signal', linestyle="--",alpha=0.6, color='green')

            if not ripple:
                peak=max(np.max(signal[:,channel]),0.1)
                trough=min(np.min(signal[:,channel]),-0.1)
            else:
                peak=max(np.max(signal[:,channel]),0.5)
                trough=min(np.min(signal[:,channel]),-0.5)
    
            mean=np.mean(signal[:,channel])
            ax.vlines(time[up_down_ripple[:,channel,0] == 1],mean,peak, alpha=0.5,
                    color='red', label='Positive Spikes' ,lw=0.5)
            ax.vlines(time[up_down_ripple[:,channel,1] == 1], trough,mean, alpha=0.5,
                    color='blue', label='Negative Spikes',lw=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")    
        print("Max region:",max(signal[:,channel]),"\n Min:",min(signal[:,channel]))
   
    legend_elements = [
    Line2D([0], [0], color='black', lw=1, label='Filtered Signal'),
    Line2D([0], [0], color='red', lw=1, label='Positive Spikes'),
    Line2D([0], [0], color='blue', lw=1, label='Negative Spikes'),
    Line2D([0], [0], color='green', lw=1, label='Reconstructed Signal'),
    ]

    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))
    plt.show()

def evaluate_encoding(spikified=None,filtered=None,save_dir=save_dir,bandpass=bandpass,downsampled_fs=downsampled_fs,parent=parent,save=save):
    """

    Evaluate the encoding of the UP/DOWN spikes

    """
    metrics={}
    total_up_all = 0
    total_down_all = 0
    global_metrics_sum = {
        "SNR": [],
        "RMSE": [],
        "R_squared": [],
        "AFR": [],
        "SNR_ripples": [],
        "RMSE_ripples": [],
        "R_squared_ripples": [],
        "AFR_ripples": []
    }
    ############## Load the data ##################
    datasets=os.listdir(parent)
    
    for dataset in datasets:
        metrics[dataset]={}
        dataset_path=os.path.join(parent,dataset)
        up_down_path=os.path.join(save_dir,dataset,f"{downsampled_fs}")
        liset= liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=0, verbose=False)
        print("Loaded LFPs:",dataset_path)
        if filtered is None:
            filtered_liset=np.zeros((int(liset.data.shape[0]),int(liset.data.shape[1])))
            channels=[i for i in range(liset.data.shape[1])]
            for channel in channels:
                print("Channel:", channel+1)
                filtered_liset[:,channel]=bandpass_filter(liset.data[:,channel], bandpass=bandpass, fs=liset.fs)
        else:
            filtered_liset=filtered
            print("Filtered Loaded")

        if spikified is None:
            path=os.path.join(up_down_path, f'data_up_down_{bandpass[0]}_{bandpass[1]}.npy')
            up_down= np.load(path)
            print("Loaded UP/DN SPikes:", path)
        else:
            up_down=spikified
            print("Spikified Loaded")
        
        with open(os.path.join(up_down_path, f'params_{bandpass[0]}_{bandpass[1]}.json'), 'r') as f:
            parameters=json.load(f)
            thresholds=parameters["threshold"]
            metrics[dataset]["parameters"]=parameters
        print(f'Shape of the filtered data: {filtered_liset.shape}')
        print(f'Shape of the UP/DN data: {up_down.shape}')

        # Reconstruct the signal
        reconstructed_signal=np.zeros((up_down.shape[0],up_down.shape[1]))

        ripples=liset.ripples_GT

        for channel in channels:
            metrics[dataset][channel]={}
            reconstructed_signal[0,channel]=filtered_liset[0,channel]
            # Loop through the up_down data and reconstruct the signal
            for t in range(1, up_down.shape[0]):
                spike_plus = up_down[t, channel,0]
                spike_minus = up_down[t, channel,1]
                if spike_plus == 1:
                    reconstructed_signal[t, channel] = reconstructed_signal[t - 1, channel] + thresholds[channel]
                elif spike_minus == 1:
                    reconstructed_signal[t, channel] = reconstructed_signal[t - 1, channel] - thresholds[channel]
                else:
                    reconstructed_signal[t, channel] = reconstructed_signal[t - 1, channel]

        # Calculate error metrics between the original and reconstructed signal
        for channel in channels:
            s_full = filtered_liset[:, channel]
            r_full = reconstructed_signal[:, channel]
            spikes_full = up_down[:, channel, 0] + up_down[:, channel, 1]

            # Calculate metrics
            metrics[dataset][channel]["general"]={
                "SNR": calculate_snr(s_full, r_full),
                "RMSE": calculate_rmse(s_full, r_full),
                "R_squared": calculate_r_squared(s_full, r_full),
                "AFR": calculate_average_spike_rate(spikes_full,downsampled_fs)
            }

            # Calculate metrics for ripples
            # --- Ripple metrics (average across ripple windows)
            snrs, rmses, r2s, afrs = [], [], [], []

            for ripple in ripples:
                start, end = ripple[0], ripple[1]
                s = s_full[start:end]
                r = r_full[start:end]
                spikes = spikes_full[start:end]

                snrs.append(calculate_snr(s, r))
                rmses.append(calculate_rmse(s, r))
                r2s.append(calculate_r_squared(s, r))
                afrs.append(calculate_average_spike_rate(spikes,downsampled_fs))

            # Store averaged ripple metrics
            if snrs:  # in case ripples is empty
                metrics[dataset][channel]["ripples"] = {
                    "SNR": float(np.mean(snrs)),
                    "RMSE": float(np.mean(rmses)),
                    "R_squared": float(np.mean(r2s)),
                    "AFR": float(np.mean(afrs))
                }
            else:
                metrics[dataset][channel]["ripples"] = {
                    "SNR": None,
                    "RMSE": None,
                    "R_squared": None,
                    "AFR": None
                }
        total_up_spikes = int(np.sum(up_down[:, :, 0]))
        total_down_spikes = int(np.sum(up_down[:, :, 1]))
        metrics[dataset]["total_up_spikes"] = total_up_spikes
        metrics[dataset]["total_down_spikes"] = total_down_spikes
        metrics[dataset]["total_spikes"] = total_up_spikes + total_down_spikes
        
        metrics[dataset]["average_channels"]={
            "SNR": float(np.mean([metrics[dataset][channel]["general"]["SNR"] for channel in channels])),
            "RMSE": float(np.mean([metrics[dataset][channel]["general"]["RMSE"] for channel in channels])),
            "R_squared": float(np.mean([metrics[dataset][channel]["general"]["R_squared"] for channel in channels])),
            "AFR": float(np.mean([metrics[dataset][channel]["general"]["AFR"] for channel in channels]))
        }
        metrics[dataset]["average_ripples"]={
            "SNR": float(np.mean([metrics[dataset][channel]["ripples"]["SNR"] for channel in channels])),
            "RMSE": float(np.mean([metrics[dataset][channel]["ripples"]["RMSE"] for channel in channels])),
            "R_squared": float(np.mean([metrics[dataset][channel]["ripples"]["R_squared"] for channel in channels])),
            "AFR": float(np.mean([metrics[dataset][channel]["ripples"]["AFR"] for channel in channels]))
        }

        # Accumulate UP/DOWN spike counts
        total_up_all += total_up_spikes
        total_down_all += total_down_spikes
        
        # Accumulate channel-level averages
        global_metrics_sum["SNR"].append(metrics[dataset]["average_channels"]["SNR"])
        global_metrics_sum["RMSE"].append(metrics[dataset]["average_channels"]["RMSE"])
        global_metrics_sum["R_squared"].append(metrics[dataset]["average_channels"]["R_squared"])
        global_metrics_sum["AFR"].append(metrics[dataset]["average_channels"]["AFR"])

        # Accumulate ripple-level averages
        global_metrics_sum["SNR_ripples"].append(metrics[dataset]["average_ripples"]["SNR"])
        global_metrics_sum["RMSE_ripples"].append(metrics[dataset]["average_ripples"]["RMSE"])
        global_metrics_sum["R_squared_ripples"].append(metrics[dataset]["average_ripples"]["R_squared"])
        global_metrics_sum["AFR_ripples"].append(metrics[dataset]["average_ripples"]["AFR"])

    overall_total_spikes = total_up_all + total_down_all

    overall_metrics = {
        "total_spikes": overall_total_spikes,
        "total_up_spikes": total_up_all,
        "total_down_spikes": total_down_all,
        "average_channels": {
            "SNR": float(np.mean(global_metrics_sum["SNR"])),
            "RMSE": float(np.mean(global_metrics_sum["RMSE"])),
            "R_squared": float(np.mean(global_metrics_sum["R_squared"])),
            "AFR": float(np.mean(global_metrics_sum["AFR"]))
        },
        "average_ripples": {
            "SNR": float(np.mean(global_metrics_sum["SNR_ripples"])),
            "RMSE": float(np.mean(global_metrics_sum["RMSE_ripples"])),
            "R_squared": float(np.mean(global_metrics_sum["R_squared_ripples"])),
            "AFR": float(np.mean(global_metrics_sum["AFR_ripples"]))
        }
    }
        
    metrics["overall_metrics"] = overall_metrics

    # Save the metrics
    if save:
        os.makedirs(save_dir, exist_ok=True)
        ############## Save the metrics ##################	
        metrics_path=os.path.join(save_dir,f"metrics_{downsampled_fs}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)  # optional: indent=4 for readability
        print(f"Metrics saved in {metrics_path}")
    else:
        return metrics






def plot_reconstruction(spikified=None,filtered=None,save_dir=save_dir,bandpass=bandpass,downsampled_fs=downsampled_fs,parent=parent,save=save, channels=[0],ripple=7,id=0):
    """
    Plot the reconstruction of the UP/DOWN spikes

    """
    both_counter=0
    ############## Load the data ##################
    dirs=os.listdir(parent)
    datasets=[dirs[id]]
    for dataset in datasets:
        dataset_path=os.path.join(parent,dataset)
        up_down_path=os.path.join(save_dir,dataset,f"{downsampled_fs}")
        liset= liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=0, verbose=False)
        print("Loaded LFPs:",dataset_path)
        if filtered is None:
            filtered_liset=np.zeros((int(liset.data.shape[0]),len(channels)))
            for i,channel in enumerate(channels):
                print("Channel:", channel+1)
                filtered_liset[:,i]=bandpass_filter(liset.data[:,channel], bandpass=bandpass, fs=liset.fs)
        else:
            filtered_liset=filtered[:,channels]
            print("Filtered Loaded")

        if spikified is None:
            path=os.path.join(up_down_path, f'data_up_down_{bandpass[0]}_{bandpass[1]}.npy')
            up_down= np.load(path)
            print("Loaded UP/DN SPikes:", path)
        else:
            up_down=spikified
            print("Spikified Loaded")
        up_down=up_down[:,channels,:]
        
        with open(os.path.join(up_down_path, f'params_{bandpass[0]}_{bandpass[1]}.json'), 'r') as f:
            parameters=json.load(f)
            thresholds=parameters["threshold"]
        print(f'Shape of the filtered data: {filtered_liset.shape}')
        print(f'Shape of the UP/DN data: {up_down.shape}')

        # Reconstruct the signal
        reconstructed_signal=np.zeros((up_down.shape[0],len(channels)))
        
        for i,channel in enumerate(channels):
            # reconstructed_signal[0,i]=filtered_liset[0,i]
            reconstructed_signal[0, i] = 0
            # Loop through the up_down data and reconstruct the signal
            for t in range(1, up_down.shape[0]):
                spike_plus = up_down[t, i,0]
                spike_minus = up_down[t, i,1]
                if spike_plus and spike_minus:
                    both_counter+=1
                if spike_plus == 1:
                    reconstructed_signal[t, i] = reconstructed_signal[t - 1, i] + thresholds[channel]
                elif spike_minus == 1:
                    reconstructed_signal[t, i] = reconstructed_signal[t - 1, i] - thresholds[channel]
                else:
                    reconstructed_signal[t, i] = reconstructed_signal[t - 1, i]
            print(both_counter)
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
        print(f"Ripple start: {ripple_ids[0]}, Ripple end: {ripple_ids[1]}")

    fig,axes=plt.subplots(len(channels),figsize=(int(12),int(4*len(channels))),sharex=True,sharey=True,constrained_layout=True)
    fig.suptitle(f"{title}")
    if len(channels) == 1:
        axes = [axes]  # ensure axes is always a list   
    
    filtered=filtered_liset[ripple_ids[0]:ripple_ids[1],:]
    up_down_ripple=up_down[ripple_ids[0]:ripple_ids[1],:,:]
    time=np.arange(ripple_ids[0],ripple_ids[1])/downsampled_fs
    reconstructed_signal=reconstructed_signal[ripple_ids[0]:ripple_ids[1],:]
    
    for channel in range(len(channels)):
        ax=axes[i]
        ax.plot(time, filtered[:,channel], label='Original Signal', color='black')
        ax.plot(time, reconstructed_signal, label='Reconstructed Signal', linestyle="--",alpha=0.6, color='green')
        if not ripple:
            peak=max(np.max(filtered[:,channel]),0.1)
            trough=min(np.min(filtered[:,channel]),-0.1)
        else:
            peak=max(np.max(filtered[:,channel]),0.5)
            trough=min(np.min(filtered[:,channel]),-0.5)

        mean=np.mean(filtered[:,channel])
        ax.vlines(time[up_down_ripple[:,channel,0] == 1],mean,peak, alpha=0.5,
                color='red', label='Positive Spikes' ,lw=0.5)
        ax.vlines(time[up_down_ripple[:,channel,1] == 1], trough,mean, alpha=0.5,
                color='blue', label='Negative Spikes',lw=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")    
        print("Max region:",max(filtered[:,channel]),"\n Min:",min(filtered[:,channel]))
   
    legend_elements = [
    Line2D([0], [0], color='black', lw=1, label='Filtered Signal'),
    Line2D([0], [0], color='red', lw=1, label='Positive Spikes'),
    Line2D([0], [0], color='blue', lw=1, label='Negative Spikes'),
    Line2D([0], [0], color='green', lw=1, label='Reconstructed Signal'),
    ]

    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))
    plt.show()

def plot_ripple_stats(parent=parent,downsampled_fs=downsampled_fs):
    """
    Plot the ripple statistics

    """
    # Load the ripples from the parent directory
    ripples_list = []  # Initialize an empty list to store the arrays
    time=[]
    ripple_num=[]

    for i in os.listdir(parent):
        dataset_path=os.path.join(parent,i)
        # ripples_list.append(load_ripple_times(dataset_path))  # Append arrays to the list
        liset=liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=0, verbose=False)

        ripples = np.array(liset.ripples_GT)

        if ripples.size == 0:
            print(f"No ripples found in {i}")
            continue
        ripples_list.append(ripples)        

        # Duration is the stop - start
        durations = (ripples[:, 1] - ripples[:, 0])/downsampled_fs*1000 # Convert to milliseconds
        # Calculate the mean and standard deviation of the durations
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        num_ripples=ripples.shape[0]
        time.append(liset.data.shape[0]/downsampled_fs)
        ripple_num.append(num_ripples)
        rate=num_ripples/(liset.data.shape[0]/downsampled_fs) #ripples/s
        print(f"Dataset: {i}, Mean Duration: {mean_duration:.2f} ms, Std Duration: {std_duration:.2f} ms, Rate: {rate:.2f} ripples/s")
    # Concatenate all ripples across datasets
    if len(ripples_list) > 0:
        all_ripples = np.concatenate(ripples_list, axis=0)
        durations = (all_ripples[:, 1] - all_ripples[:, 0]) / downsampled_fs * 1000

        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        mean_rate=sum(ripple_num)/sum(time) # ripples/s
        print(f"\nOverall Mean Duration: {mean_duration:.2f} ms, Std: {std_duration:.2f} ms, Rate: {mean_rate:.2f} ripples/s")

        plt.hist(durations, bins=50, alpha=0.7)
        plt.xlabel('Duration (ms)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Ripple Durations (All Datasets)')
        plt.show()
    else:
        print("No ripple data found.")
    

def plot_reconstruction_whole(spikified=None,filtered=None,save_dir=save_dir,bandpass=bandpass,downsampled_fs=downsampled_fs,parent=parent,save=save, channels=None,window=[],id=0):
    """
    Plot the reconstruction of the UP/DOWN spikes

    """
    both_counter=0
    ############## Load the data ##################
    dirs=os.listdir(parent)
    datasets=[dirs[id]]
    for dataset in datasets:
        dataset_path=os.path.join(parent,dataset)
        up_down_path=os.path.join(save_dir,dataset,f"{downsampled_fs}")
        liset= liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=0, verbose=False)
        print("Loaded LFPs:",dataset_path)
        if channels is None:
            channels=[i for i in range(liset.data.shape[1])]
        if filtered is None:
            filtered_liset=np.zeros((int(liset.data.shape[0]),len(channels)))
            for i,channel in enumerate(channels):
                print("Channel:", channel+1)
                filtered_liset[:,i]=bandpass_filter(liset.data[:,channel], bandpass=bandpass, fs=liset.fs)
        else:
            filtered_liset=filtered[:,channels]
            print("Filtered Loaded")

        if spikified is None:
            path=os.path.join(up_down_path, f'data_up_down_{bandpass[0]}_{bandpass[1]}.npy')
            up_down= np.load(path)
            print("Loaded UP/DN SPikes:", path)
        else:
            up_down=spikified
            print("Spikified Loaded")
        up_down=up_down[:,channels,:]
        
        with open(os.path.join(up_down_path, f'params_{bandpass[0]}_{bandpass[1]}.json'), 'r') as f:
            parameters=json.load(f)
            thresholds=parameters["threshold"]
        print(f'Shape of the filtered data: {filtered_liset.shape}')
        print(f'Shape of the UP/DN data: {up_down.shape}')

       
        # Reconstruct the signal
        reconstructed_signal=np.zeros((up_down.shape[0],len(channels)))
        print("R")
        for i,channel in enumerate(channels):
            # reconstructed_signal[0,i]=filtered_liset[0,i]
            reconstructed_signal[0, i] = 0
            # Loop through the up_down data and reconstruct the signal
            for t in range(1, up_down.shape[0]):
                spike_plus = up_down[t, i,0]
                spike_minus = up_down[t, i,1]
                if spike_plus and spike_minus:
                    both_counter+=1
                if spike_plus == 1:
                    reconstructed_signal[t, i] = reconstructed_signal[t - 1, i] + thresholds[channel]
                elif spike_minus == 1:
                    reconstructed_signal[t, i] = reconstructed_signal[t - 1, i] - thresholds[channel]
                else:
                    reconstructed_signal[t, i] = reconstructed_signal[t - 1, i]
            print(both_counter)

    ripple_ids=liset.ripples_GT
    fig,axes=plt.subplots(len(channels),figsize=(int(12),int(4*len(channels))),sharex=True,sharey=False,constrained_layout=True)
    fig.suptitle(f"Reconstructed signal vs UP/DOWN Spikes")
    if len(channels) == 1:
        axes = [axes]  # ensure axes is always a list   
    
    filtered=filtered_liset[window[0]:window[1],:]
    up_down_ripple=up_down[window[0]:window[1],:,:]
    print("Filtered shape:",filtered.shape)
    print("Reconstructed shape:",reconstructed_signal.shape)
    print("UP/DOWN shape:",up_down_ripple.shape)

    time = np.linspace(window[0] / downsampled_fs, window[1] /  downsampled_fs, filtered.shape[0])
    print("Time length", len(time))
    reconstructed_signal=reconstructed_signal[window[0]:window[1],:]
    mask = (liset.ripples_GT[:, 1] >= window[0]) & (liset.ripples_GT[:, 0] <= window[1])
    window_ripples = liset.ripples_GT[mask]
    
    for channel in range(len(channels)):
        min_val=filtered[:,channel].min()*1.2
        max_val=filtered[:,channel].max()*1.2
        ax=axes[channel]

        ax.plot(time, filtered[:,channel], label='Original Signal', color='black')
        ax.plot(time, reconstructed_signal[:,channel], label='Reconstructed Signal', linestyle="--",alpha=0.6, color='green')
        for ripple in window_ripples:
            fill_GT = ax.fill_between([ripple[0] / downsampled_fs, ripple[1] / downsampled_fs],  min_val, max_val, color="lightblue", alpha=0.3)
        
        peak=max(np.max(filtered[:,channel]),0.5)
        trough=min(np.min(filtered[:,channel]),-0.5)

        mean=np.mean(filtered[:,channel])
        ax.vlines(time[up_down_ripple[:,channel,0] == 1],mean,peak, alpha=0.5,
                color='red', label='Positive Spikes' ,lw=0.5)
        ax.vlines(time[up_down_ripple[:,channel,1] == 1], trough,mean, alpha=0.5,
                color='blue', label='Negative Spikes',lw=0.5)
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")    
        print("Max region:",max(filtered[:,channel]),"\n Min:",min(filtered[:,channel]))
   
    legend_elements = [
    Line2D([0], [0], color='black', lw=1, label='Filtered Signal'),
    Line2D([0], [0], color='red', lw=1, label='Positive Spikes'),
    Line2D([0], [0], color='blue', lw=1, label='Negative Spikes'),
    Line2D([0], [0], color='green', lw=1, label='Reconstructed Signal'),
    ]

    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))
    plt.show()
    return fig,axes


plot_reconstruction_whole(save_dir=save_dir,bandpass=bandpass,downsampled_fs=10000,parent=parent,save=save, channels=[1,],window=[20000,50000],id=0)