import sys
import os
import json
liset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../liset_tk'))

sys.path.insert(0, liset_path)
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

sys.path.insert(-1,parent_path)

from train_pedro.make_windows2 import TrainData
from utils_encoding import *
from liset_aux import ripples_std, middle
from signal_aid import most_active_channel, bandpass_filter
from liset_paper import liset_paper as liset_tk
import os
import numpy as np
from copy import deepcopy
import time

# Define general variables
# parent = r"C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\CNN_TRAINING_SESSIONS" # Modify this to your data path folder
parent = r"C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\Download_from_paper"  # Modify this to your data path folder

### HOME PC
# parent=r"E:\neurospark_mat\CNN_TRAINING_SESSIONS"
# parent=r"E:\neurospark_mat\Download_from_paper"

# std, mean = ripples_std(parent) # 61 ms
# processed_ripples = []
# downsampled_fs= 1000

# up_down_path= os.path.join(os.path.dirname(__file__),"train_pedro","dataset_up_down")
# bandpass=[100,250]
# threshold=0.1
# save=True

# def concat_dataset(downsampled_fs=downsampled_fs,parent=parent,up_down_path=up_down_path,bandpass=bandpass,threshold=threshold,save=save):
   
#     """    
#     Concatenate all the channels into a single array [Timesteps x Num_channels, 2], to extract the windows
#     Remove the channels with the baseline below the threshold value
#     Save ground truth
#     """

#     save_dir = os.path.join(up_down_path,str(downsampled_fs))
#     os.makedirs(save_dir, exist_ok=True)  # <-- creates directory if it doesn't exist
#     concatenated_data=[]
#     ripple_position=0
#     ripples_concat=[]
#     total_length = 0

#     for i in os.listdir(parent):
#         path_dataset=os.path.join(up_down_path,str(i),str(downsampled_fs))
#         print(path_dataset)
#         with open(os.path.join(path_dataset, f'params_{bandpass[0]}_{bandpass[1]}.json'), 'r') as f:
#             parameters=json.load(f)
#             thresholds=parameters["threshold"]
#         print(thresholds)    
#         dataset_path = os.path.join(parent, i)
#         up_down_file=os.path.join(path_dataset,f"data_up_down_{bandpass[0]}_{bandpass[1]}.npy")
#         ripples_file=os.path.join(path_dataset,f"ripples.npy")
#         ripples=np.load(ripples_file)
#         up_down=np.load(up_down_file)
#         ripples.sort(axis=0)
#         valid_channels = [ch for ch in range(up_down.shape[1]) if thresholds[ch] >= threshold]
#         # Keep track of the ripples in the valid channels (create an array with the concatenated ripples)
        
#         print(f"  → {len(valid_channels)} channels kept (out of {up_down.shape[1]})")

#         if valid_channels:
#             filtered = up_down[:, valid_channels, :]         # shape: [T, valid_C, 2]
#             reshaped = filtered.reshape(-1, 2)               # shape: [T * valid_C, 2]
#             concatenated_data.append(reshaped)
        
#         # Adjust ripple indices to account for both dataset length and channel offset
#         adjusted_ripples = [
#             ripple + total_length + (channel_idx * up_down.shape[0])
#             for channel_idx in range(len(valid_channels))
#             for ripple in ripples
#         ]
#         ripples_concat.extend(adjusted_ripples)
#         total_length += up_down.shape[0] * len(valid_channels)   
#     concatenated_data = np.concatenate(concatenated_data, axis=0)  # shape: [T * valid_C, 2]   
#     ripples_both=np.array(ripples_concat)  # shape: [N, 2]
#     print(f"Total concatenated ripples: {len(ripples_both)}") 
#     print(f"Total concatenated data: {len(concatenated_data)}")
#     print(f"Ripples shape:", ripples_both.shape)
#     print("Data Shape:", concatenated_data.shape)
#     if save:
#         np.save(os.path.join(save_dir, f"concat_both.npy"), concatenated_data)
#         np.save(os.path.join(save_dir, f"ripples_both.npy"), ripples_both)

#     return concatenated_data,ripples_both

save=True
def concat_dataset_final(parent=parent,save=save):
    config={}
    ripples_concat=[]
    spikified_concat=[]
    filtered_concat=[]
    total_length = 0
    ### Load configuration
    params_dir=os.path.join(os.path.dirname(__file__),"train_pedro","windowed_data")
    with open(os.path.join(params_dir, "config.json"), 'r') as f:
        parameters = json.load(f)
        downsampled_fs = parameters["downsampled_fs"]
        bandpass = parameters["bandpass"]
        factor = parameters["factor"]
        fraction = 1-parameters["fraction"]
        refractory = parameters["refractory"]
        print(f"Downsampled fs: {downsampled_fs}, Bandpass: {bandpass}, Factor: {factor}, Fraction: {fraction}, Refractory: {refractory}") 
    
    config=fill_config(config, parameters)

    for dataset in os.listdir(parent):
        print(f"Processing dataset: {dataset}")
        dataset_path = os.path.join(parent, dataset)
        liset= liset_tk(dataset_path, shank=1, downsample=False, verbose=False)
        liset=TrainData(liset,fraction,beginning=False)
        downsample_factor=liset.fs//downsampled_fs
        ripples=np.array(liset.ripples_GT)//downsample_factor
        # Downsample ripples
        ripples=ripples//factor

        print("Dataset: ", dataset)
        print("data shape: ", liset.data.shape)
        print("ripples shape: ", ripples.shape)
        # print("Head of data_concat: ", data[:10][:])
        # print("Head of ripples_concat: ", ripples[:10])
        ripples = ripples[np.argsort(ripples[:, 0])]
        thresholds=parameters[dataset]["thresholds"]
        print("Thresholds: ", thresholds)
        for channel in range(liset.data.shape[1]):
            threshold=thresholds[str(channel)]
            channel_signal = liset.data[:, channel]
            filtered_liset=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
            if downsample_factor>1:
                filtered_liset=decimation_downsampling(filtered_liset,downsample_factor)
            spikified=up_down_channel(filtered_liset,threshold,downsampled_fs,refractory)
            # spikified[:, channel, :]=up_down_channel_SF(filtered_liset,thresholds[channel],downsampled_fs,refractory)
            if factor>1:
                downsampled,spikes_lost=extract_spikes_downsample(spikified,factor)
                filtered_liset=decimation_downsampling(filtered_liset,factor)
            else:
                downsampled=spikified

            spikified_concat.append(downsampled)
            filtered_concat.append(filtered_liset)
            
            adjusted_ripples = [
            ripple + total_length
            for ripple in ripples
            ]
            total_length += filtered_liset.shape[0]
            ripples_concat.append(adjusted_ripples)

    concatenated_spikes = np.concatenate(spikified_concat,axis=0)  # shape: [T * valid_C, 2]   
    ripples_both=np.concatenate(ripples_concat,axis=0)  # shape: [N, 2]
    filtered_both=np.concatenate(filtered_concat,axis=0)  # shape: [N]
    print(f"Total concatenated ripples: {len(ripples_both)}")
    print(f"Total concatenated spikes: {np.sum(concatenated_spikes)}")
    print(f"Total concatenated filtered: {len(filtered_both)}")
    print(f"Ripples shape:", ripples_both.shape)
    print("Spikes Shape:", concatenated_spikes.shape)
    config=get_stats(config,concatenated_spikes,ripples_both)
    if save:
        data_dir=os.path.join(os.path.dirname(__file__),"train_pedro","dataset_up_down",f"{downsampled_fs}_{int(downsampled_fs/factor)}")
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, f"concat_spikes.npy"), concatenated_spikes)
        np.save(os.path.join(data_dir, f"concat_ripples.npy"), ripples_both)
        np.save(os.path.join(data_dir, f"concat_data.npy"), filtered_both)
        with open(os.path.join(data_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Data saved in {data_dir}")
    else:
        return concatenated_spikes, ripples_both, filtered_both

def fill_config(config, parameters, keys_to_exclude=None):
    if keys_to_exclude is None:
        keys_to_exclude = [
        "num_hfo_windows",
        "num_windows",
        "percentage_hfo_windows",
        "average_hfo_up_counts",
        "average_hfo_down_counts",
        "average_total_hfo_counts",
        "average_nonhfo_up_counts",
        "average_nonhfo_down_counts",
        "average_total_nonhfo_counts",
        "average_total_counts",
        "median_hfo_counts",
        "median_nonhfo_counts"
        ]

    for key, value in parameters.items():
        if key not in keys_to_exclude:
            config[key] = value

    return config

def get_stats(config,concatenated_spikes,ripples_both):
    """
    Get statistics from the concatenated spikes and ripples
    """
    num_spikes=np.sum(concatenated_spikes)
    num_ripples=len(ripples_both)
    spikes_per_ripple = []
    ripple_length=[]
    for ripple in ripples_both:
        start, end = ripple
        spikes_in_ripple = np.sum(concatenated_spikes[start:end])
        spikes_per_ripple.append(spikes_in_ripple)
        ripple_length.append(end - start)
        
    ripple_length=np.array(ripple_length)
    spikes_per_ripple=np.array(spikes_per_ripple)
    average_spikes_per_ripple = np.mean(spikes_per_ripple)
    median_spikes_per_ripple = np.median(spikes_per_ripple)
    max_spikes_per_ripple = np.max(spikes_per_ripple)
    min_spikes_per_ripple = np.min(spikes_per_ripple)
    average_spike_rate_per_ripple = np.sum(spikes_per_ripple) /np.sum(ripple_length) if len(ripple_length) > 0 else 0
    median_spike_rate_per_ripple = np.median(spikes_per_ripple/ripple_length) if len(ripple_length) > 0 else 0
    max_spike_rate_per_ripple = np.max(spikes_per_ripple/ripple_length) if len(ripple_length) > 0 else 0
    min_spike_rate_per_ripple = np.min(spikes_per_ripple/ripple_length) if len(ripple_length) > 0 else 0

    average_spike_rate_overall= num_spikes / len(concatenated_spikes)
    average_spike_rate_non_ripple= (num_spikes - np.sum(spikes_per_ripple)) / (len(concatenated_spikes) - np.sum(ripple_length))


    overall_duration = len(concatenated_spikes)
    non_ripple_duration = overall_duration - np.sum(ripple_length)
    non_ripple_spikes = num_spikes - np.sum(spikes_per_ripple)

    config["num_ripples"] = int(num_ripples)
    config["num_spikes"] = int(num_spikes)
    config["average_spikes_per_ripple"] = float(average_spikes_per_ripple)
    config["median_spikes_per_ripple"] = float(median_spikes_per_ripple)
    config["max_spikes_per_ripple"] = float(max_spikes_per_ripple)
    config["min_spikes_per_ripple"] = float(min_spikes_per_ripple)
    config["average_spike_rate_per_ripple"] = float(average_spike_rate_per_ripple * 1000)
    config["median_spike_rate_per_ripple"] = float(median_spike_rate_per_ripple * 1000)
    config["max_spike_rate_per_ripple"] = float(max_spike_rate_per_ripple * 1000)
    config["min_spike_rate_per_ripple"] = float(min_spike_rate_per_ripple * 1000)
    config["average_spike_rate_overall"] = float(average_spike_rate_overall * 1000)
    config["average_spike_rate_non_ripple"] = float(average_spike_rate_non_ripple * 1000)
    config["overall_duration"] = float(overall_duration)
    config["non_ripple_duration"] = float(non_ripple_duration)
    config["non_ripple_spikes"] = int(non_ripple_spikes)

    return config



def plot_dataset_testing(window,downsampled_fs="30000_1000",title='Live Test Data', xlabel='Time (s)', ylabel='Value',input=True):
    parent_dir=os.path.dirname(__file__)
    # Load the spike data, gt and original data from npy files
    data_dir=os.path.join(parent_dir,"train_pedro","dataset_up_down",str(downsampled_fs))
    spikes= np.load(os.path.join(data_dir, f'concat_spikes.npy'))
    gt = np.load(os.path.join(data_dir, f'concat_ripples.npy'))
    data=np.load(os.path.join(data_dir, f'concat_data.npy'))

    # Create the figure and axis

    fig, ax = plt.subplots(figsize=(20, 6))
    
    if window is not None:
        start, end = window
        # Adjust the data, spikes, gt, and outputspikes to the specified window
        data = data[start:end]
        spikes = spikes[start:end]

        # Adjust ground truth events (gt): select events that overlap with the window
        # gt[:, 0] = start of ripple, gt[:, 1] = end of ripple
        # Keep ripples that overlap the window [start, end)
        gt = gt[(gt[:, 1] >= start) & (gt[:, 0] < end)]
        # Shift the ripple times to be relative to the window
        gt = gt



    # Convert to seconds
    up_spike_times = np.where(spikes[:, 0] == 1)[0]
    down_spike_times = np.where(spikes[:, 1] == 1)[0]
        # Use the same time base (in seconds)
    up_spike_times_sec = up_spike_times / 1000
    down_spike_times_sec = down_spike_times / 1000
    # Convert ground truth ripples to seconds
    gt_sec = gt / 1000



    time = np.arange(start, end) / 1000  # In seconds
    # Plot the original data
    ax.plot(time,data, label='Original Data', color='black', alpha=1)
    
    # Plot the Input Up and Down Spikes
    if input:
        ax.vlines(up_spike_times_sec,0,3, color='green', alpha=0.3,label='Up Spikes')
        ax.vlines(down_spike_times_sec,-3,0, color='red',alpha=0.3, label='Down Spikes')

    # Plot the Ground Truth Ripples
    for i,ripple in enumerate(gt_sec):
        label = 'Ground Truth Ripple' if i == 0 else None  # Add label only to the first
        ax.fill_between([ripple[0], ripple[1]], -5,5, color='yellow', alpha=0.2, label=label)
    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    plt.show()
    return fig, ax


# concat_dataset_final(parent=parent,save=True)
# plot_dataset_testing(window=(0,10000),downsampled_fs="30000_1000",title='Live Test Data', xlabel='Time (s)', ylabel='Value',input=True)
