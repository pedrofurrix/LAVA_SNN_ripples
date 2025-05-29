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

    for dataset in os.listdir(parent):
        print(f"Processing dataset: {dataset}")
        dataset_path = os.path.join(parent, dataset)
        liset= liset_tk(dataset_path, shank=1, downsample=False, verbose=False)
        liset=TrainData(liset,fraction,beginning=False)

        downsample_factor=liset.fs//downsampled_fs
        ripples=np.array(liset.ripples_GT)//downsample_factor
        spikified=np.zeros((liset.data.shape[0]//downsample_factor, liset.data.shape[1], 2))
        filtered=np.zeros((liset.data.shape[0]//downsample_factor, liset.data.shape[1]))
        downsampled=np.zeros((spikified.shape[0]//factor, liset.data.shape[1], 2))

        print("Dataset: ", dataset)
        print("data shape: ", liset.data.shape)
        print("ripples shape: ", ripples.shape)
        # print("Head of data_concat: ", data[:10][:])
        # print("Head of ripples_concat: ", ripples[:10])
        ripples = ripples[np.argsort(ripples[:, 0])]
        thresholds=parameters[dataset]["thresholds"]

        for channel in range(liset.data.shape[1]):
            threshold=thresholds[str(channel)]
            channel_signal = liset.data[:, channel]
            filtered_liset=bandpass_filter(channel_signal, bandpass=bandpass, fs=liset.fs)
            if downsample_factor>1:
                filtered_liset=decimation_downsampling(filtered_liset,downsample_factor)
            spikified[:, channel, :]=up_down_channel(filtered_liset,threshold,downsampled_fs,refractory)
            # spikified[:, channel, :]=up_down_channel_SF(filtered_liset,thresholds[channel],downsampled_fs,refractory)
            if factor>1:
                downsampled[:,channel,:],spikes_lost=extract_spikes_downsample(spikified[:,channel,:],factor)
                filtered_liset=decimation_downsampling(filtered_liset,factor)
            else:
                downsampled[:,channel,:]=spikified[:,channel,:]
            spikified_concat.append(downsampled[:, channel, :])
            filtered_concat.append(filtered_liset)
            adjusted_ripples = [
            ripple + total_length
            for ripple in ripples
            ]
            total_length += filtered_liset.data.shape[0]
            ripples_concat.append(adjusted_ripples)

    concatenated_spikes = np.concatenate(spikified_concat,axis=0)  # shape: [T * valid_C, 2]   
    ripples_both=np.concatenate(ripples_concat,axis=0)  # shape: [N, 2]
    filtered_both=np.concatenate(filtered_concat,axis=0)  # shape: [N]
    print(f"Total concatenated ripples: {len(ripples_both)}")
    print(f"Total concatenated spikes: {len(concatenated_spikes)}")
    print(f"Total concatenated filtered: {len(filtered_both)}")
    print(f"Ripples shape:", ripples_both.shape)
    print("Spikes Shape:", concatenated_spikes.shape)
    if save:
        data_dir=os.path.join(os.path.dirname(__file__),"train_pedro","dataset_up_down",f"{downsampled_fs}_{int(downsampled_fs/factor)}")
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, f"concat_spikes.npy"), concatenated_spikes)
        np.save(os.path.join(data_dir, f"concat_ripples.npy"), ripples_both)
        np.save(os.path.join(data_dir, f"concat_data.npy"), filtered_both)
        print(f"Data saved in {data_dir}")
    else:
        return concatenated_spikes, ripples_both, filtered_both

concat_dataset_final(parent=parent,save=True)