import sys
import os
import json
liset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../liset_tk'))

sys.path.insert(0, liset_path)

from liset_aux import ripples_std, middle
from signal_aid import most_active_channel, bandpass_filter
from liset_tk import liset_tk
import os
import numpy as np
from copy import deepcopy
import time

# Define general variables
# parent = r"C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\CNN_TRAINING_SESSIONS" # Modify this to your data path folder

### HOME PC
parent=r"E:\neurospark_mat\CNN_TRAINING_SESSIONS"

std, mean = ripples_std(parent) # 61 ms
processed_ripples = []
downsampled_fs= 1000

up_down_path= os.path.join(os.path.dirname(__file__),"train_pedro","dataset_up_down")
bandpass=[100,250]
threshold=0.1
save=True

def concat_dataset(downsampled_fs=downsampled_fs,parent=parent,up_down_path=up_down_path,bandpass=bandpass,threshold=threshold,save=save):
   
    """    
    Concatenate all the channels into a single array [Timesteps x Num_channels, 2], to extract the windows
    Remove the channels with the baseline below the threshold value
    Save ground truth
    """

    save_dir = os.path.join(up_down_path,str(downsampled_fs))
    os.makedirs(save_dir, exist_ok=True)  # <-- creates directory if it doesn't exist
    concatenated_data=[]
    ripple_position=0
    ripples_concat=[]
    total_length = 0

    for i in os.listdir(parent):
        path_dataset=os.path.join(up_down_path,str(i),str(downsampled_fs))
        print(path_dataset)
        with open(os.path.join(path_dataset, f'params_{bandpass[0]}_{bandpass[1]}.json'), 'r') as f:
            parameters=json.load(f)
            thresholds=parameters["threshold"]
        print(thresholds)    
        dataset_path = os.path.join(parent, i)
        up_down_file=os.path.join(path_dataset,f"data_up_down_{bandpass[0]}_{bandpass[1]}.npy")
        ripples_file=os.path.join(path_dataset,f"ripples.npy")
        ripples=np.load(ripples_file)
        up_down=np.load(up_down_file)
        ripples.sort(axis=0)
        valid_channels = [ch for ch in range(up_down.shape[1]) if thresholds[ch] >= threshold]
        # Keep track of the ripples in the valid channels (create an array with the concatenated ripples)
        
        print(f"  → {len(valid_channels)} channels kept (out of {up_down.shape[1]})")

        if valid_channels:
            filtered = up_down[:, valid_channels, :]         # shape: [T, valid_C, 2]
            reshaped = filtered.reshape(-1, 2)               # shape: [T * valid_C, 2]
            concatenated_data.append(reshaped)
        
        # Adjust ripple indices to account for both dataset length and channel offset
        adjusted_ripples = [
            ripple + total_length + (channel_idx * up_down.shape[0])
            for channel_idx in range(len(valid_channels))
            for ripple in ripples
        ]
        ripples_concat.extend(adjusted_ripples)
        total_length += up_down.shape[0] * len(valid_channels)   
    concatenated_data = np.concatenate(concatenated_data, axis=0)  # shape: [T * valid_C, 2]   
    ripples_both=np.array(ripples_concat)  # shape: [N, 2]
    print(f"Total concatenated ripples: {len(ripples_both)}") 
    print(f"Total concatenated data: {len(concatenated_data)}")
    print(f"Ripples shape:", ripples_both.shape)
    print("Data Shape:", concatenated_data.shape)
    if save:
        np.save(os.path.join(save_dir, f"concat_both.npy"), concatenated_data)
        np.save(os.path.join(save_dir, f"ripples_both.npy"), ripples_both)

    return concatenated_data,ripples_both

