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
parent = r"C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\CNN_TRAINING_SESSIONS" # Modify this to your data path folder
std, mean = ripples_std(parent) # 61 ms
processed_ripples = []
downsampled_fs= 4000

up_down_path= os.path.join(os.path.dirname(__file__),"train_pedro","dataset_up_down")
bandpass=[100,250]
threshold=0.1

def concat_dataset( downsampled_fs=downsampled_fs,parent=parent,up_down_path=up_down_path,bandpass=bandpass,threshold=threshold):
    """
    
    Concatenate all the channels into a single array [Timesteps x Num_channels, 2], to extract the windows
    Remove the channels with the baseline below the threshold value
    Save ground truth

    """

    save_dir = os.path.join(up_down_path,str(downsampled_fs))
    os.makedirs(save_dir, exist_ok=True)  # <-- creates directory if it doesn't exist

    start=0
    labels=[]

    for i in os.listdir(parent):
        with open(os.path.join(up_down_path,i,str(downsampled_fs), f'params_{bandpass[0]}_{bandpass[1]}.json'), 'r') as f:
            parameters=json.load(f)
            thresholds=parameters["threshold"]
        print (thresholds)    
        dataset_path = os.path.join(parent, i)
        up_down_path=os.path.join(up_down_path,i,str(downsampled_fs),f"data_up_down_{bandpass[0]}_{bandpass[1]}.npy")
        liset = liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=start, verbose=False) # Get the Ripple times (there were other ways, but this one is okay...)
        up_down=np.load(up_down_path)

        if hasattr(liset, 'data'):
                # Update the reading start for the next loop
                # Loop throough the ripples found in liset class (the ones in the range of the selected samples)
                channels=[]
                for channel in range(liset.data.shape(1)):
                    if thresholds[channel] != threshold:
                        channels.append(channel)

                # Remove channels with a baseline below the threshold:
                print("Channels to consider:",channels)
                up_down_data=up_down[:,channels,:]
                

                          
                     
                for ripple in liset.ripples_GT:

                    middle_idx = middle(ripple)
                    ripple_signal = liset.data[middle_idx - half_S : middle_idx + half_S, :]
                    for idx in range(ripple_signal.shape[1]):
                        ripple_signal[:, idx] = bandpass_filter(ripple_signal[:, idx], bandpass=bandpass, fs=liset.fs)
                    best_filtered_channel = most_active_channel(ripple_signal)
                    processed_ripples.append(best_filtered_channel)
        else:
            print(f"Dataset {i} has no data")
            return