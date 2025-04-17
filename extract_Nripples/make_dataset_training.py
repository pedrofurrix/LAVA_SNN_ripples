import sys
import os
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
save_dir = os.path.join(os.path.dirname(__file__),"train_pedro","dataset_up_down","trainSNN")
up_down_path= os.path.join(os.path.dirname(__file__),"train_pedro","dataset_up_down")
bandpass=[100,250]
def make_training_dataset(save_dir=save_dir, downsampled_fs=downsampled_fs,parent=parent,up_down_path=up_down_path,bandpass=bandpass):
    os.makedirs(save_dir, exist_ok=True)  # <-- creates directory if it doesn't exist
    samples_per_signal = 0.15*downsampled_fs # 150 ms
    start=0
    for i in os.listdir(parent):
        dataset_path = os.path.join(parent, i)
        up_down_path=os.path.join(up_down_path,i,f"{downsampled_fs}",f"data_up_down_{bandpass[0]}_{bandpass[1]}.npy")
        liset = liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=start, verbose=False) # Get the Ripple times (there were other ways, but this one is okay...)
        up_down=np.load(up_down_path)
        ground_truth=[]
        if hasattr(liset, 'data'):
                # Update the reading start for the next loop
                # Loop throough the ripples found in liset class (the ones in the range of the selected samples)
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