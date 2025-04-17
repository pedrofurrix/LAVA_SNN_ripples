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
std, mean = ripples_std(parent)
processed_ripples = []
downsampled_fs= 4000
chunk_size     = 10000000

# Define saving directory
save_dir = os.path.join(os.path.dirname(__file__),"train_pedro","dataset")
os.makedirs(save_dir, exist_ok=True)  # <-- creates directory if it doesn't exist

# downsampled_fs = int(sys.argv[1])
# bandpass = np.array(sys.argv[2].split('_')).astype(int)

# Set the length of the signals 
# It has to be suitable for the maximum events posible, so it will be the mean_len + std
samples_per_signal = round((mean + std) * downsampled_fs)

half_S = int(samples_per_signal/2)
print(samples_per_signal)

# Several BP filters (Pedro)
# bp_filters=np.array([[1,10],[100,250],[250,500]])
bp_filters=np.array([[250,500]])
for bandpass in bp_filters:
    # True positives
    print('Extracting True Positive events ...')
    for i in os.listdir(parent):
        print(i)
        # Restart loop variables
        dataset_path = os.path.join(parent, i)
        start = 0
        keep_looping = True
        # Loop until all ripples are saved in the list
        while keep_looping:
            print(f'Start: {start}', end='\r', flush=True)
            liset = liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=start, numSamples=chunk_size, verbose=False)
            if hasattr(liset, 'data'):
                # Update the reading start for the next loop
                start += chunk_size
                # Loop throough the ripples found in liset class (the ones in the range of the selected samples)
                for ripple in liset.ripples_GT:
                    middle_idx = middle(ripple)
                    ripple_signal = liset.data[middle_idx - half_S : middle_idx + half_S, :]
                    for idx in range(ripple_signal.shape[1]):
                        ripple_signal[:, idx] = bandpass_filter(ripple_signal[:, idx], bandpass=bandpass, fs=liset.fs)
                    best_filtered_channel = most_active_channel(ripple_signal)
                    processed_ripples.append(best_filtered_channel)
            else:
                keep_looping = False
                start = 0

    processed_ripples = np.array(processed_ripples)
    save_positives=os.path.join(save_dir, f'true_positives_{bandpass[0]}_{bandpass[1]}Hz.npy')
    np.save(save_positives, arr=processed_ripples, allow_pickle=True)
    print('Saved True Positives!')
    time.sleep(1)


    # True negatives
    print('Extracting True Negative events ...')
    true_negatives = []
    num_samples_per_chunk = 150
    margin_from_ripple = int(10000/4000*downsampled_fs) # 2.5s # Sample margin from ripple to avoid overlap with true positives # We should choose a value that depends on the frequency

    for i in os.listdir(parent): 
        print(i) 
        start = 0
        keep_looping = True
        dataset_path = os.path.join(parent, i)

        # Loop until all ripples are saved in the list
        while keep_looping:
            true_positives = []
            print(f'Start: {start}', end='\r', flush=True)
            liset = liset_tk(dataset_path, shank=3, downsample=downsampled_fs, start=start, numSamples=chunk_size, verbose=False)
            if hasattr(liset, 'data'):
                # Update the reading start for the next loop
                start += chunk_size
                arr = np.array(range(int((chunk_size) / liset.fs_conv_fact)))
                if arr.shape[0] > liset.data.shape[0]:
                    arr = np.array(range(int((liset.file_samples - int(liset.file_samples / (chunk_size/10)) * (chunk_size/10))/liset.fs_conv_fact)))

                # Example true positive values
                for ripple in liset.ripples_GT:
                    ripple_start = int(ripple[0])
                    ripple_end = int(ripple[1])
                    positives = np.array(range(ripple_start - margin_from_ripple, ripple_end + margin_from_ripple))
                    true_positives = np.concatenate([true_positives, positives])
                
                # Create a mask to filter out true positive values
                mask = np.isin(arr, true_positives, invert=True)

                # Apply the mask to get only values that are not true positives
                filtered_arr = arr[mask]

                # Remove values that would cause a ValueError due to not having enough samples
                filtered_arr = filtered_arr[filtered_arr + samples_per_signal <= liset.data.shape[0]]

                # Now, you can sample random values from filtered_arr
                random_samples = np.random.choice(filtered_arr, size=num_samples_per_chunk, replace=False).astype(int)
                for sample in random_samples:
                    window = deepcopy(liset.data[sample : sample + samples_per_signal][:])
                    for i, chann in enumerate(window.transpose()):
                        window[:, i] = bandpass_filter(window[:, i], bandpass=bandpass, fs=liset.fs)
                    best_channel = most_active_channel(window)
                    true_negatives.append(best_channel)
            else:
                keep_looping = False

    # Convert to numpy array and save the values
    for i in range(len(true_negatives)):
        if len(true_negatives[i]) != samples_per_signal:
            print('Signal length not equal to samples_per_signal!')
            print(true_negatives[i])
            print(i)
            print(len(true_negatives[i]))
    true_negatives = np.array(true_negatives)
    save_negatives=os.path.join(save_dir, f'true_negatives_{bandpass[0]}_{bandpass[1]}Hz.npy')
    np.save(save_negatives, arr=true_negatives, allow_pickle=True)
    print('Saved True Negatives!')