# Import the class from the liset_tk.py file
import sys
from liset_tk import liset_tk
import os
print(os.getcwd())
# Define the path to your data


path = r"C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\CNN_TRAINING_SESSIONS\Amigo2_1_hippo_2019-07-11_11-57-07_1150um" # Modify this to your data path folder
liset = liset_tk(data_path=path, shank=3, downsample=1250, verbose=True, numSamples=1000000)

print(f'Original Sampling Frequency: {liset.original_fs} Hz')
print(f'Current Sampling Frequency: {liset.fs} Hz')
print(f'Shape of the loaded data: {liset.data.shape}')
print(f'Duration of the loaded data: {liset.duration} seconds')
print('\n')
if liset.has_ripples:
    print(f'Number of loaded GT ripples: {len(liset.ripples_GT)}')
    print(f'Overview of the ripples:\n\n{liset.ripples_GT[0:5]}\n...')




# Plot the loaded channels in a time window.
window = [2000, 3000]

# Play with the offset and extend parameters to zoom in and out of the data.
offset = 3
extend = 100
liset.plot_event(window, 
                 offset=offset, 
                 extend=extend, 
                 show_ground_truth=True, 
                 label='Ripple Activity',
                 title='Ripple Activity in the Hippocampus'
                    )