import liset_tk
import matplotlib.pyplot as plt
import numpy as np
def downsample_data_test(data,fs=30000,downsampled_fs=[]):
    all_fs = [fs] + downsampled_fs  # Include original fs
    n = len(all_fs)
    fig, axes = plt.subplots(nrows=n, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot
        
    for i,freq in enumerate(all_fs):
        if freq==fs:
            downsampled_data=data
            label=f'Original Data ({freq} Hz)'
        else:
            downsampled_data=downsample_data(data, fs, downsampled_fs)



        axes[i].plot(downsampled_data[:,0])
        axes[i].set_title(label)
        axes[i].set_ylabel('Amplitude')

