from liset_data_reader.liset_paper import liset_paper
import numpy as np
import matplotlib.pyplot as plt
import os
from signal_aid import bandpass_filter

parent= r"C:\PedroFelix\Download_from_paper" # Modify this to your data path folder
datasets=os.listdir(parent)
datasets = [d for d in datasets if not d.startswith('.') and not d.startswith('~')]
print("Datasets found:", datasets)

dataset=os.path.join(parent, datasets[0])
print("Loading dataset:", dataset)
liset=liset_paper(dataset, shank=1, downsample=False, start=0, verbose=False)
print("Liset data shape:", liset.data.shape)
numsamples=liset.data.shape[0]
print("Dataset loaded:", dataset)
print("Dataset Time:",numsamples / liset.fs, "seconds")
time_seconds = numsamples / liset.fs
minutes= np.floor(time_seconds/ 60)
extra_seconds = time_seconds%60
print("Dataset time:", int(minutes), "minutes", round(extra_seconds,2), "seconds")

def plot_liset(liset,downsampled_fs,window,filter=True,offset=2):
    """
    Plot the LISet data with a specified downsampled frequency and window.
    
    Parameters:
    - data: The LISet data to plot.
    - downsampled_fs: The downsampled frequency for the plot.
    - window: The time window for the plot in seconds.
    """
    data=liset.data
    print("Data shape:", data.shape)
    ripples=liset.ripples_GT
    print("Ripples shape:", ripples.shape)
    window=np.array(window)  # Ensure window is a numpy array
    liset_time=liset.data.shape[0] / liset.fs  # Total time in seconds
    start=max(0, window[0])  # Ensure start is not negative
    end=min(liset_time, window[1])  # Ensure end does not exceed data length
    window_samples=(start,end)*liset.fs

    
    window_len=window_samples[1]-window_samples[0] # Convert window to samples
    downsample= int(liset.fs / downsampled_fs)

    
    data= data[int(window_samples[0]):int(window_samples[1]), :]  # Select the window of data
    filtered=np.zeros(data.shape) 
    ripples_in_window=[]  

    for ripple in ripples:
        ripple_start,ripple_end = ripple
        if ripple_start <= window_samples[1] and ripple_end >= window_samples[0]:
            print(f"Ripple start: {round(ripple_start/liset.fs,4)} s, end: {round(ripple_end/liset.fs,4)}")
            ripples_in_window.append(ripple)
                
    if filter:
        for channel in range(data.shape[1]):
            filtered[:, channel] = bandpass_filter(data[:, channel], bandpass=(100,250),fs=liset.fs,)
    else:
        filtered = data
    downsampled = filtered[::downsample, :]
       
    
    # Create time axis for plotting
    start_sec = start  # Starting second of the window
    time_axis = start_sec + (np.arange(downsampled.shape[0]) / downsampled_fs)

    plt.figure(figsize=(12, 6))
    for ch in range(downsampled.shape[1]):
        plt.plot(time_axis, downsampled[:, ch] + ch * offset, label=f'Ch {ch}')  # Offset each channel
    # Plot ripple regions
    for start_sample, end_sample in ripples_in_window:
        ripple_start_sec = start_sample / liset.fs
        ripple_end_sec = end_sample / liset.fs
        plt.axvspan(ripple_start_sec, ripple_end_sec, color='yellow', alpha=0.2)

    plt.title(f'LISet Data ({downsampled_fs} Hz, Bandpass 100–250 Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (offset)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

plot_liset(liset, downsampled_fs=1000, window=[0,1000], filter=True, offset=2) 