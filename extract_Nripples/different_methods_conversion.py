
"""
This script converts the dataset obtained from extract_ripples_for_training into a neuromorphic format -spikes
It uses the functions defined in the liset_tk library to process the data.
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

liset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../liset_tk'))
sys.path.insert(0, liset_path)
from signal_aid import y_discretize_1Dsignal, cutoff_amplitude
import os
from liset_tk import liset_tk

# bandpass=[100,250]
# Load the saved dataset
data_dir = os.path.join(os.path.dirname(__file__),"train_pedro","dataset")
# true_positives = np.load(os.path.join(data_dir, f'true_positives_{bandpass[0]}_{bandpass[1]}Hz.npy'))
# true_negatives = np.load(os.path.join(data_dir, f'true_negatives_{bandpass[0]}_{bandpass[1]}Hz.npy'))
# downsampled_fs=4000


def level_crossing(signal,threshold=0.3,downsampled_fs=4000,bandpass=[10,250],positives=True,timestep=1,save=True):
    """
    Function to encode Spikes when the variation of the signal is above a certain threshold.
    This function is used to discretize the signal into spikes and non-spikes.
    Two Input Channels - for positive and negative spikes.
    The function also saves the resulting arrays in a specified directory.
    """
    # Define parameters
    num_timesteps = signal.shape[1]
    num_samples = signal.shape[0] 
    spikified = np.zeros((num_samples, 2, num_timesteps))
    
    timestep_high= int(timestep/bandpass[0]*downsampled_fs) # Lowest Frequency of the Bandpass Filter
    timestep_low= int(timestep/bandpass[1]*downsampled_fs) # Highest Frequency of the Bandpass Filter

    for idx in range(num_samples):
        for i in range(timestep_high, num_timesteps):  # start at 'timestep' to avoid negative index
            
            # Calculate the delta between the current and previous values
            delta_long = signal[idx, i] - signal[idx, i - timestep_high]
            delta_short= signal[idx, i] - signal[idx, i - timestep_low]
            
            # Check if the delta is above the threshold
            if delta_long >= threshold or delta_short >= threshold:
                spikified[idx, 0, i] = 1
            elif delta_long <= -threshold or delta_short <= -threshold:
                spikified[idx, 1, i] = 1

    
    if save:
        # Save the arrays
        save_dir=data_dir = os.path.join("train_pedro","n_dataset","level_crossing")
        os.makedirs(save_dir, exist_ok = True)
        if positives:
            np.save(os.path.join(save_dir,f"ntrue_positives_{bandpass[0]}_{bandpass[1]}Hz.npy"), arr=spikified, allow_pickle=True)
        else:
            np.save(os.path.join(save_dir,f"ntrue_negatives_{bandpass[0]}_{bandpass[1]}Hz.npy"), arr=spikified, allow_pickle=True)

    return spikified

def up_down_spikes(signal,downsampled_fs=4000,bandpass=[10,250],positives=True,timestep=1,refractory=False,threshold=0.3,save=True):
    """
    Considering an average of 0 for the signal and std=1 - based on the z-score normalization
    The function detects spikes in the signal based on the specified threshold.
    Acts as a Delta Modulator, where the signal is represented by two channels: one for positive spikes and another for negative spikes.
    The function also saves the resulting arrays in a specified directory.
    """

    # Define parameters
    num_timesteps = signal.shape[1]
    num_samples = signal.shape[0] 
    spikified = np.zeros((num_samples, 2, num_timesteps))
    value=0

    if refractory:
        refractory_samples = int(refractory*downsampled_fs)
    else:
        refractory_samples = 0

    for idx in range(num_samples):
        i = 0
        while i <num_timesteps:
            delta = signal[idx, i] - value
            if delta >= threshold:
                spikified[idx, 0, i] = 1
                value = signal[idx, i]
                i += refractory_samples  # skip refractory period
            elif delta <= -threshold:
                spikified[idx, 1, i] = 1
                value = signal[idx, i]
                i += refractory_samples  # skip refractory period
            else:
                i += 1  # no spike, move to next time step


    if save:
        # Save the arrays
        save_dir=data_dir = os.path.join("train_pedro","n_dataset","up_down")
        
        os.makedirs(save_dir, exist_ok = True)
        if positives:
            np.save(os.path.join(save_dir,f"ntrue_positives_{bandpass[0]}_{bandpass[1]}Hz.npy"), arr=spikified, allow_pickle=True)
        else:
            np.save(os.path.join(save_dir,f"ntrue_negatives_{bandpass[0]}_{bandpass[1]}Hz.npy"), arr=spikified, allow_pickle=True)

    return spikified


def up_down_and_marcos(signal,downsampled_fs=4000,bandpass=[10,250],positives=True,timestep=1,refractory=False,threshold=0.3,y_size=10,save=True):
    """
    Considering an average of 0 for the signal and std=1 - based on the z-score normalization
    The function detects spikes in the signal based on the specified threshold.
    Detects the changes bigger than the threshold and then discretizes the signal into y_num_samples - this can also be based on the 
    The function also saves the resulting arrays in a specified directory.
    Refractory period is also considered - in seconds.

    """

    # Define parameters
    num_timesteps = signal.shape[1]
    num_samples = signal.shape[0] 
    spikified = np.zeros((num_samples, y_size, num_timesteps))

    cutoff = cutoff_amplitude(signal)
   
    value=0

    if refractory:
        refractory_samples = int(refractory*downsampled_fs)
    else:
        refractory_samples = 0

    for idx in range(num_samples):
        i = 0
        while i < num_timesteps:
            delta = signal[idx, i] - value
            if delta >= threshold or delta<= -threshold:
                if np.abs(signal[idx,i]) < cutoff:  
                        y_val = (y_size - 1) - int(signal[idx,i]/cutoff * y_size/2 + y_size/2)
                        spikified[idx,y_val,i] = 1
                else:
                    if signal[idx,i] < 0:
                        y_val = y_size - 1
                    else:
                        y_val = 0
                    spikified[idx,y_val,i] = 1
                i += refractory_samples  # skip refractory period
                value = signal[idx, i]
            i+=1  # no spike, move to next time step
    

    if save:
        # Save the arrays
        save_dir=data_dir = os.path.join("train_pedro","n_dataset","up_down_discretized")
        
        os.makedirs(save_dir, exist_ok = True)
        if positives:
            np.save(os.path.join(save_dir,f"ntrue_positives_{bandpass[0]}_{bandpass[1]}Hz.npy"), arr=spikified, allow_pickle=True)
        else:
            np.save(os.path.join(save_dir,f"ntrue_negatives_{bandpass[0]}_{bandpass[1]}Hz.npy"), arr=spikified, allow_pickle=True)

    return spikified

# Define parameters
def marcos_ydiscretization(signal,downsampled_fs=4000,bandpass=[0,250],y_size=20,save=False,positives=True):
    """
    Function to discretize a signal into y_size bins.
    """

    samples_len = signal.shape[1]
    y_num_samples = y_size
    cutoff = cutoff_amplitude(signal)

    num_samples = signal.shape[0] #1794 # length of the true positives # acho que devia ser true_positives.shape[0]

    n_true_values = np.zeros((num_samples, y_num_samples, samples_len))
    # n_true_negatives = np.zeros((num_samples, y_num_samples, samples_len))

    for idx in range(signal.shape[0]):
        n_true_values[idx, :, :] = y_discretize_1Dsignal(signal[idx], y_num_samples, cutoff)
        # n_true_negatives[idx, :, :] = y_discretize_1Dsignal(true_negatives[idx], y_num_samples, cutoff)  

    # Save the arrays
    if save:
        save_dir=data_dir = os.path.join("train_pedro","n_dataset","marcos",f"{y_size}")
        os.makedirs(save_dir, exist_ok = True)
        if positives:
            np.save(os.path.join(save_dir,f"ntrue_positives_{bandpass[0]}_{bandpass[1]}Hz.npy"), arr=n_true_values, allow_pickle=True)
        else:
            np.save(os.path.join(save_dir,f"ntrue_negatives_{bandpass[0]}_{bandpass[1]}Hz.npy"), arr=n_true_values, allow_pickle=True)
    return n_true_values

def up_down_and_marcos_variation(signal,downsampled_fs=4000,bandpass=[10,250],positives=True,timestep=1,refractory=False,threshold=0.3,y_size=10,save=True):
    """
    Considering an average of 0 for the signal and std=1 - based on the z-score normalization
    The function detects spikes in the signal based on the specified threshold.
    Detects the changes bigger than the threshold and then discretizes the signal into y_num_samples - this can also be based on the 
    The function also saves the resulting arrays in a specified directory.
    Refractory period is also considered - in seconds.

    """

    # Define parameters
    num_timesteps = signal.shape[1]
    num_samples = signal.shape[0] 
    spikified = np.zeros((num_samples, y_size, num_timesteps))

    cutoff = 2 #1 is the std so this is 2*std
   
    value=0

    if refractory:
        refractory_samples = int(refractory*downsampled_fs)
    else:
        refractory_samples = 0

    for idx in range(num_samples):
        i = 0
        while i < num_timesteps:
            delta = signal[idx, i] - value
            if delta >= threshold or delta<= -threshold:
                if np.abs(delta) < cutoff:  
                        y_val = (y_size - 1) - int(delta/cutoff * y_size/2 + y_size/2)
                        spikified[idx,y_val,i] = 1
                else:
                    if delta < 0:
                        y_val = y_size - 1
                    else:
                        y_val = 0
                    spikified[idx,y_val,i] = 1
                i += refractory_samples  # skip refractory period
                value = signal[idx, i]
            i+=1  # no spike, move to next time step

    if save:
        # Save the arrays
        save_dir=data_dir = os.path.join("train_pedro","n_dataset","up_down_discretized_variation")
        
        os.makedirs(save_dir, exist_ok = True)
        if positives:
            np.save(os.path.join(save_dir,f"ntrue_positives_{bandpass[0]}_{bandpass[1]}Hz.npy"), arr=spikified, allow_pickle=True)
        else:
            np.save(os.path.join(save_dir,f"ntrue_negatives_{bandpass[0]}_{bandpass[1]}Hz.npy"), arr=spikified, allow_pickle=True)

    return spikified


def plot_spikes_levelcrossing(signal,original,sample_num=7,fs=4000,diff_plots=False):
    original=original[sample_num,:]
    signal=signal[sample_num,:,:] 
    time = np.arange(len(original)) / fs  # Adjust fs if needed
    if diff_plots:
        # Plot
        plt.subplots(2,1,figsize=(12, 8),sharex=True)
        plt.subplot(211)
        plt.plot(time, original, label='Original Signal', color='black')

        # Overlay positive spikes
        plt.subplot(212)
        plt.vlines(time[signal[0] == 1], 0.5,1.5, alpha=0.5,
                color='red', label='Positive Spikes' ,lw=0.5)
        # Overlay negative spikes
        plt.vlines(time[signal[1] == 1], -0.5,0.5, alpha=0.5,
                color='blue', label='Negative Spikes',lw=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude") 
        plt.show()
    else:
        plt.figure(figsize=(12, 4))
        plt.plot(time, original, label='Original Signal', color='black')
        peak=np.max(original)
        trough=np.min(original)
        mean=np.mean(original)
        plt.vlines(time[signal[0] == 1],mean,peak, alpha=0.5,
                color='red', label='Positive Spikes' ,lw=0.5)
        plt.vlines(time[signal[1] == 1], trough,mean, alpha=0.5,
                color='blue', label='Negative Spikes',lw=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude") 
        plt.show()


def plot_marcos(signal,original,sample_num=7,fs=4000,diff_plots=True):
    original=original[sample_num,:]
    signal=signal[sample_num,:,:]
    time = np.arange(len(original)) / fs  # Adjust fs if needed
    # Get the coordinates of all 1s in the array
    y_coords, x_coords = np.where(signal == 1)
    if diff_plots:
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        axs[0].plot(time, original, label='Original Signal', color='steelblue')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title(f"Original Signal - Sample {sample_num}")
        axs[0].grid(True)

        axs[1].scatter(x_coords / fs, y_coords, s=5, color='black')
        axs[1].set_ylabel("Channel (Y)")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_title("Spike Events (Marcos)")
        axs[1].invert_yaxis()
        axs[1].grid(False)

        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(12, 5))
        plt.plot(time, original, label='Original Signal', color='steelblue')
        plt.scatter(x_coords / fs, y_coords, s=10, color='red', label='Spikes')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude / Channel")
        plt.title(f"Overlay: Original Signal & Spikes - Sample {sample_num}")
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.show()



