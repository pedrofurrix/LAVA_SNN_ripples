import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# liset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../liset_tk'))
# sys.path.insert(0, liset_path)


# Based on https://github.com/kburel/snn-hfo-detection/blob/main/snn_hfo_detection/functions/signal_to_spike/utility.py#L43
def calculate_threshold(signal,downsampled_fs,window_size,sample_ratio,scaling_factor,plot=False,verbose=False):
    times=np.arange(0, len(signal)) / downsampled_fs  # Time in seconds 
    min_time = np.min(times)
   
    if np.min(times) < 0:
        raise ValueError(
            f'Tried to find thresholds for a dataset with a negative time: {min_time}')
    duration = np.max(times) - min_time
    if verbose:
        print(f"Duration of the signal: {duration} seconds, between {np.min(times)} and {np.max(times)}")
        
    if duration <= 0:
        raise ValueError(
            f'Tried to find thresholds for a dataset with a duration that under or equal to zero. Got duration: {duration}')

    if len(signal) == 0:
        raise ValueError('signals is not allowed to be empty, but was'
                         )
    if len(times) == 0:
        raise ValueError('times is not allowed to be empty, but was')

    if len(signal) != len(times):
        raise ValueError(
            f'signals and times need to have corresponding indices, but signals has length {len(signal)} while times has length {len(times)}')

    if not 0 < sample_ratio < 1:
        raise ValueError(
            f'sample_ratio must be a value between 0 and 1, but was {sample_ratio}'
        )

    num_timesteps = int(np.ceil(duration / window_size))
    if verbose:
        print(f"Number of time steps: {num_timesteps} for window size {window_size} seconds")
    max_min_amplitude = np.zeros((num_timesteps, 2))
    variation = np.zeros((num_timesteps))
    for interval_nr, interval_start in enumerate(np.arange(start=0, stop=duration, step=window_size)):
        interval_end = interval_start + window_size
        index = np.where((times >= interval_start) & (times <= interval_end))
        max_amplitude = np.max(signal[index])
        min_amplitude = np.min(signal[index])
        max_min_amplitude[interval_nr, 0] = max_amplitude
        max_min_amplitude[interval_nr, 1] = min_amplitude
        variation[interval_nr] = abs(max_amplitude - min_amplitude)

    chosen_samples = max(int(np.round(num_timesteps * sample_ratio)), 1)
    if verbose:
        print(f"Chosen samples for threshold calculation: {chosen_samples}")
    # threshold_up = np.mean(np.sort(max_min_amplitude[:, 0])[:chosen_samples])
    # threshold_dn = np.mean(
    #     np.sort(max_min_amplitude[:, 1] * -1)[:chosen_samples])
    # if verbose:
    #     print(f"Threshold up: {threshold_up}, Threshold down: {threshold_dn}")
    #     print(f"Final Threshold: {scaling_factor * (threshold_up + threshold_dn)}")
    # if plot:
    #     plot_threshold_hist(max_min_amplitude[:,0],max_min_amplitude[:,1],threshold_up*scaling_factor,bins=20)
    # threshold= scaling_factor*(threshold_up + threshold_dn)
    threshold = np.mean(np.sort(variation[:])[:chosen_samples])*scaling_factor
    return threshold

def threshold_percentile(signal,downsampled_fs,window_size,percentile,scaling_factor,plot=False,verbose=False):
    times=np.arange(0, len(signal)) / downsampled_fs  # Time in seconds # This will be for the original data...

    min_time = np.min(times)
    if np.min(times) < 0:
        raise ValueError(
            f'Tried to find thresholds for a dataset with a negative time: {min_time}')
    duration = np.max(times) - min_time
    if verbose:
        print(f"Duration of the signal: {duration} seconds, between {np.min(times)} and {np.max(times)}")
    if duration <= 0:
        raise ValueError(
            f'Tried to find thresholds for a dataset with a duration that under or equal to zero. Got duration: {duration}')

    if len(signal) == 0:
        raise ValueError('signals is not allowed to be empty, but was'
                         )
    if len(times) == 0:
        raise ValueError('times is not allowed to be empty, but was')

    if len(signal) != len(times):
        raise ValueError(
            f'signals and times need to have corresponding indices, but signals has length {len(signal)} while times has length {len(times)}')

    num_timesteps = int(np.ceil(duration / window_size))
    max_amplitudes = np.zeros((num_timesteps))
    min_amplitudes = np.zeros((num_timesteps))

    if verbose:
        print(f"Number of time steps: {num_timesteps} for window size {window_size} seconds")

    for interval_nr, interval_start in enumerate(np.arange(start=0, stop=duration, step=window_size)):
        interval_end = interval_start + window_size
        index = np.where((times >= interval_start) & (times <= interval_end))
        max_amplitude = np.max(signal[index])
        min_amplitude = np.min(signal[index])
        max_amplitudes[interval_nr] = max_amplitude
        min_amplitudes[interval_nr] = min_amplitude

    threshold_percentile = np.percentile(max_amplitudes, percentile)
    threshold=threshold_percentile*scaling_factor
    if verbose:
        print(f"Threshold for percentile {percentile}:  {threshold_percentile}")
        print(f"Final Threshold: {threshold}")
    if plot:
        plot_threshold_hist(max_amplitudes,min_amplitudes,threshold,bins=20)
    return threshold

def plot_threshold_hist(max_amplitudes,min_amplitudes,threshold,bins=20):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot max amplitudes
    axs[0].hist(max_amplitudes, bins=bins, alpha=0.7, color='blue', label='Max Amplitudes')
    axs[0].axvline(threshold, color='r', linestyle='dashed', linewidth=1, label='Threshold')
    axs[0].set_title('Histogram of Max Amplitudes')
    axs[0].set_xlabel('Amplitude')
    axs[0].set_ylabel('Count')
    axs[0].legend()

    # Plot min amplitudes
    axs[1].hist(min_amplitudes, bins=bins, alpha=0.7, color='green', label='Min Amplitudes')
    # axs[1].axvline(threshold, color='r', linestyle='dashed', linewidth=1, label='Threshold')
    axs[1].set_title('Histogram of Min Amplitudes')
    axs[1].set_xlabel('Amplitude')
    axs[1].set_ylabel('Count')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def up_down_channel(signal,threshold,downsampled_fs,refractory=0,initial_value=None,return_value=False):
    # Define parameters
    # print("Threshold=",threshold)
    num_timesteps = len(signal)
    spikified = np.zeros((num_timesteps, 2 ))
    if initial_value is not None:	
        value = initial_value
    else:
        value=signal[0]
    refractory_samples = int(refractory*downsampled_fs)
    
    if refractory_samples == 0:
        refractory_samples = 1

    i = 0
    # print("Max Signal:", max(signal),"\n Min Signal:",min(signal))
    while i < num_timesteps:
        delta = signal[i] - value
        if delta >= threshold:
            spikified[i,0] = 1
            value = signal[i]
            i += refractory_samples  # skip refractory period
            # print(delta)
        elif delta <= -threshold:
            spikified[i,1] = 1
            value = signal[i]
            i += refractory_samples  # skip refractory period    
            # print(delta)
        else:
            i += 1  # no spike, move to next time step
    if return_value:
        return spikified, value
    else:
        return spikified

def up_down_channel_SF(signal,threshold,downsampled_fs,refractory=0):
    # Define parameters
    # print("Threshold=",threshold)
    num_timesteps = len(signal)
    spikified = np.zeros((num_timesteps, 2 ))
    value=signal[0]
    refractory_samples = int(refractory*downsampled_fs)
    
    if refractory_samples == 0:
        refractory_samples = 1

    i = 0
    # print("Max Signal:", max(signal),"\n Min Signal:",min(signal))
    while i < num_timesteps:
        delta = signal[i] - value
        if delta >= threshold:
            spikified[i,0] = 1
            value +=threshold
            i += refractory_samples  # skip refractory period
            # print(delta)
        elif delta <= -threshold:
            spikified[i,1] = 1
            value -=threshold
            i += refractory_samples  # skip refractory period    
            # print(delta)
        else:
            i += 1  # no spike, move to next time step

    return spikified



def calculate_snr(original, reconstructed):
    """
    Calculate the Signal-to-Noise Ratio (SNR) between the original and reconstructed signals.
    The SNR is calculated as the ratio of the power of the original signal to the power of the noise (difference between original and reconstructed signals).
    """
    # Ensure inputs are numpy arrays
    s = np.asarray(original)
    r = np.asarray(reconstructed)

    # Compute the power of the original signal
    power_signal = np.mean(s ** 2)

    # Compute the power of the noise (difference)
    power_noise = np.mean((s - r) ** 2)

    # Avoid division by zero
    if power_noise == 0:
        return float('inf')  # Perfect reconstruction

    # Compute SNR in dB
    snr_db = 20 * np.log10(power_signal / power_noise)
    return snr_db


def calculate_rmse(original, reconstructed):
    error=np.sqrt(np.mean((reconstructed-original) ** 2))
    return error

def calculate_r_squared(original, reconstructed):  
    s = np.asarray(original)
    r = np.asarray(reconstructed)

    ss_res = np.sum((s - r) ** 2)
    ss_tot = np.sum((s - np.mean(s)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else -np.inf  # Edge case: constant signal

    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def calculate_average_spike_rate(spike_train,downsampled_fs=0):  
    sp = np.asarray(spike_train)
    afr = np.sum(np.abs(sp)) / len(sp)
    # Convert to Hz
    afr = afr * downsampled_fs

    return afr


def extract_spikes_downsample(spike_train,factor,verbose=False):
    """
    Downsamples a binary (UP/DOWN) spike train from original_freq to target_freq
    keeping at most 1 spike per ms. Chooses the direction with more spikes in each window.
    
    Parameters:
        spike_train (np.ndarray): Shape (n_samples, 2), binary values for UP and DOWN
        original_freq (int): Original sampling rate (default: 30000 Hz)
        target_freq (int): Target sampling rate (default: 1000 Hz)
    
    Returns:
        np.ndarray: Downsampled spike train (shape: n_bins, 2)
    """

    n_samples = spike_train.shape[0]
    n_bins = n_samples // factor
    
    # Trim excess samples if needed
    trimmed = spike_train[:n_bins * factor]
    # Total spikes before downsampling
    total_spikes_before = np.sum(trimmed)
    # Reshape to (n_bins, factor, 2)
    # Each row corresponds to 1 ms window with 30 time points of 2D spikes (UP/DOWN)
    reshaped = trimmed.reshape(n_bins, factor, 2)
    
    # Sum UP and DOWN spikes within each bin
    up_sum = reshaped[:, :, 0].sum(axis=1)
    down_sum = reshaped[:, :, 1].sum(axis=1)
    
    # Allocate result array
    result = np.zeros((n_bins, 2), dtype=int)
    
    # Assign dominant spike direction
    result[up_sum > down_sum, 0] = 1  # UP spike
    result[down_sum > up_sum, 1] = 1  # DOWN spike
    # If equal or both zero → remains [0, 0]
    # Total spikes after downsampling
    total_spikes_after = np.sum(result)
    # Print lost spike count
    spikes_lost = total_spikes_before - total_spikes_after
    if verbose:
        print(f"Total spikes before: {total_spikes_before}")
        print(f"Total spikes after: {total_spikes_after}")
        print(f"Spikes lost during downsampling: {spikes_lost}")

    return result,spikes_lost


def average_downsampling(signal,factor):
    """
    Downsamples a signal from original_freq to target_freq using averaging.
    
    Parameters:
        signal (np.ndarray): Input signal to be downsampled
        original_freq (int): Original sampling rate (default: 30000 Hz)
        target_freq (int): Target sampling rate (default: 1000 Hz)
    
    Returns:
        np.ndarray: Downsampled signal
    """
    
    n_samples = len(signal)
    n_bins = n_samples // factor
    
    # Trim excess samples if needed
    trimmed = signal[:n_bins * factor]
    
    # Reshape to (n_bins, factor)
    reshaped = trimmed.reshape(n_bins, factor)
    
    # Average within each bin
    downsampled_signal = reshaped.mean(axis=1)
    
    return downsampled_signal

def decimation_downsampling(signal,factor):
    """
    Downsamples a signal from original_freq to target_freq using decimation.
    
    Parameters:
        signal (np.ndarray): Input signal to be downsampled
        original_freq (int): Original sampling rate (default: 30000 Hz)
        target_freq (int): Target sampling rate (default: 1000 Hz)
    
    Returns:
        np.ndarray: Downsampled signal
    """
    trim_len = len(signal) - (len(signal) % factor)
    signal = signal[:trim_len]
    downsampled_signal = signal[::factor]
    
    return downsampled_signal

from scipy.spatial import distance
from scipy.stats import entropy, chisquare
from pyemd import emd

def evaluate_histogram(hist1,hist2,config):
    
    h1 = hist1[0].astype(np.float64)
    h2 = hist2[0].astype(np.float64)

    epsilon = 1e-8
    h1 += epsilon
    h2 += epsilon
    h1 /= np.sum(h1)
    h2 /= np.sum(h2)

    ### 1. Bhattacharyya Distance
    bc = np.sum(np.sqrt(h1 * h2))  # Bhattacharyya coefficient
    bhatta_dist = -np.log(bc + 1e-10)  # Avoid log(0)
    print(f"Bhattacharyya Distance: {bhatta_dist:.4f}")
    config['bhattacharyya'] = bhatta_dist

    ### 2. KL Divergence
    kl_div = entropy(h1, h2)
    print(f"KL Divergence: {kl_div:.4f}")
    config['kl_divergence'] = kl_div

    ### 3. Earth Mover’s Distance (EMD)
    # Distance matrix: 1D histogram → distances between bin positions
    bin_ids = np.arange(len(h1)).astype(np.float64)
    distance_matrix = np.abs(np.subtract.outer(bin_ids, bin_ids))
    emd_value = emd(h1, h2, distance_matrix)
    print(f"Earth Mover’s Distance: {emd_value:.4f}")
    config['emd'] = emd_value

    ### 4. Chi-Square Test
    # Assumes hist1 is observed, hist2 is expected
    chi_stat, chi_p = chisquare(h1* np.sum(h1), h2* np.sum(h2))
    print(f"Chi-Square Statistic: {chi_stat:.4f}, p-value: {chi_p:.4f}")
    config['chi_square_stat'] = chi_stat
    config['chi_square_p'] = chi_p



    return config

def bandpass_filter(signal, bandpass, fs, order=4):
    """
    Filters the signal with butterworth bandpass filter.
    """
    # Definir la frecuencia de corte del filtro pasa-bandas
    low_cutoff = bandpass[0]  # Frecuencia de corte inferior en Hz
    high_cutoff = bandpass[1] # Frecuencia de corte superior en Hz

    # Calcular las frecuencias de corte normalizadas
    nyquist_freq = 0.5 * fs  # Frecuencia de Nyquist para una señal con frecuencia de muestreo de 1000 Hz
    low = low_cutoff / nyquist_freq
    high = high_cutoff / nyquist_freq

    # Diseñar el filtro pasa-bandas de Butterworth
    b, a = butter(order, [low, high], btype='band')

    # Aplicar el filtro a la señal
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal
