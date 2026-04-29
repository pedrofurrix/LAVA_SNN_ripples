import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, wilcoxon, mannwhitneyu
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
    for interval_nr, interval_start in enumerate(np.arange(start=0, stop=duration, step=window_size)):
        interval_end = interval_start + window_size
        index = np.where((times >= interval_start) & (times <= interval_end))
        max_amplitude = np.max(signal[index])
        min_amplitude = np.min(signal[index])
        max_min_amplitude[interval_nr, 0] = max_amplitude
        max_min_amplitude[interval_nr, 1] = min_amplitude
   
    variation=np.abs(max_min_amplitude[:,0]-max_min_amplitude[:,1])
    sorted_variation = np.sort(variation)
    chosen = max(1, int(len(sorted_variation) * sample_ratio))
    threshold=scaling_factor*np.mean(sorted_variation[:chosen])
        
    return threshold

def cliffs_delta(u_stat, n_A, n_B):
     # Compute_cliff_delta:
    U1 = u_stat
    U2 = n_A * n_B - U1

    # Cliff’s delta must use U1 (vals_A vs vals_B)
    cliffs_d = (U1 - U2) / (n_A * n_B)
    # Interpretation (Romano et al., 2006)
    abs_d = abs(cliffs_d)
    if abs_d < 0.147:
        effect = "Negligible"
    elif abs_d < 0.33:
        effect = "Small"
    elif abs_d < 0.474:
        effect = "Medium"
    else:
        effect = "Large"
        
    # print(f"Cliff's Delta: {cliffs_d:.3f} ({effect})")

    return (cliffs_d,effect)

def barplot_annotate_brackets(num1, num2, data, center, height, ax, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values. 
    Adapted from the provided script.
    """
    if type(data) is str:
        text = data
    else:
        if data > 0.05:
          return
        text = ''
        p = .05
        while data < p:
            text += '*'
            p /= 10.
            if maxasterix and len(text) == maxasterix:
                break
        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)

def effect_size_r(u_stat, n_A, n_B):
    # A metric to use with Mann-Whitney U test is the effect size r:
    # Compute_effect_size_r:
    z=(u_stat-n_A*n_B/2)/np.sqrt(n_A * n_B * (n_A + n_B + 1) / 12)
    r=z/np.sqrt(n_A + n_B)

    # Interpretation (Cohen, 1988)
    if abs(r) < 0.1:
        effect = "Negligible"
    elif abs(r) < 0.3:
        effect = "Small"
    elif abs(r) < 0.5:
        effect = "Medium"
    else:
        effect = "Large"
    # print(f"Effect Size r: {r:.3f} ({effect})")

    return (r,effect)

def vargha_delaney(u_stat, n_A, n_B):
    # A non-parametric effect size measure for the Mann-Whitney U test is the Vargha-Delaney A measure:
    # Compute_Vargha_Delaney_A:
    A = u_stat / (n_A * n_B)

    # Interpretation (Vargha & Delaney, 2000)
    if 0.44 < A < 0.56:
        effect = "Negligible"
    elif 0.56 <= A < 0.64 or 0.36 < A <= 0.44:
        effect = "Small"
    elif 0.64 <= A < 0.71 or 0.29 < A <= 0.36:
        effect = "Medium"
    else:
        effect = "Large"
    # print(f"Vargha-Delaney A: {A:.3f} ({effect})")

    return (A,effect)

def effect_size_r_wilcoxon(x, y):
    
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    N = len(diff)

    stat, p = wilcoxon(x, y)

    mu = N * (N + 1) / 4
    sigma = np.sqrt(N * (N + 1) * (2*N + 1) / 24)

    z = (stat - mu) / sigma
    r = z / np.sqrt(N)

    if abs(r) < 0.1:
        effect = "Negligible"
    elif abs(r) < 0.3:
        effect = "Small"
    elif abs(r) < 0.5:
        effect = "Medium"
    else:
        effect = "Large"

    return r, effect

def probability_of_superiority_wilcoxon(x, y):
    x, y = np.array(x), np.array(y)
    diff = x - y
    # Remove ties as per Wilcoxon standard (Pratt method is an alternative)
    diff = diff[diff != 0]
    ties=diff[diff == 0]
    N = len(diff)
    
    if N == 0:
        return 0.5, "Negligible"

    # Get the sum of positive ranks specifically
    ranks = rankdata(np.abs(diff))
    w_plus = np.sum(ranks[diff > 0])
    
    # Total possible rank sum
    w_total = N * (N + 1) / 2
    
    # Probability of Superiority (A)
    # This represents P(X > Y)
    A = (w_plus+ 0.5 * len(ties)) / w_total
    
    # Classification based on Vargha and Delaney (2000)
    # We use the distance from 0.5 to determine magnitude
    val = abs(A - 0.5)
    
    if val < 0.06: # 0.44 to 0.56
        effect = "Negligible"
    elif val < 0.14: # 0.36 to 0.64
        effect = "Small"
    elif val < 0.21: # 0.29 to 0.71
        effect = "Medium"
    else:
        effect = "Large"

    return A, effect


def calculate_matched_pairs_effect_size_wilcoxon(x, y):
    x, y = np.array(x), np.array(y)
    diff = x - y
    diff = diff[diff != 0] # Remove ties
    N = len(diff)
    
    if N == 0:
        return 0, "Negligible"

    ranks = rankdata(np.abs(diff))
    R_plus = np.sum(ranks[diff > 0])
    R_minus = np.sum(ranks[diff < 0])
    total_rank_sum = R_plus + R_minus
    
    # 1. Matched-pairs rank biserial correlation (rc
    # If all values in x are greater than y, R_plus = total_rank_sum and R_minus = 0, giving rc = 1 (perfect positive association).
    rc = (R_plus - R_minus) / total_rank_sum
    

    
    # 2. Interpretation (using rc absolute thresholds from your text)
    abs_rc = abs(rc)
    if abs_rc < 0.11:
        effect = "Negligible"
    elif abs_rc < 0.28:
        effect = "Small"
    elif abs_rc < 0.43:
        effect = "Medium"
    else:
        effect = "Large"
        
    return rc, effect

def holm_bonferroni(p_values, alpha=0.05):
    p_values = np.asarray(p_values)
    m = len(p_values)

    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    adjusted = (m - np.arange(m)) * sorted_pvals
    adjusted = np.maximum.accumulate(adjusted)   # enforce monotonicity
    adjusted = np.clip(adjusted, 0, 1)

    # reorder to original order
    adjusted_pvals = np.empty_like(adjusted)
    adjusted_pvals[sorted_indices] = adjusted

    significant = adjusted_pvals <= alpha
    return significant, adjusted_pvals

def forceAspect(ax, aspect=1):
    images = ax.get_images()
    if images:
        extent = images[0].get_extent()
        ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if (x1 - x0) != 0 and (y1 - y0) != 0:
        ax.set_aspect(abs((x1 - x0) / (y1 - y0)) / aspect)
