'''
----------- Duration of HFO events -----------
Important Constants regarding HFO events in the Synthetic Dataset
Taken from:https://github.com/monkin77/snn-torch/blob/master/src/snnt_utils/hfo.py 
'''
# HFO Detection Offsets [MIN_OFFSET, MAX_OFFSET, MEAN_OFFSET, TOLERANCE_OFFSET]
RIPPLE_DETECTION_OFFSET = [18, 45, 31, 20] # it's calculated as 4.5 periods of the ripple wavelet - for 100 Hz and 250 Hz as the limit frequencies
# FR_DETECTION_OFFSET = [9, 18, 13, 5]
# BOTH_DETECTION_OFFSET = [9, 57, 33, 24]


# HFO MAX Durations (Not the same as the Offset for SNN Detection)
RIPPLE_MAX_DUR = 120
FR_MAX_DUR = 40

# The Windows for HFO detection are based on the MAX DETECTION OFFSET
RIPPLE_CONFIDENCE_WINDOW = int(round(RIPPLE_DETECTION_OFFSET[1] * 1.8)) 
# FR_CONFIDENCE_WINDOW = int(round(FR_DETECTION_OFFSET[1] * 2.2))
# BOTH_CONFIDENCE_WINDOW = int(round(BOTH_DETECTION_OFFSET[1] * 2.2))

def window_plot(window_signal,window_spikes,gt,downsampled_fs=1000,detection_window=RIPPLE_DETECTION_OFFSET):
    import matplotlib.pyplot as plt
    import numpy as np
    from copy import deepcopy
    print(len(window_signal))
    # Create a time vector for the x-axis
    time_vector = np.linspace(0,len(window_signal),len(window_signal))/downsampled_fs # Assuming a sampling rate of 1000 Hz

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the signal
    ax.plot(time_vector, window_signal, label='Signal', color='blue')

    # Plot the spikes
    peak=max(np.max(window_signal[:]),0.1)
    trough=min(np.min(window_signal[:]),-0.1)
    mean=np.mean(window_signal[:])
    ax.vlines(time_vector[window_spikes[:,0] == 1],mean,peak, alpha=0.5,
            color='red', label='Positive Spikes' ,lw=0.5)
    ax.vlines(time_vector[window_spikes[:,1] == 1], trough,mean, alpha=0.5,
            color='blue', label='Negative Spikes',lw=0.5)

    # Plot the ground truth (gt)
    if gt>=0:
        gt_time=np.array(gt)/downsampled_fs  # Convert to seconds
        ax.vlines(gt_time,trough*1.2,peak*1.2, color='green',linestyle="--", label='Ground Truth',alpha=0.5)
        ax.fill_between([gt_time-(detection_window[2]/downsampled_fs), gt_time+(detection_window[2]/downsampled_fs)],trough*1.2, peak*1.2, color='green', alpha=0.1)

    # Add labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f"Signal with Spikes - {'Ripple' if gt >= 0 else 'No Ripple'}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    # Show the plot
    plt.show()