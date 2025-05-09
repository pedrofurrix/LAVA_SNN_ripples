from numpy import dtype

markers_dt = dtype([('label', '<U64'), ('position', '<f4'), ('duration', '<f4')])

# Define the indices for the up and down spikes in the INPUT / OUTPUT arrays
DN_SPIKE_IDX = 0
UP_SPIKE_IDX = 1

# Create enum for the different types of markers
class MarkerType:
    """Enum for the different types of markers"""
    RIPPLE = 1
    FAST_RIPPLE = 2
    BOTH = 3
    OTHER = 4

'''
----------- Duration of HFO events -----------
Important Constants regarding HFO events in the Synthetic Dataset
'''
# HFO Detection Offsets [MIN_OFFSET, MAX_OFFSET, MEAN_OFFSET, TOLERANCE_OFFSET]
RIPPLE_DETECTION_OFFSET = [18, 57, 37, 20]
FR_DETECTION_OFFSET = [9, 18, 13, 5]
BOTH_DETECTION_OFFSET = [9, 57, 33, 24]


# HFO MAX Durations (Not the same as the Offset for SNN Detection)
RIPPLE_MAX_DUR = 120
FR_MAX_DUR = 40
BOTH_MAX_DUR = int((RIPPLE_MAX_DUR + FR_MAX_DUR) / 2)  # Avg. of the 2 above ~ 80 ms
# Ground Truth Durations
MIN_GT_DISTANCE = 500  # 500 ms. From our analysis, the closest GT events are actually 1500ms apart. So 500ms is a safe margin.

# The Windows for HFO detection are based on the MAX DETECTION OFFSET
RIPPLE_CONFIDENCE_WINDOW = int(round(RIPPLE_DETECTION_OFFSET[1] * 1.8)) 
FR_CONFIDENCE_WINDOW = int(round(FR_DETECTION_OFFSET[1] * 2.2))
BOTH_CONFIDENCE_WINDOW = int(round(BOTH_DETECTION_OFFSET[1] * 2.2))

def band_to_gt_max_offset(band: MarkerType):
    """
    This function returns the Maximum time the SNN can take to detect the event.
    If the band is unknown, it returns 0.
    @band (MarkerType): The band to get the confidence window for.
    """
    if band == MarkerType.RIPPLE:
        return RIPPLE_DETECTION_OFFSET[1]
    elif band == MarkerType.FAST_RIPPLE:
        return FR_DETECTION_OFFSET[1]
    elif band == MarkerType.BOTH:
        return BOTH_DETECTION_OFFSET[1]
    else:
        raise ValueError("Unknown band type on band_to_confidence_window()")
    
def band_to_gt_offset(band: MarkerType):
    """
    This function returns the average event offset for the band.
    If the band is unknown, it returns 0.
    @band (MarkerType): The band to get the average duration for.
    """
    if band == MarkerType.RIPPLE:
        return RIPPLE_DETECTION_OFFSET[2]  # 37 ms
    elif band == MarkerType.FAST_RIPPLE:
        return FR_DETECTION_OFFSET[2]   # 13 ms
    elif band == MarkerType.BOTH:
        return BOTH_DETECTION_OFFSET[2]  # Avg. of the 2 above ~ 33 ms
    else:
        raise ValueError("Unknown band type on band_to_avg_duration()")
    
def band_to_gt_tolerance(band: MarkerType):
    """
    This function returns the tolerance for the band prediction.
    The raw GT events annotate the insertion of a relevant event. However,
    we are incrementing by the mean duration of the HFO type to approximate the
    end of the event. The SNN should spike at the end of the event (or in the following steps)
    
    This function indicates how much time before/after the approximated end of the event we consider
    as a valid prediction.
    """
    # TODO: Change these values?
    if band == MarkerType.RIPPLE:
        return RIPPLE_DETECTION_OFFSET[3]
    elif band == MarkerType.FAST_RIPPLE:
        return FR_DETECTION_OFFSET[3]
    elif band == MarkerType.BOTH:
        return BOTH_DETECTION_OFFSET[3]
    else:
        raise ValueError("Unknown band type on band_to_avg_duration()")
    
def band_to_sample_window_size(band: MarkerType):
    """
    This function returns the WINDOW_SIZE used when predicting events for a given band
    If the band is unknown, raises a ValueError
    @band (MarkerType): The band to get the confidence window for.
    """
    MAX_DETECTION_OFFSET = band_to_gt_max_offset(band)

    if band == MarkerType.RIPPLE:
        return int(round(MAX_DETECTION_OFFSET * 1.8))
    elif band == MarkerType.FAST_RIPPLE:
        return int(round(MAX_DETECTION_OFFSET * 2.5))
    elif band == MarkerType.BOTH:
        return int(round(MAX_DETECTION_OFFSET * 2.2))
    else:
        raise ValueError("Unknown band type on band_to_confidence_window()")

import numpy as np
import matplotlib.pyplot as plt

def input_sample_to_spike_raster(input_sample, gt_time, gt_tolerance, verbose=False):
    '''
    Creates a Matplotlib Event Plot containing a Spike Raster Plot of the UP and DN Spikes of a given input sample.
    ---------
    Parameters:
    - input_sample: The input sample to plot. Shape: (time_steps, num_features)
    - gt_time: The ground truth time of the event (HFO).
    - gt_tolerance: The confidence interval for the event [gt_time - gt_tolerance, gt_time + gt_tolerance].
    - verbose: If True, prints the spike times for the UP and DN neurons.
    ---------
    Returns:
    - fig: The Matplotlib Figure object containing the plot.
    - ax: The Matplotlib Axes object containing the plot.
    '''
    total_steps = input_sample.shape[0]

    # Find spike times for UP and DN neurons
    input_spike_times = [0, 0]
    input_spike_times[UP_SPIKE_IDX] = np.where(input_sample[:, UP_SPIKE_IDX] > 0)[0]
    input_spike_times[DN_SPIKE_IDX] = np.where(input_sample[:, DN_SPIKE_IDX] > 0)[0]

    # Print the spike times for the UP and DN neurons
    if verbose:
        if np.sum(gt_time) < 0:
            print(f"No HFO Event")
        else:
            print(f"GT Interval: {gt_time - gt_tolerance} to {gt_time + gt_tolerance}")

        print(f"input_spike_times: {input_spike_times}")

    # Creates a Figure and Axis objects
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="w")

    # Create the Event Plot
    ax.eventplot(
        input_spike_times,
        orientation="horizontal",  # Horizontal orientation
        linelengths=0.2,  # Length of the lines (height when horizontal orientation)
        colors=["blue", "red"],  # Color for each neuron
        lineoffsets=[0, 1],  # Offset for each spike train (y-axis offset)
    )

    # Add the Ground Truth time and tolerance lines
    if np.sum(gt_time) >= 0:
        # Add a Rectangle to indicate the GT Detection Window IF there is a relevant GT event (HFO)
        ax.axvline(gt_time, color="green", linestyle="--", label="GT Time", alpha=0.6)
        ax.axvspan(gt_time - gt_tolerance, gt_time + gt_tolerance, color="green", alpha=0.2, label="GT Tolerance")
        ax.legend(loc="upper right")

    # Customize the plot
    ax.set_xlabel("Steps or Time (ms)")
    ax.set_ylabel("UP/DN Spike Trains")
    ax.set_title("Input Spike Trains")
    ax.set_yticks([0, 1])
    ax.set_ylim([-0.5, 1.5])
    ax.set_xlim([0, total_steps])   # Set range of x-axis

    return fig, ax