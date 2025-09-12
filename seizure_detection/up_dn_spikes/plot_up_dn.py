import os
import numpy as np
import matplotlib.pyplot as plt

def plot_up_dn(id, shank, window=None, downsampled_fs=1000,channels=None):
    """
    Loads and plots filtered LFP data with overlaid up/down spikes, IISs,
    and seizure events for a given recording.

    Args:
        id (int): Index of the recording sub-directory to load.
        shank (int): The shank number to load data for.
        channels (list, optional): A list of channel indices to plot.
                                   If None, all channels are plotted. Defaults to None.
    """
    # --- 1. Data Loading ---
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(curr_dir, "up_dn_data")
    path = os.listdir(data_path)[id]
    data_path = os.path.join(data_path, path)
    
    print(f"Loading data from: {data_path}")

    # Load the data files. Assumes specific filenames.
    # NOTE: I've replaced 'spikified' with 'up_spikes' and 'down_spikes'
    #       to match the plotting goal.
    try:
        # up_spikes = np.load(os.path.join(data_path, f"up_spikes_{shank}.npy"))
        # down_spikes = np.load(os.path.join(data_path, f"down_spikes_{shank}.npy"))
        spikified=np.load(os.path.join(data_path, f"spikified_{shank}.npy"))
        filtered = np.load(os.path.join(data_path, f"filtered_{shank}.npy"))
        iiss = np.load(os.path.join(data_path, f"IISs_{shank}.npy"))
        seizures = np.load(os.path.join(data_path, f"seizures_{shank}.npy"))
    except FileNotFoundError as e:
        print(f"Error: A required data file was not found. {e}")
        return

    # --- 2. Setup Plotting ---
    
    # If no specific channels are requested, plot all of them
    if not channels:
        channels_to_plot = range(filtered.shape[1])
    else:
        channels_to_plot = channels
    if not window:
        window = (0, filtered.shape[0])
    else:
        window=(window[0]*downsampled_fs, window[1]*downsampled_fs)
    
    num_channels = len(channels_to_plot)
    time_vector = np.arange(filtered.shape[0])
    # Apply the window to the time vector
    time_vector = time_vector[window[0]:window[1]]/downsampled_fs

    # Create a figure with one subplot per channel
    # `sharex=True` links the time axes for easy comparison
    fig, axes = plt.subplots(num_channels, 1, figsize=(20, 4 * num_channels), sharex=True)
    
    # If there's only one channel, axes will not be an array, so we wrap it
    if num_channels == 1:
        axes = [axes]

    # --- 3. Loop Through Channels and Plot ---
    for i, channel_idx in enumerate(channels_to_plot):
        ax = axes[i]
        signal_slice=filtered[window[0]:window[1], channel_idx] 
        min_val, max_val = np.min(signal_slice), np.max(signal_slice)

        # Plot the filtered LFP signal
        ax.plot(time_vector, signal_slice, color='black', linewidth=0.7, label='Filtered LFP')
        
        # --- Plot Events for the current channel ---
        
        # Filter and plot UP spikes for this channel
        up_spikes_ch = np.where(spikified[:, channel_idx, 0]==1)
        up_spikes_slice=up_spikes_ch[0][(up_spikes_ch[0]>=window[0]) & (up_spikes_ch[0]<window[1])]/downsampled_fs
        ax.vlines(up_spikes_slice, ymin=max_val*0.6, ymax=max_val*0.8, color='red', linewidth=0.5, label="UP Spikes")
        
        # Filter and plot DOWN spikes for this channel
        down_spikes_ch = np.where(spikified[:, channel_idx, 1]==1)
        dn_spikes_slice=down_spikes_ch[0][(down_spikes_ch[0]>=window[0]) & (down_spikes_ch[0]<window[1])]/downsampled_fs
        ax.vlines(dn_spikes_slice, ymin=min_val*0.8, ymax=min_val*0.6, color='blue', linewidth=0.5, label="DN Spikes")

        # Plot IISs (as vertical lines across the whole subplot)
        # Find plot y-limits to draw the lines nicely

        # Find the IISs that fall within the current window
        iiss_window = iiss[(iiss >= window[0]) & (iiss < window[1])]/downsampled_fs
        ax.vlines(iiss_window, ymin=min_val, ymax=max_val, color='green', linestyle='--', alpha=0.6, label='IIS')

        seizure_starts = seizures[:, 0]
        seizure_ends = seizures[:, 1]
        mask = (seizure_ends >= window[0]) & (seizure_starts < window[1])
        seizures_in_window = seizures[mask]
        
        for j, (start, end) in enumerate(seizures_in_window):
            ax.fill_between([start / downsampled_fs, end / downsampled_fs], min_val, max_val, 
                            color='purple', alpha=0.2, 
                            label='Seizure' if j == 0 else "")

        # --- Formatting for each subplot ---
        ax.set_ylabel(f"Ch {channel_idx}")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axhline(0, color='gray', linewidth=0.5) # Add a zero line

    # --- 4. Final Figure Formatting ---
    
    # Create a single, clean legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    # Use a dictionary to remove duplicate labels (like 'Seizure')
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"UP/DN Spikes and Events - Shank {shank}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # Adjust layout for legend and title
    plt.show()

plot_up_dn(id=0, shank=2, window=None, channels=[3], downsampled_fs=1000)