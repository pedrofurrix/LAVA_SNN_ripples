import matplotlib.pyplot as plt
import numpy as np
import os
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,os.pardir))


def plot_livetest(prefix,parent_dir,downsampled_fs,window=None, title='Live Test Data', xlabel='Time', ylabel='Value',input=True):
    # Load the spike data, gt and original data from npy files
    data_dir=os.path.join(parent_dir,"extract_Nripples","train_pedro","dataset_up_down",str(downsampled_fs))
    spikes= np.load(os.path.join(data_dir, f'concat_spikes.npy'))
    gt = np.load(os.path.join(data_dir, f'concat_ripples.npy'))
    data=np.load(os.path.join(data_dir, f'concat_data.npy'))

    # Load output spikes
    outputspikes = np.load(os.path.join(os.path.dirname(__file__),"spikes", f'{prefix}_spikes_.npy'))
    
    # Load the parameters
    json_path = os.path.join(os.path.dirname(__file__),f'{prefix}_results.json')
    with open(json_path, 'r') as f:
        params = json.load(f)
    max_detection_offset=params["max_detection_offset"]/1000 # Convert to seconds
    refractory_period=params["refractory_period"] /1000*2 # Convert to seconds*2
    ripple_detection_offset=params["ripple_detection_offset"] # Convert to seconds

    # Create the figure and axis

    fig, ax = plt.subplots(figsize=(20, 6))
    
    if window is not None:
        start, end = window
        # Adjust the data, spikes, gt, and outputspikes to the specified window
        data = data[start:end]
        spikes = spikes[start:end]

        # Adjust ground truth events (gt): select events that overlap with the window
        # gt[:, 0] = start of ripple, gt[:, 1] = end of ripple
        # Keep ripples that overlap the window [start, end)
        gt = gt[(gt[:, 1] >= start) & (gt[:, 0] < end)]
        # Shift the ripple times to be relative to the window
        gt = gt
        
        # Adjust output spikes: keep those within [start, end)
        outputspikes = outputspikes[(outputspikes >= start) & (outputspikes < end)]



    # Convert to seconds
    up_spike_times = np.where(spikes[:, 0] == 1)[0]
    down_spike_times = np.where(spikes[:, 1] == 1)[0]
        # Use the same time base (in seconds)
    up_spike_times_sec = up_spike_times / 1000
    down_spike_times_sec = down_spike_times / 1000
    # Convert ground truth ripples to seconds
    gt_sec = gt / 1000
    # Convert output spikes to seconds
    outputspikes_sec = outputspikes / 1000


    time = np.arange(start, end) / 1000  # In seconds
    # Plot the original data
    ax.plot(time,data, label='Original Data', color='blue', alpha=0.5)
    
    # Plot the Input Up and Down Spikes
    if input:
        ax.vlines(up_spike_times_sec,0,3, color='green', alpha=0.5,label='Up Spikes')
        ax.vlines(down_spike_times_sec,-3,0, color='red',alpha=0.5, label='Down Spikes')
        ax.scatter(outputspikes_sec, np.ones_like(outputspikes_sec)*-4, color='purple', marker='o', label='Output Spikes')
    else:
        ax.scatter(outputspikes_sec, np.ones_like(outputspikes_sec)*-3, color='purple', marker='o', label='Output Spikes')

    # Plot the Ground Truth Ripples
    for i,ripple in enumerate(gt_sec):
        label = 'Ground Truth Ripple' if i == 0 else None  # Add label only to the first
        ax.fill_between([ripple[0], ripple[1]], -5,5, color='yellow', alpha=0.2, label=label)
    
    # Plot the Predicted Ripples
    spike_before = -10000  # Initialize spike_before to -10000
    for i, spike in enumerate(outputspikes_sec):
        label = 'Predicted Ripples' if i == 0 else None  # Add label only to the first
        if spike - refractory_period > spike_before:
            ax.fill_between([spike - max_detection_offset, spike + max_detection_offset], -5, 5, color='lightblue', alpha=0.2, label=label)
            spike_before = spike

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    plt.show()
    return fig, ax

plot_livetest(prefix="updnb4ds_100_8", parent_dir=parent_dir, downsampled_fs="30000_1000",
               window=(100000,150000), title='Live Test Data', xlabel='Time (s)', ylabel='Value',input=False)
