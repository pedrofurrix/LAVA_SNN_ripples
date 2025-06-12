import matplotlib.pyplot as plt
import numpy as np
import os
import json
import plotly.graph_objects as go
from plotly.offline import plot as plotly_save

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
    else: 
        start = 0
        end = len(data)   

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
    up_spike_times = np.where(spikes[:, 0] == 1)[0] + start
    down_spike_times = np.where(spikes[:, 1] == 1)[0] + start
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
        ax.vlines(up_spike_times_sec,0.8,1, color='green', alpha=0.3,label='Up Spikes')
        ax.vlines(down_spike_times_sec,-1,-0.8, color='red',alpha=0.3, label='Down Spikes')
        ax.scatter(outputspikes_sec, np.ones_like(outputspikes_sec)*-2, color='purple', marker='o', label='Output Spikes')
    else:
        ax.scatter(outputspikes_sec, np.ones_like(outputspikes_sec)*-2, color='purple', marker='o', label='Output Spikes')

    # Plot the Ground Truth Ripples
    for i,ripple in enumerate(gt_sec):
        label = 'Ground Truth Ripple' if i == 0 else None  # Add label only to the first
        ax.fill_between([ripple[0], ripple[1]], -3,3, color='yellow', alpha=0.2, label=label)
    
    # Plot the Predicted Ripples
    spike_before = -10000  # Initialize spike_before to -10000
    for i, spike in enumerate(outputspikes_sec):
        label = 'Predicted Ripples' if i == 0 else None  # Add label only to the first
        if spike - refractory_period > spike_before:
            ax.fill_between([spike - max_detection_offset, spike + max_detection_offset], -3, 3, color='lightblue', alpha=0.2, label=label)
            spike_before = spike

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    plt.show()
    return fig, ax




def plot_livetest_interactive(prefix, parent_dir, downsampled_fs, window=None, 
                               title='Live Test Data', xlabel='Time (s)', ylabel='Value', 
                               input=True, save_path=None,seed=None):
    # Load the data
    data_dir = os.path.join(parent_dir, "extract_Nripples", "train_pedro", "dataset_up_down", str(downsampled_fs))
    spikes = np.load(os.path.join(data_dir, 'concat_spikes.npy'))
    gt = np.load(os.path.join(data_dir, 'concat_ripples.npy'))
    data = np.load(os.path.join(data_dir, 'concat_data.npy'))
    if seed is not None:
        outputspikes = np.load(os.path.join(os.path.dirname(__file__), "spikes", f'{prefix}_spikes_seed{seed}.npy'))
        param_filename=f'{prefix}_results_seed{seed}.json'
    else:
        outputspikes = np.load(os.path.join(os.path.dirname(__file__), "spikes", f'{prefix}_spikes.npy'))
        param_filename=f'{prefix}_results.json'

    with open(os.path.join(os.path.dirname(__file__), param_filename), 'r') as f:
        params = json.load(f)
    if "time_duration" in params.keys():
        time_duration = params["time_duration"] * 1000
        window_data=np.arange(seed*time_duration,(seed+1)*time_duration,1)
        data=data[window_data]
        ripples_window=[]
        print("Window:", window_data[0], window_data[-1])
        for ripple in gt:
            if ripple[1] >= window_data[0] and ripple[0] <= window_data[-1]:
                ripples_window.append(ripple)
        gt=np.array(ripples_window)-window_data[0]

    max_detection_offset = params["max_detection_offset"] / 1000
    refractory_period = params["refractory_period_gt"] / 1000
    tolerance= params["tolerance"] / 1000
    if window is not None:
        start, end = window
    else:
        start = 0
        end = len(data)

    data = data[start:end]
    spikes = spikes[start:end]
    gt = gt[(gt[:, 1] >= start) & (gt[:, 0] < end)]
    outputspikes = outputspikes[(outputspikes >= start) & (outputspikes < end)]

    # Time axis in seconds
    time = np.arange(start, end) / 1000
    up_spike_times_sec = (np.where(spikes[:, 0] == 1)[0]+start) / 1000
    down_spike_times_sec = (np.where(spikes[:, 1] == 1)[0]+start) / 1000
    gt_sec = gt / 1000
    outputspikes_sec = outputspikes / 1000

    # Create figure
    fig = go.Figure()

    # Original signal
    fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name='Original Data', line=dict(color='blue')))

    # Input spikes
    if input:
        fig.add_trace(go.Scatter(x=up_spike_times_sec, y=[1.2]*len(up_spike_times_sec), mode='markers',
                                 marker=dict(color='green',symbol="triangle-down",  size=4), name='Up Spikes'))
        fig.add_trace(go.Scatter(x=down_spike_times_sec, y=[-1.2]*len(down_spike_times_sec), mode='markers',
                                 marker=dict(color='red',symbol="triangle-up", size=4), name='Down Spikes'))

    # Output spikes
    y_val = -2
    fig.add_trace(go.Scatter(x=outputspikes_sec, y=[y_val]*len(outputspikes_sec), mode='markers',
                             marker=dict(color='purple', symbol='circle', size=6), name='Output Spikes'))

    # Ground truth ripples
    for i, ripple in enumerate(gt_sec):
        fig.add_shape(type='rect',
                      x0=ripple[0], x1=ripple[1], y0=-3, y1=3,
                      line=dict(color='yellow'), fillcolor='yellow', opacity=0.2,
                      name='Ground Truth Ripple')
        if tolerance > 0:
            fig.add_shape(type='rect',
                        x0=ripple[0]-tolerance, x1=ripple[0], y0=-3, y1=3,
                        line=dict(color='orange'), fillcolor='orange', opacity=0.1,
                        name='Tolerance')

    # Predicted ripples (from output spikes)
    spike_before = -10000
    for spike in outputspikes_sec:
        if spike - refractory_period > spike_before:
            fig.add_shape(type='rect',
                          x0=spike - max_detection_offset, x1=spike,
                          y0=-3, y1=3,
                          line=dict(color='lightblue'), fillcolor='lightblue', opacity=0.2)
            spike_before = spike
        
        # Add legend entry for Ground Truth Ripple (yellow)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='yellow', width=10),
        name='Ground Truth Ripple'
    ))

    # Add legend entry for Predicted Ripple (lightblue)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='lightblue', width=10),
        name='Predicted Ripple'
    ))
    if tolerance > 0:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='orange', width=10),
            name='Tolerance'
        ))


   

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white",
        showlegend=True,
        autosize=True,
        width=None,
        height=None,
    )

    # Save HTML
    if save_path:
        if seed is not None:
            file=f"{prefix}_live_test_plot_seed{seed}.html"
        else:
            file=f"{prefix}_live_test_plot.html"
        file_path = os.path.join(save_path, file)
        os.makedirs(save_path, exist_ok=True)
        import plotly.io as pio
        pio.write_html(fig, file=file_path, auto_open=False,
                       full_html=True, include_plotlyjs='cdn',
                       config={"responsive": True})
    return fig



from plotly.subplots import make_subplots

def plot_livetest_channels(prefix, parent_dir, identifier, window=None, 
                               title='Live Test Data', xlabel='Time (s)', ylabel='Value', dataset=0,
                               input=True, save_path=None,seed=None,channels=[0]):
    # Load the data
    data_dir = os.path.join(parent_dir, "extract_Nripples", "train_pedro", "dataset_up_down", str(identifier))
    datasets=os.listdir(data_dir)
    spikes = np.load(os.path.join(data_dir,datasets[dataset], 'spike_data.npy'))
    gt = np.load(os.path.join(data_dir,datasets[dataset], 'ripples.npy'))
    data = np.load(os.path.join(data_dir,datasets[dataset], 'filtered_data.npy'))
    if seed is not None:
        outputspikes = np.load(os.path.join(os.path.dirname(__file__), "spikes", f'{prefix}_spikes_seed{seed}.npy'))
        param_filename=f'{prefix}_results_seed{seed}.json'
    else:
        outputspikes = np.load(os.path.join(os.path.dirname(__file__), "spikes", f'{prefix}_spikes.npy'))
        param_filename=f'{prefix}_results.json'

    with open(os.path.join(os.path.dirname(__file__), param_filename), 'r') as f:
        params = json.load(f)
    if "time_duration" in params.keys():
        time_duration = params["time_duration"] * 1000
        window_data=np.arange(seed*time_duration,(seed+1)*time_duration,1)
        data=data[window_data]
        ripples_window=[]
        print("Window:", window_data[0], window_data[-1])
        for ripple in gt:
            if ripple[1] >= window_data[0] and ripple[0] <= window_data[-1]:
                ripples_window.append(ripple)
        gt=np.array(ripples_window)-window_data[0]
    parameters=params["parameters"]
    max_detection_offset = parameters["max_detection_offset"] / 1000
    refractory_period = parameters["refractory_period_gt"] / 1000
    tolerance= parameters["tolerance"] / 1000
    if window is not None:
        start, end = window
    else:
        start = 0
        end = len(data)

    data = data[start:end,:]
    spikes = spikes[start:end,:]
    gt = gt[(gt[:, 1] >= start) & (gt[:, 0] < end)]
    outputspikes = outputspikes[dataset*8:(dataset+1)*8,:]
    filtered_outputspikes_sec = []
    
    for ch_spikes in outputspikes:
        ch_valid = ch_spikes[(ch_spikes >= start) & (ch_spikes < end)]
        filtered_outputspikes_sec.append(ch_valid/1000)
    # filtered_outputspikes = np.array(filtered_outputspikes)
    # Time axis in seconds
    time = np.arange(start, end) / 1000

    gt_sec = gt / 1000

    # Create figure
    fig = make_subplots(rows=len(channels), cols=1, shared_xaxes=True,shared_yaxes=True,)
                    # subplot_titles=[f"Channel {ch}" for ch in channels])

    for i, ch in enumerate(channels):

        spikes_ch= spikes[:, ch, :]
        up_spike_times_sec = (np.where(spikes_ch[:, 0] == 1)[0]+start) / 1000
        down_spike_times_sec = (np.where(spikes_ch[:, 1] == 1)[0]+start) / 1000
        # Plot raw data
        fig.add_trace(
            go.Scatter(x=time, y=data[:, ch], mode='lines', name=f'Ch {ch}'),
            row=i+1, col=1
        )

        # Input spikes
        if input:
            fig.add_trace(go.Scatter(x=up_spike_times_sec[:], y=[1.2]*len(up_spike_times_sec[:]), mode='markers',
                                    marker=dict(color='green',symbol="triangle-down",  size=4), 
                                    name='Up Spikes' if i == 0 else None, showlegend=(i == 0)),
                                    row=i+1, col=1)
            fig.add_trace(go.Scatter(x=down_spike_times_sec[:], y=[-1.2]*len(down_spike_times_sec[:]), mode='markers',
                                    marker=dict(color='red',symbol="triangle-up", size=4), 
                                    name='Down Spikes' if i == 0 else None, showlegend=(i == 0)),
                                    row=i+1, col=1)

        # Output spikes
        ch_spikes_sec = filtered_outputspikes_sec[ch]
        fig.add_trace(go.Scatter(x=ch_spikes_sec, y=[-2]*len(ch_spikes_sec), mode='markers',
                                marker=dict(color='purple', symbol='circle', size=6), 
                                name='Output Spikes' if i == 0 else None, showlegend=(i == 0)),
                                row=i+1, col=1)

        # Predicted ripple rectangles for this channel
        spike_before = -10000
        for spike in ch_spikes_sec:
            if spike - refractory_period > spike_before:
                fig.add_shape(type='rect',
                            x0=spike - max_detection_offset, x1=spike+tolerance,
                            y0=-3, y1=3,
                            line=dict(color='lightblue'), fillcolor='lightblue', opacity=0.2,
                            row=i+1, col=1)
                spike_before = spike

        # Set Y-axis range
        fig.update_yaxes(range=[-3.5, 3.5], row=i+1, col=1)

    # Add ground truth ripples
    for i in range(len(channels)):
        for ripple in gt_sec:
            fig.add_shape(type='rect',
                        x0=ripple[0], x1=ripple[1], y0=-3, y1=3,
                        line=dict(color='yellow'), fillcolor='yellow', opacity=0.2,
                        row=i+1, col=1)
            fig.add_shape(type='rect',
                        x0=ripple[0]-tolerance, x1=ripple[0]+max_detection_offset, y0=-3, y1=3,
                        line=dict(color='orange'), fillcolor='orange', opacity=0.1,
                        row=i+1, col=1)

    
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color='yellow', width=10), name='Ground Truth Ripple'))
     

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color='lightblue', width=10), name='Predicted Ripple'))

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color='orange', width=10), name='Tolerance'))


     # TEST CHANNEL HEIGHT
    # Dynamically scale figure height based on number of channels
    base_height_per_channel = 300  # pixels per channel row
    legend_height = 100  # space for legend
    total_height = len(channels) * base_height_per_channel + legend_height

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white",
        showlegend=True,
        autosize=True,
        width=None,
        height=total_height,
        hovermode='x unified',  # shows all channel tooltips together

    )

    # Save HTML
    if save_path:
        channels_str= "_".join(map(str, channels))
        if seed is not None:
            file=f"{prefix}_live_test_plot_seed{seed}_channels_{channels_str}.html"
        else:
            file=f"{prefix}_live_test_plot_channels_{channels_str}.html"
        file_path = os.path.join(save_path, file)
        os.makedirs(save_path, exist_ok=True)
        import plotly.io as pio
        pio.write_html(fig, file=file_path, auto_open=False,
                       full_html=True, include_plotlyjs='cdn',
                       config={"responsive": True})
        print(f"Plot saved to {file_path}")
    return fig


# prefix= "updnb4ds_100_7"
prefix="dsb4updn_median_200_7f"


window=(000000,50000)
# plot_livetest(prefix=prefix, parent_dir=parent_dir, downsampled_fs="30000_1000",
#                window=window, title='Live Test Data', xlabel='Time (s)', ylabel='Value',input=True)
identifier="1000_200_median"

save_path= os.path.join(os.path.dirname(__file__), "live_plots")
# fig=plot_livetest_interactive(prefix=prefix, parent_dir=parent_dir, downsampled_fs="30000_1000", window=window, 
#                                title='Live Test Data', xlabel='Time (s)', ylabel='Value', 
#                                input=True, save_path=save_path,seed=1)
# fig.show()

fig=plot_livetest_channels(prefix=prefix, 
                           parent_dir=parent_dir,
                           identifier=identifier, 
                           window=window, 
                           dataset=3,
                           channels=[0,1,2,3,4,5,6,7],
                           save_path=save_path,
                           input=True)