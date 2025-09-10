import matplotlib.pyplot as plt
import numpy as np
import os
import json
import plotly.graph_objects as go
from plotly.offline import plot as plotly_save

from collections import defaultdict
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
                               input=True, save_path=None,seed=None,channels=[0],tolerance=None,jitter=100,padding=0,max_detection_offset=80):
    
    ### Load the data ###
    data_dir = os.path.join(parent_dir, "extract_Nripples", "train_pedro", "dataset_up_down", str(identifier))
    datasets=os.listdir(data_dir)
    if "config.json" in datasets:
        datasets.remove("config.json")
    
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
    # max_detection_offset = parameters["max_detection_offset"] / 1000
    max_detection_offset/=1000  # Convert to seconds
    # refractory_period = parameters["refractory_period_gt"]/1000
    refractory_period = 0
    if tolerance is None:
        tolerance= parameters["tolerance"] / 1000
    else:
        tolerance = tolerance / 1000 # Convert to seconds
    padding = padding / 1000  # Convert padding to seconds
    if window is not None:
        start, end = window
        end=min(end,data.shape[0])
    else:
        start = 0
        end = len(data)

    data = data[start:end,:]
    spikes = spikes[start:end,:]
    gt = gt[(gt[:, 1] >= start) & (gt[:, 0] < end)]
    outputspikes = outputspikes[dataset*8:(dataset+1)*8,:]
    filtered_outputspikes_sec = []
    gt_sec = gt / 1000
    for ch_spikes in outputspikes:
        ch_valid = ch_spikes[(ch_spikes >= start) & (ch_spikes < end)]
        filtered_outputspikes_sec.append(ch_valid/1000)
    # filtered_outputspikes = np.array(filtered_outputspikes)
    # Time axis in seconds
    time = np.arange(start, end) / 1000

    # Classify output spikes into True Positives (TP) and False Positives (FP)
    classified_spikes_by_channel = defaultdict(list)

    for ch, spike_times in enumerate(filtered_outputspikes_sec):
        used_gt = set()  # Reset for each channel
        last_fp_time = -1e10
        last_tp_time = -1e10

        for spike in spike_times:
            matched_gt = False

            for gt_idx, (start_gt, end_gt) in enumerate(gt_sec):
                if gt_idx in used_gt:
                    continue  # Already matched for this channel

                # Match spike within expected window before GT onset
                if start_gt - tolerance <= spike <= start_gt + max_detection_offset + tolerance:
                    if spike - last_tp_time >= refractory_period:
                        classified_spikes_by_channel[ch].append(('TP', spike))
                        used_gt.add(gt_idx)
                        last_tp_time = spike
                        matched_gt = True
                        break

            if not matched_gt:
                # Make sure this unmatched spike is not inside *any* GT window
                in_any_gt = any(start_gt - tolerance <= spike <= max(start_gt - tolerance + max_detection_offset,end_gt) + padding for (start_gt, end_gt) in gt_sec)
                if not in_any_gt and spike - last_fp_time >= jitter:
                    classified_spikes_by_channel[ch].append(('FP', spike))
                    last_fp_time = spike
    

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

        if ch in classified_spikes_by_channel:
            for label, spike_time in classified_spikes_by_channel[ch]:
                fig.add_shape(
                    type='line',
                    x0=spike_time, x1=spike_time,
                    y0=-3, y1=3,
                    line=dict(
                        color='green' if label == 'TP' else 'red',
                        width=2,
                        dash='dash'
                    ),
                    row=i + 1, col=1
                )

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
     

    # fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
    #     line=dict(color='lightblue', width=10), name='Predicted Ripple'))

    # fig.add_trace(go.Scatter(
    # x=[None], y=[None], mode='lines',
    # line=dict(color='purple', width=3, dash='dash'),  # 'dash' for dashed lines
    # name='Predicted Ripple'
    # ))

    # Add invisible traces just for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                            line=dict(color='green', width=2, dash="dash"),
                            name='Predicted TP'))

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                            line=dict(color='red', width=2, dash='dash'),
                            name='Predicted FP'))

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color='orange', width=10,), name='Tolerance'))


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
            file=f"{prefix}_live_test_plot_seed{seed}_{datasets[dataset]}.html"
        else:
            file=f"{prefix}_live_test_plot_channels_{datasets[dataset]}.html"
        file_path = os.path.join(save_path, file)
        os.makedirs(save_path, exist_ok=True)
        import plotly.io as pio
        pio.write_html(fig, file=file_path, auto_open=False,
                       full_html=True, include_plotlyjs='cdn',
                       config={"responsive": True})
        print(f"Plot saved to {file_path}")
    return fig

adapt=20

# prefix= "updnb4ds_100_7"
prefix=f"dsb4updn_median_200_15f_adapt{adapt}" if adapt>0 else "dsb4updn_median_200_15f"
padding=100
tolerance=20


# window=(100000,200000)
window=(0, 500000)  
# plot_livetest(prefix=prefix, parent_dir=parent_dir, downsampled_fs="30000_1000",
#                window=window, title='Live Test Data', xlabel='Time (s)', ylabel='Value',input=True)

# identifier="1000_200_median"
identifier=f"30000_1000_100_adaptable{adapt}" if adapt>0 else "30000_1000_100"

save_path= os.path.join(os.path.dirname(__file__), "live_plots")
# fig=plot_livetest_interactive(prefix=prefix, parent_dir=parent_dir, downsampled_fs="30000_1000", window=window, 
#                                title='Live Test Data', xlabel='Time (s)', ylabel='Value', 
#                                input=True, save_path=save_path,seed=1)
# fig.show()

# fig=plot_livetest_channels(prefix=prefix, 
#                            parent_dir=parent_dir,
#                            identifier=identifier, 
#                            window=window, 
#                            dataset=3,
#                            channels=[0,3,6,],
#                            save_path=save_path, 
#                            padding=padding,
#                            tolerance=tolerance,
#                            input=True)


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
import json


def plot_livetest_channels_matplotlib(prefix, parent_dir, identifier, window=None,
                                       title='Live Test Data', xlabel='Time (s)', ylabel='Value', dataset=0,
                                       input=True, filename=None, seed=None, channels=[0],
                                       tolerance=None, jitter=100, padding=0, max_detection_offset=80):

    data_dir = os.path.join(parent_dir, "extract_Nripples", "train_pedro", "dataset_up_down", str(identifier))
    datasets = os.listdir(data_dir)
    if "config.json" in datasets:
        datasets.remove("config.json")

    spikes = np.load(os.path.join(data_dir, datasets[dataset], 'spike_data.npy'))
    gt = np.load(os.path.join(data_dir, datasets[dataset], 'ripples.npy'))
    data = np.load(os.path.join(data_dir, datasets[dataset], 'filtered_data.npy'))

    if seed is not None:
        outputspikes = np.load(os.path.join(os.path.dirname(__file__), "spikes", f'{prefix}_spikes_seed{seed}.npy'))
        param_filename = f'{prefix}_results_seed{seed}.json'
    else:
        outputspikes = np.load(os.path.join(os.path.dirname(__file__), "spikes", f'{prefix}_spikes.npy'))
        param_filename = f'{prefix}_results.json'

    with open(os.path.join(os.path.dirname(__file__), param_filename), 'r') as f:
        params = json.load(f)

    if "time_duration" in params:
        time_duration = params["time_duration"] * 1000
        window_data = np.arange(seed * time_duration, (seed + 1) * time_duration, 1)
        data = data[window_data]
        ripples_window = [ripple for ripple in gt if ripple[1] >= window_data[0] and ripple[0] <= window_data[-1]]
        gt = np.array(ripples_window) - window_data[0]

    parameters = params["parameters"]
    max_detection_offset /= 1000
    refractory_period = 0
    tolerance = parameters["tolerance"] / 1000 if tolerance is None else tolerance / 1000
    padding = padding / 1000
    jitter = jitter / 1000

    if window is not None:
        start, end = window
        start = int(start * 1000)
        end = int(end * 1000)
        end = min(end, data.shape[0])
    else:
        start, end = 0, len(data)

    data = data[start:end, :]
    spikes = spikes[start:end, :]
    gt = gt[(gt[:, 1] >= start) & (gt[:, 0] < end)]
    outputspikes = outputspikes[dataset * 8:(dataset + 1) * 8, :]

    gt_sec = gt / 1000
    filtered_outputspikes_sec = [(ch_spikes[(ch_spikes >= start) & (ch_spikes < end)] / 1000)
                                  for ch_spikes in outputspikes]
    time = np.arange(start, end) / 1000

    classified_spikes_by_channel = defaultdict(list)

    for ch, spike_times in enumerate(filtered_outputspikes_sec):
        used_gt = set()
        last_fp_time, last_tp_time = -1e10, -1e10
        for spike in spike_times:
            matched_gt = False
            for gt_idx, (start_gt, end_gt) in enumerate(gt_sec):
                if gt_idx in used_gt:
                    continue
                if start_gt - tolerance <= spike <= start_gt + max_detection_offset + tolerance:
                    if spike - last_tp_time >= refractory_period:
                        classified_spikes_by_channel[ch].append(('TP', spike))
                        used_gt.add(gt_idx)
                        last_tp_time = spike
                        matched_gt = True
                        # if spike <= start_gt:
                            # print(f"Spike before onset at time: {spike} s, GT onset at {start_gt} s")
                        break
            if not matched_gt:
                in_any_gt = any(start_gt - tolerance <= spike <= start_gt + tolerance + max_detection_offset + padding for (start_gt, end_gt) in gt_sec)
                if not in_any_gt and spike - last_fp_time >= jitter:
                    classified_spikes_by_channel[ch].append(('FP', spike))
                    last_fp_time = spike

    fig, axes = plt.subplots(len(channels), 1, figsize=(15, 3 * len(channels)), sharex=True, sharey=True)
    if len(channels) == 1:
        axes = [axes]

    for i, ch in enumerate(channels):
        ax = axes[i]
        ax.plot(time, data[:, ch], label=f"Ch {ch}", color="black", alpha=0.5)
        y_max = np.max(data[:, ch]) * 1.2
        y_min = np.min(data[:, ch]) * 1.2
        output_y = y_min + (y_max - y_min) * 0.2
        ch_spikes_sec = filtered_outputspikes_sec[ch]
        ax.scatter(ch_spikes_sec, [output_y]*len(ch_spikes_sec), color='purple', marker='o', s=30, label='Output Spikes')

        if input:
            spikes_ch = spikes[:, ch, :]
            up_spike_times_sec = (np.where(spikes_ch[:, 0] == 1)[0] + start) / 1000
            down_spike_times_sec = (np.where(spikes_ch[:, 1] == 1)[0] + start) / 1000
            spike_height = (y_max - y_min) * 0.2
            ax.vlines(up_spike_times_sec, ymin=y_max - 2*spike_height, ymax=y_max - spike_height, color='red', alpha=0.3, label='Up Spikes')
            ax.vlines(down_spike_times_sec, ymin=y_min + spike_height, ymax=y_min + 2*spike_height, color='blue', alpha=0.3, label='Down Spikes')

        # Identify TP indices to detect FNs
        used_gt = set(idx for (label, time_spike) in classified_spikes_by_channel[ch] if label == 'TP' 
                      for idx, (start_gt, end_gt) in enumerate(gt_sec)
                      if start_gt - tolerance <= time_spike <= start_gt + max_detection_offset + tolerance)

        for idx, (label, spike_time) in enumerate(classified_spikes_by_channel[ch]):
            color = 'green' if label == 'TP' else 'red'
            ax.axvline(x=spike_time, color=color, linestyle='--', linewidth=1, label=f"{label}" if idx == 0 else None)
            if label == "FP":
                ax.axvline(x=spike_time, color=color, linestyle='--', linewidth=1, label=f"{label}" if idx == 0 else None)
                ax.add_patch(Rectangle((spike_time, y_min), 0.1, y_max - y_min, alpha=0.1, color='red', label='FP' if idx == 0 else None))

        for idx, ripple in enumerate(gt_sec):
            ax.add_patch(Rectangle((ripple[0], y_min), ripple[1] - ripple[0], y_max - y_min, alpha=0.2, color='yellow', label='GT Ripple' if idx == 0 else None))
            ax.add_patch(Rectangle((ripple[0] - tolerance, y_min), max_detection_offset + 2 * tolerance, y_max - y_min, alpha=0.1, color='orange', label='Tolerance' if idx == 0 else None))

            if idx not in used_gt:
                # fn_center = ripple[0] + (ripple[1] - ripple[0]) / 2
                fn_center= ((ripple[0] - tolerance) +(ripple[0]+max_detection_offset+tolerance)) / 2
                ax.axvline(x=fn_center, color='blue', linestyle='--', linewidth=1, label='FN' if idx == 0 else None)

        ax.set_ylim([y_min, y_max])
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlim([time[0]-0.01*len(time)/1000, time[-1]+0.01*len(time)/1000])
    axes[-1].set_xlabel(xlabel, fontsize=14)
    axes[-1].tick_params(axis='both', labelsize=14)
    plt.tight_layout()

    if filename:
        save_path = os.path.join(os.path.dirname(__file__), "live_plots")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, filename), dpi=300)
        print(f"Plot saved to {os.path.join(save_path, filename)}")

    plt.show()
    return fig


def plot_small_part(prefix, parent_dir, identifier, window=None,
                                       title='Live Test Data', xlabel='Time (s)', ylabel='Value', dataset=0,
                                       input=True, seed=None, channels=[0],
                                       tolerance=None, jitter=100, padding=0, max_detection_offset=80,filename=None):

    data_dir = os.path.join(parent_dir, "extract_Nripples", "train_pedro", "dataset_up_down", str(identifier))
    datasets = os.listdir(data_dir)
    if "config.json" in datasets:
        datasets.remove("config.json")

    spikes = np.load(os.path.join(data_dir, datasets[dataset], 'spike_data.npy'))
    gt = np.load(os.path.join(data_dir, datasets[dataset], 'ripples.npy'))
    data = np.load(os.path.join(data_dir, datasets[dataset], 'filtered_data.npy'))

    if seed is not None:
        outputspikes = np.load(os.path.join(os.path.dirname(__file__), "spikes", f'{prefix}_spikes_seed{seed}.npy'))
        param_filename = f'{prefix}_results_seed{seed}.json'
    else:
        outputspikes = np.load(os.path.join(os.path.dirname(__file__), "spikes", f'{prefix}_spikes.npy'))
        param_filename = f'{prefix}_results.json'

    with open(os.path.join(os.path.dirname(__file__), param_filename), 'r') as f:
        params = json.load(f)

    if "time_duration" in params:
        time_duration = params["time_duration"] * 1000
        window_data = np.arange(seed * time_duration, (seed + 1) * time_duration, 1)
        data = data[window_data]
        ripples_window = [ripple for ripple in gt if ripple[1] >= window_data[0] and ripple[0] <= window_data[-1]]
        gt = np.array(ripples_window) - window_data[0]

    parameters = params["parameters"]
    max_detection_offset /= 1000
    refractory_period = 0
    tolerance = parameters["tolerance"] / 1000 if tolerance is None else tolerance / 1000
    padding = padding / 1000
    jitter = jitter / 1000  # Convert jitter to seconds
    if window is not None:
        start, end = window
        start = int(start * 1000)
        end = int(end * 1000)
        end = min(end, data.shape[0])
    else:
        start, end = 0, len(data)

    data = data[start:end, :]
    spikes = spikes[start:end, :]
    gt = gt[(gt[:, 1] >= start) & (gt[:, 0] < end)]
    outputspikes = outputspikes[dataset * 8:(dataset + 1) * 8, :]

    time = np.arange(end - start) / 1000

    gt_sec = (gt - start) / 1000
    filtered_outputspikes_sec = [(ch_spikes[(ch_spikes >= start) & (ch_spikes < end)] - start) / 1000
                                  for ch_spikes in outputspikes]

    classified_spikes_by_channel = defaultdict(list)
    used_gt_by_channel = {}
    for ch, spike_times in enumerate(filtered_outputspikes_sec):
        used_gt = set()
        last_fp_time, last_tp_time = -1e10, -1e10
        for spike in spike_times:
            matched_gt = False
            for gt_idx, (start_gt, end_gt) in enumerate(gt_sec):
                if gt_idx in used_gt:
                    continue
                if start_gt - tolerance <= spike <= start_gt + max_detection_offset + tolerance:
                    if spike - last_tp_time >= refractory_period:
                        classified_spikes_by_channel[ch].append(('TP', spike))
                        used_gt.add(gt_idx)
                        last_tp_time = spike
                        matched_gt = True
                        break
            if not matched_gt:
                in_any_gt = any(start_gt - tolerance <= spike <= start_gt + tolerance + max_detection_offset + padding for (start_gt, end_gt) in gt_sec)
                if not in_any_gt and spike - last_fp_time >= jitter:
                    classified_spikes_by_channel[ch].append(('FP', spike))
                    last_fp_time = spike
        used_gt_by_channel[ch] = used_gt

    fig, axes = plt.subplots(len(channels), 1, figsize=(8, 3 * len(channels)), sharex=True, sharey=True)
    if len(channels) == 1:
        axes = [axes]

    for i, ch in enumerate(channels):
        used_gt = used_gt_by_channel[ch]  # correct set for this channel
        ax = axes[i]
        ax.plot(time, data[:, ch], label=f"Ch {ch}", color="black", alpha=0.5)
        # y_max = np.max(data[:, ch]) * 1.2
        # y_min = np.min(data[:, ch]) * 1.2
        y_min=-1.75
        y_max=1.75
        output_y = y_min + (y_max - y_min) * 0.2
        ch_spikes_sec = filtered_outputspikes_sec[ch]
        ax.scatter(ch_spikes_sec, [output_y] * len(ch_spikes_sec), color='purple', marker='o', s=30, label='Output Spikes')
        if input:
            spikes_ch = spikes[:, ch, :]
            up_spike_times_sec = np.where(spikes_ch[:, 0] == 1)[0] / 1000
            down_spike_times_sec = np.where(spikes_ch[:, 1] == 1)[0] / 1000
            spike_height = (y_max - y_min) * 0.2
            ax.vlines(up_spike_times_sec, ymin=y_max - 2 * spike_height, ymax=y_max - spike_height, color='red', alpha=0.3, label='Up Spikes')
            ax.vlines(down_spike_times_sec, ymin=y_min + spike_height, ymax=y_min + 2 * spike_height, color='blue', alpha=0.3, label='Down Spikes')

        for idx, (label, spike_time) in enumerate(classified_spikes_by_channel[ch]):
            color = 'green' if label == 'TP' else 'red'
            ax.axvline(x=spike_time, color=color, linestyle='--', linewidth=1, label=f"{label}" if idx == 0 else None)
            if label == "FP":
                ax.add_patch(Rectangle((spike_time, y_min), 0.1, y_max - y_min, alpha=0.1, color='red', label='FP' if idx == 0 else None))

        for idx, ripple in enumerate(gt_sec):
            ax.add_patch(Rectangle((ripple[0], y_min), ripple[1] - ripple[0], y_max - y_min, alpha=0.2, color='yellow', label='GT Ripple' if idx == 0 else None))
            ax.add_patch(Rectangle((ripple[0] - tolerance, y_min), max_detection_offset+tolerance*2, y_max - y_min, alpha=0.1, color='orange', label='Tolerance' if idx == 0 else None))
            if idx not in used_gt:
                # fn_center = ripple[0] + (ripple[1] - ripple[0]) / 2
                fn_center= ((ripple[0] - tolerance) +(ripple[0]+max_detection_offset+tolerance)) / 2
                ax.axvline(x=fn_center, color='blue', linestyle='--', linewidth=1, label='FN' if idx == 0 else None)
        ax.set_ylim([y_min, y_max])
        # ax.set_ylabel(f"Ch {ch}")
        # ax.set_xticks(np.arange(0, (end - start)+25, 25))
        ax.set_xticklabels([])
        ax.tick_params(axis='both', labelsize=14)
        # ax.set_title(f"Channel {ch}", fontsize=14)
    # axes[-1].set_xlabel(xlabel)
    # fig.suptitle(title)
    plt.tight_layout()

    if filename:
        save_path = os.path.join(os.path.dirname(__file__),"live_plots")
        fig.savefig(os.path.join(save_path, filename), dpi=300)
        print(f"Plot saved to {os.path.join(save_path, filename)}")

    plt.show()
    return fig

adapt=0
prefix="updnb4ds_100_7"
# prefix=f"dsb4updn_median_200_12b"
prefix+=f"_{adapt}" if adapt>0 else ""
identifier="30000_1000_100"
window=None
# window=(100,110)

dataset=3
filename=None
# filename=f"{prefix}_{dataset}_100_110.png"

# fig=plot_livetest_channels_matplotlib(prefix, parent_dir, identifier, window=window,
#                                        title='Live Test Data', xlabel=None, ylabel='Amigo', dataset=dataset,
#                                        input=False, filename=filename, seed=None, channels=[1],
#                                        tolerance=20, jitter=100, padding=100, max_detection_offset=80)
# window=(301.4, 302.8)  # Example window in seconds
# window=(301.4,302.0)
# window=(216.4,216.6)
# window=(209.7,210.7)
window=input("Enter window in seconds (e.g., 209.7,210.7): ")
start_str, end_str = window.split(",")
start, end = float(start_str), float(end_str)
window = (start, end)

i=input("Enter figure index (e.g., 1): ")
filename=f"fn_{i}.png"
fig=plot_small_part(prefix, parent_dir, identifier, window=window,
                                       title='Live Test Data', xlabel='Time (s)', ylabel='Value', dataset=dataset,
                                       input=False, filename=filename, seed=None, channels=[1],
                                       tolerance=20, jitter=100, padding=100, max_detection_offset=80)
# fig.show()