def plot_livetest_channels(prefix, parent_dir, identifier, window=None, 
                           title='Live Test Data', xlabel='Time (s)', ylabel='Value', dataset=0,
                           input=True, save_path=None, seed=None, channels=[0]):
    import os, json
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    data_dir = os.path.join(parent_dir, "extract_Nripples", "train_pedro", "dataset_up_down", str(identifier))
    datasets = os.listdir(data_dir)
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
        window_data = np.arange(seed * time_duration, (seed + 1) * time_duration)
        data = data[window_data]
        spikes = spikes[window_data]
        ripples_window = [ripple for ripple in gt if ripple[1] >= window_data[0] and ripple[0] <= window_data[-1]]
        gt = np.array(ripples_window) - window_data[0]

    max_detection_offset = params["max_detection_offset"] / 1000
    refractory_period = params["refractory_period_gt"] / 1000
    tolerance = params["tolerance"] / 1000

    if window is not None:
        start, end = window
    else:
        start = 0
        end = len(data)

    data = data[start:end, :]
    spikes = spikes[start:end, :]
    gt = gt[(gt[:, 1] >= start) & (gt[:, 0] < end)]
    outputspikes = outputspikes[dataset * 8:(dataset + 1) * 8, :]

    filtered_outputspikes_sec = []
    for ch_spikes in outputspikes:
        ch_valid = ch_spikes[(ch_spikes >= start) & (ch_spikes < end)]
        filtered_outputspikes_sec.append(ch_valid / 1000)

    time = np.arange(start, end) / 1000
    up_spike_times_sec = (np.where(spikes[:, 0] == 1)[0] + start) / 1000
    down_spike_times_sec = (np.where(spikes[:, 1] == 1)[0] + start) / 1000
    gt_sec = gt / 1000

    fig = make_subplots(rows=len(channels), cols=1, shared_xaxes=True,
                        subplot_titles=[f"Channel {ch}" for ch in channels])

    for i, ch in enumerate(channels):
        fig.add_trace(go.Scatter(x=time, y=data[:, ch], mode='lines', name=f'Ch {ch}'),
                      row=i + 1, col=1)

        # Input Spikes
        if input:
            fig.add_trace(go.Scatter(x=up_spike_times_sec, y=[1.2] * len(up_spike_times_sec), mode='markers',
                                     marker=dict(color='green', symbol="triangle-down", size=4),
                                     name='Up Spikes' if i == 0 else None),
                          row=i + 1, col=1)

            fig.add_trace(go.Scatter(x=down_spike_times_sec, y=[-1.2] * len(down_spike_times_sec), mode='markers',
                                     marker=dict(color='red', symbol="triangle-up", size=4),
                                     name='Down Spikes' if i == 0 else None),
                          row=i + 1, col=1)

        y_val = -2
        channel_output = filtered_outputspikes_sec[ch]
        fig.add_trace(go.Scatter(x=channel_output, y=[y_val] * len(channel_output), mode='markers',
                                 marker=dict(color='purple', symbol='circle', size=6),
                                 name='Output Spikes' if i == 0 else None),
                      row=i + 1, col=1)

        # Match output spikes with GT (TP, FP logic)
        gt_used = np.zeros(len(gt_sec), dtype=bool)
        spike_before = -10000

        for spike in channel_output:
            if spike - refractory_period > spike_before:
                spike_before = spike
                matched = False
                for idx, (start_r, end_r) in enumerate(gt_sec):
                    if (start_r - tolerance <= spike <= end_r + tolerance) and not gt_used[idx]:
                        matched = True
                        gt_used[idx] = True
                        break
                fig.add_shape(type='rect',
                              x0=spike - max_detection_offset, x1=spike + max_detection_offset,
                              y0=-3, y1=3,
                              line=dict(color='blue'), fillcolor='lightblue', opacity=0.3,
                              row=i + 1, col=1)
                if matched:
                    fig.add_trace(go.Scatter(x=[spike], y=[y_val],
                                             mode='markers', marker=dict(color='green', symbol='star', size=10),
                                             name="TP" if i == 0 else None),
                                  row=i + 1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=[spike], y=[y_val],
                                             mode='markers', marker=dict(color='red', symbol='x', size=10),
                                             name="FP" if i == 0 else None),
                                  row=i + 1, col=1)

        # FN highlighting: GT that wasn’t matched
        for idx, (start_r, end_r) in enumerate(gt_sec):
            fig.add_shape(type='rect',
                          x0=start_r, x1=end_r, y0=-3, y1=3,
                          line=dict(color='yellow'), fillcolor='yellow', opacity=0.2,
                          row=i + 1, col=1)
            if not gt_used[idx]:
                fig.add_trace(go.Scatter(x=[(start_r + end_r) / 2], y=[2.5],
                                         mode='markers', marker=dict(color='black', symbol='cross', size=10),
                                         name="FN" if i == 0 else None),
                              row=i + 1, col=1)

            if tolerance > 0:
                fig.add_shape(type='rect',
                              x0=start_r - tolerance, x1=start_r,
                              y0=-3, y1=3,
                              line=dict(color='orange'), fillcolor='orange', opacity=0.1,
                              row=i + 1, col=1)

        fig.update_yaxes(range=[-3.5, 3.5], row=i + 1, col=1)

    # Legend elements
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='green', symbol='star', size=10),
                             name='True Positive'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='red', symbol='x', size=10),
                             name='False Positive'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='black', symbol='cross', size=10),
                             name='False Negative'))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white",
        showlegend=True,
        autosize=True,
    )

    # Save HTML
    if save_path:
        if seed is not None:
            channels_str = "_".join(map(str, channels))
            file = f"{prefix}_live_test_plot_seed{seed}_channels{channels_str}.html"
        else:
            file = f"{prefix}_live_test_plot.html"
        file_path = os.path.join(save_path, file)
        os.makedirs(save_path, exist_ok=True)
        import plotly.io as pio
        pio.write_html(fig, file=file_path, auto_open=False,
                       full_html=True, include_plotlyjs='cdn',
                       config={"responsive": True})

    return fig
 