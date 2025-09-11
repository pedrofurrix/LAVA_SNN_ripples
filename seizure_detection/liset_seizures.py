###########################################################################################
#                                                                                         #
#                         Developed by: Marcos Oriol Pagonabarraga                          #
#                           Contact: marcos.oriol.p@gmail.com                             #
#                                                                                         #
###########################################################################################

#
#  This module is developed for 3 main reasons:
#           - Loading correctly the data from liset labs (Instituto Cajal)
#           - Testing different models.
#           - Visualizing performance and data quality.
#
#

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from copy import deepcopy
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(parent_dir)
# from format_predictions import get_predictions_indexes
from liset_tk.liset_aux import *
# from liset_tk.load_data import *
from liset_tk.signal_aid import *




class liset_seizures():
    """
    Class for handling data processing and visualization related to ripple events.

    Parameters:
    - data_path (str): Path to the directory containing the data.
    - shank (int): Shank of the electrode.
    - downsample (bool, optional): Whether to downsample the data. Default is False.
    - normalize (bool, optional): Whether to normalize the data. Default is True.
    - numSamples (int, optional): Number of samples. Default is False.
    - verbose (bool, optional): Whether to display verbose output. Default is True.
    """
     
    def __init__(self, data_path, shank, downsample = False, normalize = True, numSamples = False, start = 0, verbose=True, original_fs=30000):

        # Set the verbose
        self.verbose = verbose
        self.numSamples = numSamples
        self.start = start
        self.original_fs = original_fs
        if downsample:
            self.downsampled_fs = downsample
            self.fs_conv_fact = self.original_fs/self.downsampled_fs
        else:
            self.fs_conv_fact = 1
        self.shank=shank-1
        # Initialize class variables.
        self.prediction_times = []
        self.model = None
        self.default_channels = [16, 4, 1, 13, 15, 3, 2, 14]

        # Load the data.
        self.load(data_path, self.shank, downsample = downsample, normalize=normalize)


        # Try to load the ripples if the path contain a file with ripple times.
        # Only load ripples in the interval selected of data (chunk)
        # if hasattr(self, 'fs'):
        #     self.ripples_GT = self.ripples_in_chunk(load_ripple_times(data_path), start, numSamples, self.fs, self.fs_conv_fact)
        
        #     if type(self.ripples_GT) is not np.ndarray:
        #         self.has_ripples = False
        #         self.has_ripplesGT = False
        #     else:
        #         self.has_ripples = True
        #         self.has_ripplesGT = True
        #         self.num_ripples = len(self.ripples_GT)
        #         self.ripples_GT = (self.ripples_GT * self.fs - start / self.fs_conv_fact).astype(int)

    @plain_plot
    @hide_y_ticks_on_offset
    def plot_event(self, 
                   event, 
                   offset=0, 
                   extend=0, 
                   delimiter=False, 
                   show=True, 
                   filtered=[], 
                   title='', 
                   label='', 
                   ch=False,
                   ylim=False,
                   line_color=False,
                   show_ground_truth=False, 
                   show_predictions=False, 
                   plain=False):
        """
        Plot the ripple signal number idx.

        Parameters:
        - idx (int): Index of the ripple to plot.
        - offset (float): Offset between channels for visualization.
        - extend (float): Extend the plotted time range before and after the ripple.
        - delimiter (bool): Whether to highlight the ripple area.

        Returns:
        - fig (matplotlib.figure.Figure): The generated figure.
        - ax (matplotlib.axes.Axes): The axes object containing the plot.
        """
            
        prop = self.fs_conv_fact
        interval = deepcopy(event)
        handles = []
        labels = []

        try:
            if extend != 0:
                if (interval[0] - extend) < 0:
                    interval[0] = int(self.start / prop)
                else:
                    interval[0] = interval[0] - extend

                if (interval[1] + extend) > self.numSamples/prop:
                    interval[1] = int((self.start + self.numSamples)/prop)
                else:
                    interval[1] = interval[1] + extend

        except IndexError:
            print('IndexError')
            print(f'There no data available for the selected samples.\nLength of loaded data: {int(self.numSamples/self.fs_conv_fact)}')
            return None, None

        # Define window data
        self.window_interval = interval
        mask = (self.ripples_GT[:, 1] >= interval[0]) & (self.ripples_GT[:, 0] <= interval[1])
        self.window_ripples = self.ripples_GT[mask]

        interval_data = self.data[interval[0]: interval[1]][:]
        self.window = deepcopy(interval_data)
        
        time_vector = np.linspace(interval[0] / self.fs, interval[1] / self.fs, interval_data.shape[0])
        if show:
            fig, ax = plt.subplots(figsize=(10, 6))
        for i, chann in enumerate(interval_data.transpose()):
            if filtered:
                bandpass = filtered
                chann = bandpass_filter(chann, bandpass, self.fs)
                self.window[:, i] = chann
            if show:
                if ch:
                    if i in ch:
                        if line_color:
                            ax.plot(time_vector, chann + i * offset, line_color)
                        else:
                            ax.plot(time_vector, chann + i * offset)
                else:
                    if line_color:
                        ax.plot(time_vector, chann + i * offset, line_color)
                    else:
                        ax.plot(time_vector, chann + i * offset)
            
            if ylim:
                ax.set_ylim(ylim)
                

        max_val = np.max(self.window.reshape((self.window.shape[0]*self.window.shape[1]))) + offset*8
        min_val = np.min(self.window.reshape((self.window.shape[0]*self.window.shape[1])))

        if delimiter and show:
            if extend > 0:
                ripple_area = [time_vector[round(extend)], time_vector[-round(extend)]]
                if not label:
                    label='Event area'
                fill_DEL = ax.fill_between(ripple_area, min_val, max_val, color="tab:blue", alpha=0.2)
                handles.append(fill_DEL)
                labels.append(label)
            else:
                if self.verbose:
                    print('Delimiter not applied because there is no extend.')

        if show_ground_truth:
            if hasattr(self.ripples_GT, 'dtype'):
                for ripple in self.window_ripples:
                    fill_GT = ax.fill_between([ripple[0] / self.fs, ripple[1] / self.fs],  min_val, max_val, color="tab:red", alpha=0.3)

            if 'fill_GT' in locals():
                handles.append(fill_GT)
                labels.append('Ground truth' if not label else label)

        if show_predictions:
            if hasattr(self, 'prediction_idxs'):
                mask = (self.prediction_idxs[:, 1] >= interval[0]) & (self.prediction_idxs[:, 0] <= interval[1])
                self.prediction_times_from_window = self.prediction_times[mask]
                for times in self.prediction_times_from_window:
                    fill_PRED = ax.fill_between([times[0], times[1]], min_val, max_val, color="tab:blue", alpha=0.3)

            if 'fill_PRED' in locals():
                handles.append(fill_PRED)
                labels.append(f'{self.model_type} predict')

        # Figure styles
        if filtered and not title:
            title = f'Filtered channels\nEvent {interval}\nBandpass: {bandpass[0]}-{bandpass[1]}'
        if not title:
            title = f'Channels for samples {interval}'

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        if not len(handles) == 0:        
            ax.legend(handles, labels)

        text = ax.set_title(title, loc='center', fontfamily='serif', fontsize=12, fontweight='bold')
        ax.grid(True)
        self.fig = fig
        self.ax = ax

        if show:
            return fig, ax
                

    def ripples_in_chunk(self, ripples, start, numSamples, fs, prop):
        if not numSamples:
            numSamples = self.file_samples - self.start

        in_chunk = ripples[(ripples[:,0] > start/prop/fs) & (ripples[:,0] < (start + numSamples)/prop/fs)]

        return in_chunk
    
    def IISs_in_chunk(self, start, numSamples, fs, prop):
        """
        Return IIS times within the selected chunk.
        """
        if not numSamples:
            numSamples = self.file_samples - self.start

        lower = start / prop / fs
        upper = (start + numSamples) / prop / fs

        in_chunk = self.IISs_times[
            (self.IISs_times >= lower * fs) &
            (self.IISs_times <= upper * fs)
        ]
        return in_chunk
    
    def seizures_in_chunk(self, start, numSamples, fs, prop):
        """
        Return seizure intervals overlapping the selected chunk.
        Safeguards against single seizure (shape (2,)) or no seizures.
        """
        if not hasattr(self, "seizure_times"):
            return np.empty((0, 2), dtype=int)

        # ensure seizure_times is 2D: (N,2)
        seizure_times = np.array(self.seizure_times)
        if seizure_times.ndim == 1:
            if seizure_times.size == 0:
                return np.empty((0, 2), dtype=int)
            elif seizure_times.size == 2:
                seizure_times = seizure_times.reshape(1, 2)
            else:
                raise ValueError(f"Unexpected seizure_times shape: {seizure_times.shape}")

        if not numSamples:
            numSamples = self.file_samples - self.start

        lower = start / prop
        upper = (start + numSamples) / prop

        # keep seizures that overlap with [lower, upper]
        in_chunk = seizure_times[
            (seizure_times[:, 1] >= lower) &
            (seizure_times[:, 0] <= upper)
        ]

        return in_chunk


    def load_dat(self, path, channels, numSamples = False, verbose=False):
        """
        Load data from a .dat file.

        Parameters:
        - path (str): Path to the directory containing the .dat file.
        - channels (list): List of channel IDs to load.
        - numSamples (int, optional): Number of samples to load. Default is False (load all samples).
        - sampleSize (int, optional): Size of each sample in bytes. Default is 2.
        - verbose (bool, optional): Whether to display verbose output. Default is False.

        Returns:
        - data (numpy.ndarray): Loaded data as a NumPy array.
        """

        try:
            filename = f"{path}/{[i for i in os.listdir(path) if i.endswith('.dat')][0]}"
            self.file_len = os.path.getsize(filename=filename)
            self.file_samples = self.file_len / self.n_channels / 2
        except:
            if self.verbose:
                print('.dat file not in path')
            return False
        
        nChannels = len(channels)
        if (len(channels) > nChannels):
            if self.verbose:
                print("Cannot load specified channels (listed channel IDs inconsistent with total number of channels).")
            return False
        
        start = self.start * self.n_channels * 2
        numSamples = self.numSamples * self.n_channels * 2

        if start > self.file_len:
            if self.verbose:
                print(f'the start must be lower than the total file samples.\nTotal file samples: {self.file_samples}')
            return False
        if (numSamples + start) > self.file_len:
            numSamples = self.file_len - start


        if (self.file_len < numSamples) or ((numSamples + self.start) > self.file_len):
            if self.verbose:
                print(f'file has only {self.file_samples} samples')
            return False
            
        with open(filename, "rb") as f:
            # Put the reader at the starting index
            f.seek(start)

            if numSamples:
                raw = f.read(numSamples)
            else:
                raw = f.read(self.file_len - start)
            data = np.frombuffer(raw, dtype=np.int16)
            data = RAW2ORDERED(data, channels, self.n_channels)
            if self.verbose:
                print(f"Data loaded from {filename}")
                print(f"Data shape: {data.shape}")
                print(f"Data type: {data.dtype}")
            return data
            

    def load(self, data_path, shank, downsample, normalize):
        """
        Load all, optionally downsample and normalize it.

        Parameters:
        - data_path (str): Path to the data directory.
        - shank (int): Shank of the electrode.
        - downsample (float): Downsample factor.
        - normalize (bool): Whether to normalize the data.
    
        Returns:
        - data (numpy.ndarray): Loaded and processed data.
        """
        mat_file= [f for f in os.listdir(data_path) if f.endswith(".mat")][0]
        try:
            self.info = loadmat(f'{data_path}/{mat_file}')
            if self.verbose:
                print('.mat file loaded with scipy')
        except:
            try:
                self.info = h5py.File(f'{data_path}/{mat_file}', 'r')
                if self.verbose:
                    print('.mat file loaded with h5py')
            except:
                print('.mat file cannot be opened or is not in path.')
                return
            
        try:
            channels = self.info['neurosparkmat']['channels'][0][0]
            self.n_channels=len(channels)
            channels = channels[8 * (shank):8 * (shank + 1)]
        except Exception as err:
            try:
                channels =np.array(self.info['neurosparkmat']['channels'])
                channels = channels.flatten()
                self.n_channels=len(channels)

                channels = channels[8*(shank):8*(shank+1)]

            except Exception as err:  
                print(f'No data available for shank {shank}\n\n{err}')
                return 

        if self.verbose:
                    print(f'Channels loaded: {channels}')
                    print(f'Number of channels: {self.n_channels}')
        raw_data = self.load_dat(data_path, channels, numSamples=self.numSamples)

        if hasattr(raw_data, 'shape'):
            self.data = self.clean(raw_data, downsample, normalize)
            self.duration = self.data.shape[0]/self.fs
        self.load_IISs_times()
        self.load_seizure_times()
        self.IISs_times=self.IISs_in_chunk(self.start, self.numSamples, self.fs, self.fs_conv_fact)
        self.seizure_times=self.seizures_in_chunk(self.start, self.numSamples, self.fs, self.fs_conv_fact)

    def clean(self, data, downsample, normalize):
        """
        Clean the loaded data by downsampling and normalizing it.

        Parameters:
        - data (numpy.ndarray): Raw data to be cleaned.
        - downsample (bool): Whether to downsample the data.
        - normalize (bool): Whether to normalize the data.

        Returns:
        - data (numpy.ndarray): Cleaned data after downsampling and normalization.
        """

        if downsample:
            self.fs = self.downsampled_fs
            # Downsample data
            if self.verbose:
                print("Downsampling data from %d Hz to %d Hz..."%(self.original_fs, self.downsampled_fs), end=" ")
            data = downsample_data(data, self.original_fs, self.downsampled_fs)
            if self.verbose:
                print("Done!")
        else:
            self.fs = self.original_fs


        if normalize:
            # Normalize it with z-score
            if self.verbose:
                print("Normalizing data...", end=" ")
            data = z_score_normalization(data)
            if self.verbose:
                print("Done!")
                print("Shape of loaded data after downsampling and z-score: ", np.shape(data))
     
        return data
        
    
    def savefig(self, fname, background=False):
        if fname.endswith('.svg'):
            self.fig.savefig(fname, transparent=not background, format='svg', bbox_inches='tight')
        else:
            self.fig.savefig(fname, transparent=not background, bbox_inches='tight')

    def load_IISs_times(self):
        if hasattr(self,"fs"):
            try:
                times = self.info['neurosparkmat']['IISs'][0,0]['times'][self.shank, 0]
                SD = self.info['neurosparkmat']['IISs'][0,0]['SD']
            except:
                ref = self.info['neurosparkmat']['IISs']['times'][self.shank, 0]
                data_object = self.info[ref]
                times = data_object[()]
                SD = np.array(self.info['neurosparkmat']['IISs']['SD'])
            self.IISs_times = np.array(times).flatten()
            self.has_IISs = True
            self.num_IISs = len(self.IISs_times)
            self.IISs_times = (self.IISs_times * self.fs - self.start / self.fs_conv_fact).astype(int)
            self.SD=SD.squeeze()
            if self.verbose:
                print(f"IISs times loaded: {self.IISs_times}")
                print(f"IISs SD loaded: {self.SD}")
                print(f"Number of IISs loaded: {self.num_IISs}")
        else:
            if self.verbose:
                print("Cannot load IISs times because the sampling frequency is not defined.")

    def load_seizure_times(self):
        if hasattr(self,"fs"):
            try:
                times = self.info['neurosparkmat']['seizure'][0,0]['times'][self.shank, 0]
                thrRate= self.info['neurosparkmat']['seizure'][0,0]['thrRate']
                dt = self.info['neurosparkmat']['seizure'][0,0]['dt']
            except:
                ref = self.info['neurosparkmat']['seizure']['times'][self.shank, 0]
                data_object = self.info[ref]
                times = data_object[()]
                thrRate = np.array(self.info['neurosparkmat']['seizure']['thrRate'])
                dt = np.array(self.info['neurosparkmat']['seizure']['dt'])
            # self.seizure_times = np.array(times).flatten()
            self.seizure_times=times.T
            self.has_seizures = True
            self.num_seizures = len(self.seizure_times)
            self.seizure_times = (self.seizure_times * self.fs - self.start / self.fs_conv_fact).astype(int)
            self.thrRate=thrRate.squeeze()
            self.seizure_dt=dt.squeeze()
            if self.verbose:
                print(f"Seizure times loaded: {self.seizure_times}")
                print(f"Seizure thresholds loaded: {self.thrRate}")
                print(f"Seizure durations loaded: {self.seizure_dt}")
                print(f"Number of seizures loaded: {self.seizure_times.shape}")
        else:
            if self.verbose:
                print("Cannot load seizure times because the sampling frequency is not defined.")

    def plot_event_with_IISs_and_seizures(
        self,
        event,             # [start, end] in samples
        offset=0,
        extend=0,
        delimiter=False,
        show=True,
        filtered=[],
        title='',
        ch=False,
        ylim=False,
        line_color=False,
        show_IISs=True,
        show_seizures=True,
    ):
        """
        Plot data around an event, with IISs (spikes) and seizures overlaid.
        """
        prop = self.fs_conv_fact
        interval = deepcopy(event)
        interval=[interval[0]*self.fs, interval[1]*self.fs]
        
        if extend != 0:
            interval[0] = max(int(interval[0] - extend), 0)
            interval[1] = min(int(interval[1] + extend), self.numSamples)

        # Slice data
        interval_data = self.data[interval[0]: interval[1], :]
        self.window = deepcopy(interval_data)

        time_vector = np.linspace(interval[0] / self.fs,
                                interval[1] / self.fs,
                                interval_data.shape[0])

        if show:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot channels
        for i, chann in enumerate(interval_data.T):
            if filtered:
                chann = bandpass_filter(chann, filtered, self.fs)
                self.window[:, i] = chann
            if show:
                if not ch or i in ch:
                    if line_color:
                        ax.plot(time_vector, chann + i * offset, line_color)
                    else:
                        ax.plot(time_vector, chann + i * offset)


        if ylim:
            ax.set_ylim(ylim)

        # Determine vertical range
        max_val = np.max(self.window) + offset * 8
        min_val = np.min(self.window)

        # Overlay IISs (point events)
        if show_IISs and hasattr(self, "IISs_times"):
            IIS_times_sec = self.IISs_times / self.fs
            mask = (IIS_times_sec >= time_vector[0]) & (IIS_times_sec <= time_vector[-1])
            IIS_in_window = IIS_times_sec[mask]
            for t in IIS_in_window:
                ax.axvline(t, color="red", linestyle="--", alpha=0.7, label="IIS")

        # Overlay seizures (intervals)
        if show_seizures and hasattr(self, "seizure_times"):
            seizure_times_sec=self.seizure_times / self.fs
            for start, end in seizure_times_sec:
                start = max(start, interval[0]/self.fs)
                end = min(end, interval[1]/self.fs)
                if end >= time_vector[0] and start <= time_vector[-1]:
                    fill_SZ = ax.fill_between([start, end],
                                            min_val, max_val,
                                            color="tab:purple", alpha=0.3,
                                            label="Seizure")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")
        ax.set_title(title if title else f"Event samples {interval}")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())  # deduplicate
        ax.grid(True)

        if show:
            return fig, ax
 
def z_score_normalization(data):
	channels = range(np.shape(data)[1])

	for channel in channels:
		# Since data is in float16 type, we make it smaller to avoid overflows
		# and then we restore it.
		# Mean and std use float64 to have enough space
		# Then we convert the data back to float16
		dmax = np.amax(data[:, channel])
		dmin = abs(np.amin(data[:, channel]))
		dabs = dmax if dmax>dmin else dmin
		m = np.mean(data[:, channel] / dmax, dtype='float64') * dmax
		s = np.std(data[:, channel] / dmax, dtype='float64') * dmax
		s = 1 if s == 0 else s # If std == 0, change it to 1, so data-mean = 0
		data[:, channel] = ((data[:, channel] - m) / s).astype('float16')
	
	return data

def downsample_data(data, fs, downsampled_fs):

    # Dowsampling
	if fs > downsampled_fs:
		downsampled_pts = np.linspace(0, data.shape[0]-1, int(np.round(data.shape[0]/fs*downsampled_fs))).astype(int)
		downsampled_data = data[downsampled_pts, :]

    # Upsampling
	elif fs < downsampled_fs:
		print("Original sampling rate below downsample frequency!")
		return None

    # Change from int16 to float16 if necessary
    # int16 ranges from -32,768 to 32,767
    # float16 has ±65,504, with precision up to 0.0000000596046
	if downsampled_data.dtype != 'float16':
		downsampled_data = np.array(downsampled_data, dtype="float16")

	return downsampled_data
