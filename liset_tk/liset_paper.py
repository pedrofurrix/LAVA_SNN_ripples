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

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/'))
sys.path.insert(0,utils_path)

ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(ROOT_DIR)
if not ROOT_DIR in sys.path:
    sys.path.insert(0, ROOT_DIR)
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from copy import deepcopy
from liset_tk.format_predictions import get_predictions_indexes
from liset_tk.liset_aux import *
from liset_tk.load_data import *
from liset_tk.signal_aid import *
import liset_tk.lists_sessions as lists_sessions
from liset_tk.gt_annotations import get_ripple_events,configs



class liset_paper():
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
     
    def __init__(self, data_path, shank, downsample = False, normalize = True, numSamples = False, start = 0, verbose=True, original_fs=30000,load_data=True,channels=None,scale_data=False,annotation_type=None):

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

            self.fs=self.original_fs if not downsample else self.downsampled_fs

        # Initialize class variables.
        self.channels=channels
        self.scale_data=scale_data
        self.annotation_type=annotation_type    

        # Load the data.
        if load_data:
            self.load(data_path, shank, downsample = downsample, normalize=normalize, channels=channels)
        else:
            self.file_samples = self.get_file_samples(data_path) # get file samples without loading data
        
        # Try to load the ripples if the path contain a file with ripple times.
        # Only load ripples in the interval selected of data (chunk)
        if hasattr(self, 'fs'):
            self.ripples_GT = self.ripples_in_chunk(load_ripple_times_paper(data_path), start, numSamples, self.fs, self.fs_conv_fact)
        
            if type(self.ripples_GT) is not np.ndarray:
                self.has_ripples = False
                self.has_ripplesGT = False
            else:
                self.has_ripples = True
                self.has_ripplesGT = True
                self.num_ripples = len(self.ripples_GT)
                self.ripples_GT = ((self.ripples_GT- start) / self.fs_conv_fact).astype(int)
                self.get_gt_annotations()

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
        
    def get_gt_annotations(self,):
        self.manual_GT=self.ripples_GT
        if self.annotation_type is not None:
            if self.annotation_type=="manual_GT":
                return self.ripples_GT # Samples
            else:
                channel=0
                if not hasattr(self, 'data'):
                    self.load(self.data_path, shank=None, downsample = False, normalize=self.normalize,invert=False, channels=[channel])
                info=get_ripple_events(self.data[:,channel],self.fs,config=self.annotation_type)
                #get start and end times from annnotations...
                ripple_idx = info[["start_idx","end_idx"]].to_numpy()
                self.ripples_GT=ripple_idx    
    
    def ripples_in_chunk(self, ripples, start, numSamples, fs, prop):
        if not numSamples:
            numSamples = self.file_samples-start

        in_chunk = ripples[(ripples[:,0] > start) & (ripples[:,0] < (start + numSamples))]

        return in_chunk

    def get_file_samples(self, path): # added method for getting total samples in .dat file without loading all data
        """
        Get the total number of samples in the .dat file.

        Parameters:
        - path (str): Path to the directory containing the .dat file.

        Returns:
        - file_samples (int): Total number of samples in the .dat file.
        """
        try:
            filename = f"{path}/{[i for i in os.listdir(path) if i.endswith('.dat')][0]}"
            file_len = os.path.getsize(filename=filename)
            # print(filename)
            num_channels_raw = 8
            file_samples = file_len / num_channels_raw / 2
            if self.verbose:
                print(f'Total file samples: {file_samples}')
            return file_samples    
        except:
            if self.verbose:
                print('.dat file not in path')
            return False

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
            num_channels_raw = 8
            self.file_samples = self.file_len // (num_channels_raw * 2)
        except:
            if self.verbose:
                print('.dat file not in path')
            return False
        
        nChannels = len(channels)
        if (len(channels) > nChannels):
            if self.verbose:
                print("Cannot load specified channels (listed channel IDs inconsistent with total number of channels).")
            return False
        
        start = self.start * num_channels_raw * 2
        numSamples = self.numSamples * num_channels_raw * 2
        
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
            data = RAW2ORDERED(data, channels,num_channels_raw=num_channels_raw)
            return data
            

    def load(self, data_path, shank, downsample, normalize, channels=None):
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

        try:
            try:
                info = loadmat(f'{data_path}/info.mat')
            except:
                info = loadmat(f'{data_path}/neurospark.mat')    
        except:
            print('.mat file cannot be opened or is not in path.')
            info=None
        
        # self.second_shank_channels=[29, 17, 20, 32, 30, 18, 19, 31]
        try:
            if channels is not None:
                channels=channels
            else:
                if info is not None:
                    if "neurosparkmat" in info:
                        channels = info['neurosparkmat']['channels'][0][0][shank-1:shank]
                    else: 
                        channels=[i for i in range(8)]
                # else:
                #     channels=self.second_shank_channels
        except Exception as err:
            print(f'No data available for shank {shank}\n\n{err}')
            return 
        
        raw_data = self.load_dat(data_path, channels, numSamples=self.numSamples)

        if hasattr(raw_data, 'shape'):
            self.data = self.clean(raw_data, downsample, normalize)
            self.duration = self.data.shape[0]/self.fs


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

     
        if self.scale_data and not normalize:
            self.bit_uvolts=0.1949999928 # uV per bit for Intan RHD2000 series
            self.ttl_bit_volts=0.0001525879 # Volts per bit for Intan RHD2000 series TTL channels
            data = data[:,:8] * self.bit_uvolts *1e-6 # Convert to volts
            # if self.channel_num==40 and 32 in self.channels:
            #     data= data[:,32:]* self.ttl_bit_volts  # Convert TTL channel to volts
        return data
    
    def plot_visualize(self, ch=None, offset=0, filtered=None, extend=0.5, title='Overview', window=None):
        """
        Plot data with overlays for:
        - Ground truth ripples (yellow)
        - Model predicted ripples (blue)
        - Light stimulation TTLs (red)

        Parameters
        ----------
        ch : int or list of ints, optional
            Channel(s) to plot. If None, plots the first channel.
        offset : float, optional
            Vertical offset between channels.
        filtered : tuple (low, high), optional
            Bandpass filter range (Hz).
        extend : float, optional
            Extra context (in seconds) around each event.
        title : str, optional
            Title for the plot.
        window : tuple (start, end), optional
            Time window in seconds to plot (e.g., (10, 20)).
        """

        # ---------------------
        # Select channels
        # ---------------------
        if ch is None:
            ch = [0]
        elif isinstance(ch, int):
            ch = [ch]

        n_samples = self.data.shape[0]
        time = np.arange(n_samples) / self.fs

        # ---------------------
        # Apply time window
        # ---------------------
        if window is not None:
            start_s, end_s = window
            start_idx = int(start_s * self.fs)
            end_idx = int(end_s * self.fs)
            time = time[start_idx:end_idx]
            data_slice = self.data[start_idx:end_idx, :]
        else:
            data_slice = self.data

        fig, ax = plt.subplots(figsize=(15, 6))
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (normalized units)")

        # ---------------------
        # Plot data (optionally filtered)
        # ---------------------
        for i, ch_idx in enumerate(ch):
            sig = deepcopy(data_slice[:, ch_idx])
            if filtered:
                from signal_aid import bandpass_filter
                sig = bandpass_filter(sig, filtered, self.fs)

            ax.plot(time, sig + i * offset, color="black", lw=0.8, label=f"Ch {ch_idx}" if i == 0 else "")

        min_y, max_y = ax.get_ylim()

        # ---------------------
        # Helper function to plot intervals safely within window
        # ---------------------
        def plot_intervals(intervals, color, label):
            if window is not None:
                mask = (intervals[:, 1] / self.fs > start_s) & (intervals[:, 0] / self.fs < end_s)
                intervals = intervals[mask]
            for i, r in enumerate(intervals):
                ax.fill_between(r / self.fs, min_y, max_y, color=color, alpha=0.3, label=label if i == 0 else "")

        # ---------------------
        # Ground truth ripples (yellow)
        # ---------------------
        if hasattr(self, "ripples_GT") and self.ripples_GT is not None:
            plot_intervals(self.ripples_GT, "yellow", "Ground truth")
        # ---------------------
        # Beautify
        # ---------------------
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

     

if __name__ == "__main__":
    data_path=r"E:\NCN\neurospark_mat\Download_from_paper"
    dataset="Som2_2019-07-24_12-01-49"
    DATA_PATH=os.path.join(data_path,dataset)
    liset=liset_paper(DATA_PATH,channels=[1],shank=1,downsample=4000)
    liset.plot_visualize(0,filtered=(100,250))