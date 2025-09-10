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

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from copy import deepcopy
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(parent_dir)
# from format_predictions import get_predictions_indexes
from liset_tk.liset_aux import *
from liset_tk.load_data import *
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

        # Initialize class variables.
        self.prediction_times = []
        self.model = None
        self.default_channels = [16, 4, 1, 13, 15, 3, 2, 14]

        # Load the data.
        self.load(data_path, shank, downsample = downsample, normalize=normalize)


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
            self.file_samples = self.file_len / 43 / 2
        except:
            if self.verbose:
                print('.dat file not in path')
            return False
        
        nChannels = len(channels)
        if (len(channels) > nChannels):
            if self.verbose:
                print("Cannot load specified channels (listed channel IDs inconsistent with total number of channels).")
            return False
        
        start = self.start * 43 * 2
        numSamples = self.numSamples * 43 * 2

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
            data = RAW2ORDERED(data, channels)
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
        mat_files = [f for f in os.listdir(data_path) if f.endswith(".mat")]
        mat_file=mat_files[0]
        try:
            self.info = loadmat(f'{data_path}/{mat_file}',squeeze_me=True, struct_as_record=False)
        except:
            try:
                self.info = h5py.File(f'{data_path}/{mat_file}', 'r')
            except:
                print('.mat file cannot be opened or is not in path.')
                return
            
        try:
            channels = self.info['neurosparkmat']['channels'][0][0][8 * (shank -1):8 * shank]
        except Exception as err:
            try:
                channels =np.array(self.info['neurosparkmat']['channels'])
                channels = channels.flatten()
                channels = channels[8*(shank-1):8*shank]
            except Exception as err:  
                print(f'No data available for shank {shank}\n\n{err}')
                return 
        
        raw_data = self.load_dat(data_path, channels, numSamples=self.numSamples)

        if hasattr(raw_data, 'shape'):
            self.data = self.clean(raw_data, downsample, normalize)
            self.duration = self.data.shape[0]/self.fs

    def load_seizures(self):
        if hasattr(self,"fs"):
            self.seizures_GT = load_seizure_times(self.data_path, self.fs)
            if type(self.seizures_GT) is not np.ndarray:
                self.has_seizures = False
                self.has_seizuresGT = False
            else:
                self.has_seizures = True
                self.has_seizuresGT = True
                self.num_seizures = len(self.seizures_GT)
                self.seizures_GT = (self.seizures_GT * self.fs - self.start / self.fs_conv_fact).astype(int)

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
            self.IISs_times = self.info['neurosparkmat']['IISs']['times'][0][0]
            if type(self.IISs_times) is not np.ndarray:
                self.has_IISs = False
            else:
                self.has_IISs = True
                self.num_IISs = len(self.IISs_times)
                self.IISs_times = (self.IISs_times * self.fs - self.start / self.fs_conv_fact).astype(int)
        else:
            if self.verbose:
                print("Cannot load IISs times because the sampling frequency is not defined.")