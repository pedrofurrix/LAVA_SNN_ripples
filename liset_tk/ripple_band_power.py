from scipy.signal import butter, filtfilt, hilbert
import numpy as np
import os 
from liset_tk_extra import liset_tk_extra
def ripple_band_power_trace(signal, fs, bandpass=(100, 250), smooth_ms=10, log_power=False,zscore=False):
    """
    Compute continuous ripple-band power over time.

    Parameters
    ----------
    signal : np.ndarray
        1D LFP trace.
    fs : float
        Sampling frequency (Hz).
    ripple_band : tuple
        Ripple frequency range (e.g., (120, 250)).
    smooth_ms : float
        Window for moving average smoothing (in milliseconds).
    log_power : bool
        If True, return log10(power) instead of linear power.

    Returns
    -------
    power_trace : np.ndarray
        Ripple-band power time series (same length as signal).
    """

    # Bandpass filter
    if bandpass is not None:
        nyq = fs / 2
        b, a = butter(4, [bandpass[0]/nyq, bandpass[1]/nyq], btype='band')
        filtered = filtfilt(b, a, signal)
    else:
        filtered = signal
    # Hilbert transform to get analytic envelope
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    # Compute power
    power = envelope ** 2
    if log_power:
        power = np.log10(power + 1e-12)

        # 4 Smooth (optional)
    if smooth_ms > 0:
        win_samples = int(fs * smooth_ms / 1000)
        if win_samples > 1:
            kernel = np.ones(win_samples) / win_samples
            power = np.convolve(power, kernel, mode='same')


     # ----- Optional z-score normalization -----
    if zscore:
        mu = np.mean(power)
        sigma = np.std(power)
        if sigma > 0:
            power = (power - mu) / sigma
        else:
            power = power - mu   # avoid division by zero
            

    return power

def find_max_ripple_channel(data, fs, bandpass=(100, 250), smooth_ms=5,
                            log_power=False, zscore=True):
    """
    Find channel with highest average ripple-band power.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_samples, n_channels)
    fs : float
        Sampling frequency
    bandpass, smooth_ms, log_power, zscore
        Passed to ripple_band_power_trace()

    Returns
    -------
    best_ch : int
        Index of channel with highest ripple power
    power_per_channel : np.ndarray
        Mean ripple power for each channel
    """

    n_channels = data.shape[1]
    power_per_channel = np.zeros(n_channels)

    for ch in range(n_channels):
        trace = data[:, ch]
        power = ripple_band_power_trace(
            trace, fs,
            bandpass=bandpass,
            smooth_ms=smooth_ms,
            log_power=log_power,
            zscore=zscore
        )
        power_per_channel[ch] = np.mean(power)

    best_ch = np.argmax(power_per_channel)
    return best_ch, power_per_channel


if __name__ == "__main__":
    data_path=r"E:\extra_data"
    names=os.listdir(data_path)
    for name in names:
        print(f'Processing {name}...')
        data_loader=liset_tk_extra(data_path=data_path, name=name, downsample=1000, scale_data=False,normalize=False, verbose=True)
        fs=data_loader.downsampled_fs
        data=data_loader.data
        best_ch,power_per_channel=find_max_ripple_channel(data, fs, bandpass=(100, 250), smooth_ms=5,
                            log_power=False, zscore=True)
        print(f'Best channel for {name}: {best_ch} with mean ripple power {power_per_channel[best_ch]}')