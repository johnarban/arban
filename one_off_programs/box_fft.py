import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt


def signal_gen(time_step):
    np.random.seed(1234)

    period = 5.

    time_vec = np.arange(0, 20, time_step)
    sig = (np.sin(2 * np.pi / period * time_vec)
        + 0.5 * np.random.randn(time_vec.size))

    # plt.figure(figsize=(6, 5))
    # plt.plot(time_vec, sig, label='Original signal')

    return sig,time_vec


def transform(sig, time_step):
    """discrete fourier transform

    Parameters
    ----------
    sig : numpy array
        regularly spaced signal
    time_step : numpy array
        time step of data

    Returns
    -------
    tuple(numpy array, numpy array)
        fft of signal, sample frequency from fft
    """
    sig_fft = fftpack.fft(sig)
    sample_freq = fftpack.fftfreq(sig.size, d=time_step)  #[1/seconds, Hz]
    return sig_fft, sample_freq


def cutoff_filter(sig, time_step, cut_freq, high_pass=True):
    """apply cuttoff filter, either high pass or low pass

    Parameters
    ----------
    sig : nd.array
        regularly spaced signal
    time_step : float
        time step for signal
    peak_freq : cuttoff frequency
        in 1/(units of time step)
    high_pass : bool, optional
        if True (default), high pass filter (cut frequences
        less than cut_freq)
    Returns
    -------
    nd.array
        the filtered signal
    """
    sig_fft,sample_freq = transform(sig, time_step)

    if high_pass:
        low_freq_fft = sig_fft.copy()
        low_freq_fft[np.abs(sample_freq) <= cut_freq] = 0
        filtered_sig = fftpack.ifft(low_freq_fft)
    else:
        high_freq_fft = sig_fft.copy()
        high_freq_fft[np.abs(sample_freq) >= cut_freq] = 0
        filtered_sig = fftpack.ifft(high_freq_fft)

    return filtered_sig


# example
time_step = 0.02 # [seconds]
sig, time_vec = signal_gen(time_step)
filtered_sig = cutoff_filter(sig, time_step, 0.3)
filtered_sig = cutoff_filter(sig, time_step, 0.3,high_pass=False)