from scipy.signal import savgol_filter
import pandas as pd
from scipy.signal import savgol_filter
from astropy.timeseries import BoxLeastSquares
import torch
import numpy as np

def detrend_and_smooth(fluxs, window_length=113, polyorder=2):
    smoothed = savgol_filter(fluxs, window_length=window_length, polyorder=polyorder)
    detrended = fluxs / smoothed - 1.0
    return detrended

def to_bls2D(times, fluxs, durations=[0.05, 0.2]):
    bls = BoxLeastSquares(times, fluxs)
    results = bls.autopower(duration=durations)
    results = pd.DataFrame(results).sort_values('power', ascending=False).head(128*128)
    results.drop('objective', axis=1, inplace=True)
    results = torch.tensor(results.to_numpy())
    results = results.transpose(1, 0).reshape(8, 128, 128)
    return results

def to_fft(time, flux):
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]

    # 1️⃣ Chuẩn hóa flux
    flux = (flux - np.mean(flux)) / np.std(flux)

    # 2️⃣ Tính sampling rate
    dt = np.median(np.diff(time))  # khoảng cách trung bình giữa các điểm
    fs = 1.0 / dt  # tần số lấy mẫu (Hz)

    # 3️⃣ Biến đổi Fourier rời rạc
    fft_flux = np.fft.fft(flux)
    freqs = np.fft.fftfreq(len(flux), d=dt)

    # 4️⃣ Lấy phần nửa phổ dương
    mask = freqs > 0
    freqs = freqs[mask]
    amplitude = np.abs(fft_flux[mask])

    return freqs, amplitude



