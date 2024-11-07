import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def apply_smoothing(data, wavelengths, window_length=11, polyorder=2, output_dir="smoothing_plots"):
    """
    Apply Savitzky-Golay smoothing to the data and save plots.

    Parameters:
    - data: DataFrame of absorbance data.
    - wavelengths: Array of wavelength values.
    - window_length: Window length for smoothing.
    - polyorder: Polynomial order for smoothing.
    - output_dir: Directory to save the plots.

    Returns:
    - smoothed_data: DataFrame of smoothed data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    smoothed_data = data.apply(lambda x: savgol_filter(x, window_length, polyorder), axis=0)

    for i in range(data.shape[1]):
        if i % 100 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, data.iloc[:, i], label='Original Spectrum')
            plt.plot(wavelengths, smoothed_data.iloc[:, i], label='Smoothed Spectrum')
            plt.xlabel('Wavelength')
            plt.ylabel('Absorbance')
            plt.title(f'Smoothing - Spectrum {i + 1}')
            plt.legend()
            plt.savefig(f"{output_dir}/spectrum_{i + 1}.png")
            plt.close()
            print(f"Plot saved: {output_dir}/spectrum_{i + 1}.png")

    return smoothed_data