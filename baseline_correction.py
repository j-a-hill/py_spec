# baseline_correction.py
import subprocess
import sys
import importlib.util
import os

def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install():
    """Check if required libraries are installed, and install them if missing."""
    required_libraries = ['scipy', 'pybaselines']

    for library in required_libraries:
        if importlib.util.find_spec(library) is None:
            print(f"{library} not found. Installing...")
            install_package(library)
        else:
            print(f"{library} is already installed.")

# Check and install the required libraries before the rest of the script runs
check_and_install()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pybaselines.polynomial import imodpoly

def apply_baseline_correction(data, wavelengths, poly_order=4, tol=1e-3, num_std=1, output_dir="baseline_correction"):
    """
    Apply baseline correction to each spectrum using imodpoly.

    Parameters:
    - data: DataFrame or numpy array of absorbance data.
    - wavelengths: The wavelengths corresponding to the absorbance data.
    - poly_order: The order of the polynomial used for baseline fitting.
    - tol: Tolerance for the baseline fitting algorithm.
    - num_std: Number of standard deviations for the fitting.
    - output_dir: Directory to save the plots.

    Returns:
    - baseline_subtracted_data: Data after baseline subtraction.
    - baseline_corrected_data: The baseline that was fitted.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    baseline_subtracted_df = pd.DataFrame(index=data.index, columns=data.columns)

    for column_index in range(data.shape[1]):
        column_to_correct = data.iloc[:, column_index].values

        # Apply baseline correction using imodpoly
        baseline_corrected, params = imodpoly(
            column_to_correct,
            x_data=wavelengths,
            poly_order=poly_order,
            tol=tol,
            num_std=num_std,
            return_coef=True
        )

        # Subtract the baseline from the original data
        baseline_subtracted = column_to_correct - baseline_corrected

        # Save the baseline-subtracted data
        baseline_subtracted_df.iloc[:, column_index] = baseline_subtracted

        # Optionally plot the correction for every 100th spectrum
        if column_index % 100 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, column_to_correct, label='Original Spectrum')
            plt.plot(wavelengths, baseline_corrected, label='Fitted Baseline')
            plt.plot(wavelengths, baseline_subtracted, label='Baseline Subtracted')
            plt.xlabel('Wavelength')
            plt.ylabel('Absorbance')
            plt.title(f'Baseline Subtraction - Spectrum {column_index + 1}')
            plt.legend()

            # Save the plot as a PNG file
            plot_filename = f"{output_dir}/spectrum_{column_index + 1}.png"
            plt.savefig(plot_filename)
            plt.close()  # Close the plot to avoid displaying it in a non-interactive environment
            print(f"Plot saved: {plot_filename}")

    return baseline_subtracted_df