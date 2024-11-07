# wavelength_time.py
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_wavelengths_over_time(data, wavelengths, specified_wavelengths, output_dir="wavelengths_time"):
    """
    Plot specified wavelengths over time.

    Parameters:
    - data: DataFrame of absorbance data.
    - wavelengths: Array of wavelength values.
    - specified_wavelengths: List of wavelengths to plot.
    - output_dir: Directory to save the plot.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    for wavelength in specified_wavelengths:
        # Find the nearest wavelength in the data
        nearest_wavelength = wavelengths[np.abs(wavelengths - wavelength).argmin()]
        plt.plot(data.columns, data.loc[nearest_wavelength], label=f'Wavelength {nearest_wavelength} nm')

    plt.xlabel('Time')
    plt.ylabel('Absorbance')
    plt.title('Specified Wavelengths Over Time')
    plt.legend()

    # Set x-ticks to show every 100th label
    plt.xticks(ticks=np.arange(0, len(data.columns), 100), labels=data.columns[::100])

    plot_filename = f"{output_dir}/wavelengths_time.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved: {plot_filename}")