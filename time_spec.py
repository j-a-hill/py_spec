# time_spec.py
import os
import matplotlib.pyplot as plt

def plot_spectra_over_time(data, wavelengths, n=args.Spectra_time, output_dir="spectra_time"):
    """
    Plot every nth spectrum on the same axis to show changes over time.

    Parameters:
    - data: DataFrame of absorbance data.
    - wavelengths: Array of wavelength values.
    - n: Interval for plotting spectra.
    - output_dir: Directory to save the plot.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    for i in range(0, data.shape[1], n):
        plt.plot(wavelengths, data.iloc[:, i], label=f'Time {data.columns[i]}')

    plt.xlabel('Wavelength')
    plt.ylabel('Absorbance')
    plt.title('')
    plt.legend()
    plot_filename = f"{output_dir}/spectra_time.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved: {plot_filename}")