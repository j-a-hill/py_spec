import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt

def load_background_spectrum(background_file, header_lines=0, footer_lines=0):
    """
    Function to load the background spectrum data, similar to the spectral data import process.
    
    - background_file: Path to the background data file.
    - header_lines: Number of header lines to skip.
    - footer_lines: Number of footer lines to skip.
    
    Returns:
    - background_data: A pandas DataFrame containing 'Wavelength' and 'Absorbance' columns.
    """
    # Read the entire background file, skipping header and footer
    all_data = []
    with open(background_file, 'r') as file:
        # Skip header lines
        for _ in range(header_lines):
            next(file)
        
        # Read lines and parse them until footer lines are encountered
        for line in file:
            # Stop reading after reaching the footer
            if footer_lines > 0 and line.strip() == "":
                footer_lines -= 1
                continue  # Skip empty lines (or footer content)
            if footer_lines == 0:
                # Parse the line (assuming it's whitespace-delimited)
                try:
                    wavelength, absorbance = map(float, line.split())
                    all_data.append([wavelength, absorbance])
                except ValueError:
                    continue  # Skip lines that don't parse correctly

    # Convert the collected data into a DataFrame
    background_data = pd.DataFrame(all_data, columns=['Wavelength', 'Absorbance'])

    return background_data


def subtract_background(mean_df, background_data):
    """
    Subtract the background absorbance from the measured data at each wavelength.
    
    Parameters:
    - mean_df: DataFrame containing the measured data, with 'Wavelength' as index and timepoints as columns.
    - background_data: DataFrame containing the background data (wavelength and absorbance).
    
    Returns:
    - DataFrame with background-subtracted data.
    """
    # Interpolate the background absorbance values at the wavelengths in the measured data
    background_interp = np.interp(mean_df.index, background_data['Wavelength'], background_data['Absorbance'])

    # Create a copy of the DataFrame to store the subtracted values
    mean_data_subtracted = mean_df.copy()

    # Subtract the background absorbance for each timepoint
    for timepoint in mean_df.columns:
        mean_data_subtracted[timepoint] = mean_df[timepoint] - background_interp
    
    return mean_data_subtracted

def plot_comparison(mean_df, mean_data_subtracted, timepoints_to_plot=[0, 10, 100], output_dir="background_subtraction_plots"):
    """
    Function to plot comparison of original and subtracted spectra at specific timepoints.
    The plots will be saved as PNG files in the specified output directory.
    
    - mean_df: The original (mean) DataFrame containing spectra.
    - mean_data_subtracted: The background-subtracted DataFrame.
    - timepoints_to_plot: List of timepoints (indices) to plot.
    - output_dir: Directory to save the plots.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through the specified timepoints and plot the spectra
    for idx in timepoints_to_plot:
        if idx >= len(mean_df.columns):
            continue  # Skip if the timepoint index is out of range

        timepoint = mean_df.columns[idx]
        plt.figure(figsize=(10, 6))
        
        # Plot the original spectrum
        plt.plot(mean_df.index, mean_df[timepoint].values, label='Original Spectrum')
        
        # Plot the subtracted spectrum
        plt.plot(mean_df.index, mean_data_subtracted[timepoint].values, label='Subtracted Spectrum')

        plt.xlabel('Wavelength')
        plt.ylabel('Absorbance')
        plt.title(f'Time {timepoint} - Original and Subtracted Spectra')
        plt.legend()

        # Save the plot as a PNG file
        plot_filename = f"{output_dir}/spectrum_{timepoint}.png"
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to avoid displaying it in a non-interactive environment
        print(f"Plot saved: {plot_filename}")

def subtract_background_and_save(mean_df, background_file, output_file='averaged_data_subtracted.csv', timepoints_to_plot=[0, 10, 100], output_dir="background_subtraction"):
    background_data = load_background_spectrum(background_file)
    mean_data_subtracted = subtract_background(mean_df, background_data)
    
    # Save the subtracted data to CSV
    mean_data_subtracted.to_csv(output_file, index=True)
    print(f"Subtracted data saved to {output_file}")
    
    # Plot comparison for specific timepoints and save the figures
    plot_comparison(mean_df, mean_data_subtracted, timepoints_to_plot, output_dir)
    
    return mean_data_subtracted
