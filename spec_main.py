import os
import argparse
import glob
import pandas as pd
from background_subtraction import subtract_background_and_save
from spec_import import load_absorbance_data
from baseline_correction import apply_baseline_correction
from smoothing import apply_smoothing
from wavelength_time import plot_wavelengths_over_time
from time_spec import plot_spectra_over_time

# Argument parsing
parser = argparse.ArgumentParser(description="Spectral data import and processing")
parser.add_argument('-i', '--input', help="Path to the input files (use wildcard for multiple files)", type=str, required=True, nargs='+')
parser.add_argument('-b', '--background', help="Path to the background spectrum file for subtraction", type=str)
parser.add_argument('-H', '--header', help="Number of header lines in the file", type=int, default=0)
parser.add_argument('-f', '--footer', help="Number of footer lines in the file", type=int, default=0)
parser.add_argument('-o', '--output', help="Path to save the processed output file", type=str, default="pyspec.csv")
parser.add_argument('-t', '--time', help='Time for each spectra in S', type=float)
parser.add_argument('--baseline', '-bl', help="Enable baseline correction", action='store_true')
parser.add_argument('--smooth', '-sm', help="Enable Savitzky-Golay smoothing", action='store_true')
parser.add_argument('--wavelengths', '-w', help="List of wavelengths to plot over time", type=int, nargs='+')
parser.add_argument('--Spectra_time', '-st', help="Plot every nth spectrum over time", type=int, default=10)

args = parser.parse_args()

if args.time is None:
    print("Warning: No time interval specified. Defaulting to 0.1s per spectrum.")
    args.time = 0.1  # Default to 0.1 seconds

# Expand wildcard paths into actual file paths
file_paths = [file for input_path in args.input for file in glob.glob(input_path)]
if not file_paths:
    print("No input files found.")
else:
    print(f"Input file paths found: {file_paths}")

all_data = []
for file_path in file_paths:
    print(f"Loading data from: {file_path}")
    df = load_absorbance_data(file_path, args.header, args.footer, time_point_interval=args.time)

    if df.empty:
        print(f"Warning: Empty data from {file_path}.")
    else:
        all_data.append(df)

# Check if any data was loaded
if all_data:
    combined_df = pd.concat(all_data, axis=0)
    print(f"Combined DataFrame shape: {combined_df.shape}")
else:
    combined_df = pd.DataFrame()  # Empty DataFrame if no files loaded
    print("No data to process.")

# Calculate the average across replicates (time columns) by grouping by 'Wavelength'
if not combined_df.empty:
    mean_df = combined_df.groupby('Wavelength').mean()
    mean_df.to_csv(args.output.replace('.csv', '_mean.csv'))
    print("Mean data calculated and saved.")
else:
    mean_df = pd.DataFrame()
    print("No mean data calculated due to empty combined DataFrame.")

# Background subtraction, if applicable
if args.background and not mean_df.empty:
    print("Performing background subtraction...")
    mean_df = subtract_background_and_save(mean_df, args.background, output_file=f"background_subtracted_{args.output}")
    mean_df.to_csv(args.output.replace('.csv', '_background_subtracted.csv'))

# Extract wavelengths from the first column of mean_df
if not mean_df.empty:
    wavelengths = mean_df.index.values

# Apply baseline correction if enabled
if args.baseline and not mean_df.empty:
    print("Applying baseline correction...")
    mean_df = apply_baseline_correction(mean_df, wavelengths)
    mean_df.to_csv(args.output.replace('.csv', '_baseline_corrected.csv'))

# Apply smoothing if enabled
if args.smooth and not mean_df.empty:
    print("Applying smoothing...")
    mean_df = apply_smoothing(mean_df, wavelengths)
    mean_df.to_csv(args.output.replace('.csv', '_smoothed.csv'))

# Plot specified wavelengths over time if enabled
if args.wavelengths and not mean_df.empty:
    print("Plotting specified wavelengths over time...")
    plot_wavelengths_over_time(mean_df, wavelengths, args.wavelengths)

# Plot spectra over time if enabled
if args.Spectra_time and not mean_df.empty:
    print("Plotting spectra over time...")
    plot_spectra_over_time(mean_df, wavelengths, n=args.Spectra_time)

# Save the final processed data
if not mean_df.empty:
    mean_df.to_csv(args.output)
    print(f"Processed data saved to {args.output}")
else:
    print("No data saved due to empty DataFrame.")