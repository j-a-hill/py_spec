import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Set the backend to 'Agg' for non-interactive plotting
#plt.switch_backend('Agg')

# Load multiple CSV files
file_paths = glob.glob('C:/Users/jake_/Desktop/HGD_nanospec/DTNB_Dose/final_pyspec_*.csv')
data_frames = [pd.read_csv(file, index_col=0) for file in file_paths]

# Extract specified wavelengths
selected_wavelengths = [412]  # Example wavelengths
extracted_data = []
cut_timepoint = 20

for df in data_frames:
    wavelengths = df.index.values.astype(float)
    time = df.columns.str.replace('s', '').astype(float)
    extracted_row = []
    for selected_wavelength in selected_wavelengths:
        closest_wavelength = min(wavelengths, key=lambda x: abs(x - selected_wavelength))
        if closest_wavelength in wavelengths:
            absorbance = df.loc[closest_wavelength].values

            # Ensure time and absorbance arrays are of the same length
            if len(time) == len(absorbance):
                # Cut the data at the specified time point
                mask = time <= cut_timepoint
                time_cut = time[mask]
                absorbance_cut = absorbance[mask]

                extracted_row.append(absorbance_cut)
    extracted_data.append(extracted_row)

# Write extracted data to a new CSV file
extracted_df = pd.DataFrame(extracted_data, columns=[f'{wavelength} nm' for wavelength in selected_wavelengths])
extracted_df.to_csv('extracted_wavelengths.csv', index=False)

# Plot the data for each dose individually
for i, (df, file_path) in enumerate(zip(data_frames, file_paths)):
    dose_label = os.path.basename(file_path).split('_')[-1].split('.')[0]
    for selected_wavelength in selected_wavelengths:
        closest_wavelength = min(wavelengths, key=lambda x: abs(x - selected_wavelength))
        if closest_wavelength in wavelengths:
            absorbance = df.loc[closest_wavelength].values

            # Ensure time and absorbance arrays are of the same length
            if len(time) == len(absorbance):
                # Cut the data at the specified time point
                mask = time <= cut_timepoint
                time_cut = time[mask]
                absorbance_cut = absorbance[mask]

                # Plot the data
                plt.scatter(time_cut, absorbance_cut, label=f'Data Dose {dose_label}', s=5)

# Finalize the plot
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Absorbance')
plt.title('Absorbance vs Time for Different Doses')
#plt.savefig('all_doses_fit.png')
plt.show()