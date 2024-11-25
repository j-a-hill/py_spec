import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import StepModel, LinearModel
import glob

# Set the backend to 'Agg' for non-interactive plotting
#plt.switch_backend('Agg')

# Load multiple CSV files
file_paths = glob.glob('C:/Users/jake_/Desktop/HGD_nanospec/DTNB_*/*/final_pyspec.csv')
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

# Fit a model to each dose individually
for i, df in enumerate(data_frames):
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

                # Create Step model with S-curve
                step_mod = StepModel(form='erf', prefix='step_')
                linear_mod = LinearModel(prefix='line_')

                # Combine the models
                model = step_mod + linear_mod

                # Create parameters for the model using guess method
                params = model.make_params()
                params.update(step_mod.guess(absorbance_cut, x=time_cut, center=2))
                params['line_slope'].set(value=-0.1, min=0, max=0.1)
                params['line_intercept'].set(value=absorbance_cut.mean())

                # Fit the model to the data
                out = model.fit(absorbance_cut, params, x=time_cut)

                # Plot the data and the fit
                plt.scatter(time_cut, absorbance_cut, label=f'Data Dose {i+1}', s=5)
                plt.plot(time_cut, out.best_fit, label=f'Best fit Dose {i+1}')
                plt.show()

# Finalize the plot
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Absorbance')
plt.title('Absorbance vs Time for Different Doses')
#plt.savefig('all_doses_fit.png')
plt.show()