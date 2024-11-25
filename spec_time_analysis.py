import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel

# Set the backend to 'Agg' for non-interactive plotting
plt.switch_backend('Agg')

# Load the data
data = pd.read_csv('C:/Users/jake_/Desktop/HGD_nanospec/DTNB_5/hgd_R37S_DTNB_1_5_transmission.asc_pyspec/final_pyspec.csv', index_col=0)
print(data)

# Strip units and convert to float
Time = data.columns.str.replace('s', '').astype(float)
wavelengths = data.index.values.astype(float)

# Specify the wavelength and time point you want to plot
selected_wavelength = 412
cut_timepoint = 20

# Function to find the closest wavelength
def find_closest_wavelength(selected_wavelength, available_wavelengths):
    return min(available_wavelengths, key=lambda x: abs(x - selected_wavelength))

# Find the closest recorded wavelength
closest_wavelength = find_closest_wavelength(selected_wavelength, wavelengths)

# Check if the closest wavelength is in the data
if closest_wavelength in wavelengths:
    Absorbance = data.loc[closest_wavelength].values

    # Cut the data at the specified time point
    mask = Time <= cut_timepoint
    Time = Time[mask]
    Absorbance = Absorbance[mask]

    # Define weights to reduce the influence of noisy data points
    weights = np.ones_like(Absorbance)
    weights[Time < 2] = 0  # Example: Assign lower weights to data points before 2 seconds

    # Create Exponential model
    exp_mod = ExponentialModel(prefix='exp_')

    # Create parameters for the model using guess method
    params = exp_mod.make_params(amplitude=0.1, decay=5)
    print(params)
    # Fit the model to the data with weights
    out = exp_mod.fit(Absorbance, params, x=Time, weights=weights)

    # Write the fit report to a log file
    with open('time_analysis_fit_report.log', 'a') as f:
        f.write(f'\nSpectra wavelength: {closest_wavelength}\n')
        f.write(out.fit_report(min_correl=0.3))

    # Plot the data and the fit
    plt.scatter(Time, Absorbance, label='Data', s=5)
    plt.plot(Time, out.init_fit, label='Initial fit')
    plt.plot(Time, out.best_fit, label='Best fit')

    # Plot the individual model component
    components = out.eval_components(x=Time)
    plt.plot(Time, components['exp_'], label='Exponential component')

    plt.legend()
    plt.title(f'Fit for wavelength {closest_wavelength} nm')
    plt.xlabel('Time (s)')
    plt.ylabel('Absorbance')
    plt.savefig(f'fit_{closest_wavelength}_nm.png')
    plt.close()

    plt.scatter(Time, Absorbance, s=5)
    plt.plot(Time, out.best_fit)
    plt.xlabel('Time (s)')
    plt.ylabel('Absorbance')
    plt.savefig(f'fit_{closest_wavelength}_nm_no_legend.png')
    plt.close()

else:
    print(f'Wavelength {closest_wavelength} not found in the data.')