import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import StepModel, LinearModel

# Set the backend to 'Agg' for non-interactive plotting
plt.switch_backend('Agg')

# Load the data
data = pd.read_csv('C:/Users/jake_/Desktop/Python_spec/pyspec.csv', index_col=0)
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

    # Create Step model with S-curve
    step_mod = StepModel(form='erf', prefix='step_')
    linear_mod = LinearModel(prefix='line_')

    # Combine the models
    model = step_mod + linear_mod

    # Create parameters for the model using guess method
    params = model.make_params()
    params.update(step_mod.guess(Absorbance, x=Time, center=2))
    params['line_slope'].set(value=-0.1, min=0, max=0.1)
    params['line_intercept'].set(value=Absorbance.mean())

    # Fit the model to the data
    out = model.fit(Absorbance, params, x=Time)

    # Write the fit report to a log file
    with open('time_analysis_fit_report.log', 'a') as f:
        f.write(f'\nSpectra wavelength: {closest_wavelength}\n')
        f.write(out.fit_report(min_correl=0.3))

    # Plot the data and the fit
    plt.scatter(Time, Absorbance, label='Data', s=5)
    plt.plot(Time, out.init_fit, label='Initial fit')
    plt.plot(Time, out.best_fit, label='Best fit')

    # Plot the individual model components
    components = out.eval_components(x=Time)
    plt.plot(Time, components['step_'], label='Step component')
    plt.plot(Time, components['line_'], label='Linear component')

    plt.legend()
    plt.title(f'Fit at {closest_wavelength} nm')
    plt.xlabel('Time (s)')
    plt.ylabel('Absorbance')
    plt.savefig(f'fit_{closest_wavelength}_nm.png')
    plt.close()
else:
    print(f'Wavelength {closest_wavelength} not found in the data.')