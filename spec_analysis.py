import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

# Set the backend to 'Agg' for non-interactive plotting
# plt.switch_backend('Agg')

# Load the processed data from spec_main.py
data = pd.read_csv('C:/Users/jake_/Desktop/HGD_nanospec/DTNB_5/hgd_R37S_DTNB_1_5_transmission.asc_pyspec/final_pyspec.csv', index_col=0)
print(data)

# Filter out wavelengths below 250
data = data[data.index.astype(float) >= 280]

# Initialize variables to store the fitted parameters
initial_params = None

# Loop through each time point and perform the fitting
for i, time_point in enumerate(data.columns):
    y = data[time_point].values  # Absorbance values at the current time point
    x = data.index.values.astype(float)  # Wavelength values

    # First Gaussian model
    model1 = GaussianModel(prefix='peak1_')
    params1 = model1.make_params(amplitude=dict(value=0.2, min=0),
                                 center=dict(value=330),
                                 sigma=dict(value=1))
    # Second Gaussian model
    model2 = GaussianModel(prefix='peak2_')
    params2 = model2.make_params(amplitude=dict(value=0.2, min=0),
                                 center=dict(value=412, min=410, max=420),
                                 sigma=dict(value=1))
    # Linear model for baseline
    model3 = LinearModel(prefix='base_')
    params3 = model3.make_params(slope=dict(value=0),
                                 intercept=dict(value=0))

    model4 = GaussianModel(prefix='peak3_')
    params4 = model4.make_params(amplitude=dict(value=0.2, min=0),
                                 center=dict(value=290, min=280, max=300),
                                 sigma=dict(value=1, min=0.1, max=1.5))

    # Combine the models and parameters
    model = model1 + model2 + model3 + model4
    params = params1 + params2 + params3 + params4

    if initial_params is not None:
        # Update parameters with tight constraints except for amplitude
        for name, param in initial_params.items():
            if 'amplitude' not in name:
                params[name].set(value=param.value, min=param.value * 0.95, max=param.value * 1.05)
            else:
                params[name].set(value=param.value, min=0)

    # Fit the model to the data
    out = model.fit(y, params, x=x, method='leastsq', max_nfev=10000)
    comps = out.eval_components()

    if initial_params is None:
        # Store the initial fitted parameters
        initial_params = out.params

    # Write the fit report to a log file
    with open('fit_report.log', 'a') as f:
        f.write(f'\nSpectra time_point: {time_point}\n')
        f.write(out.fit_report(min_correl=0.3))

    # Plot every 100th spectrum for checking
    if i < 1:
        plt.figure()
        plt.scatter(x, y, label=f'data at {time_point}', s=5)
        plt.plot(x, out.best_fit, label='best fit', color='red')
        plt.plot(x, comps['peak1_'], label='peak1')
        plt.plot(x, comps['peak2_'], label='peak2')
        plt.plot(x, comps['peak3_'], label='peak3')
        plt.plot(x, comps['base_'], label='baseline')
        #plt.plot(x, out.init_fit, label='initial fit', linestyle='--')
        plt.legend()
        plt.title(f'Spectrum at {time_point}')
        plt.show()