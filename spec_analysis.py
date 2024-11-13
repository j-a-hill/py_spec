import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit.models import GaussianModel, SplineModel

# Load the processed data from spec_main.py
data = pd.read_csv('C:/Users/jake_/Desktop/Python_spec/pyspec.csv', index_col=0)
print(data)

# Loop through each time point and perform the fitting
for i, time_point in enumerate(data.columns):
    y = data[time_point].values  # Absorbance values at the current time point
    x = data.index.values.astype(float)  # Wavelength values

    # First Gaussian model
    model1 = GaussianModel(prefix='peak1_')
    params1 = model1.make_params(amplitude=dict(value=0.1, min=0, max=0.5),
                                 center=dict(value=300, min=250, max=350),
                                 sigma=dict(value=3, min=0))

    # Second Gaussian model
    model2 = GaussianModel(prefix='peak2_')
    params2 = model2.make_params(amplitude=dict(value=0.1, min=0, max=0.5),
                                 center=dict(value=350, min=300, max=450),
                                 sigma=dict(value=3, min=0))

    # Combine the two Gaussian models
    model = model1 + model2
    params = params1 + params2

    # Define knot positions for the background spline
    # Either use a list of positions in np.array([])
    # Or use np.concatenate(np.arange(start, end, step), np.arange(start, end, step)) to excluded the peak regions
    knot_xvals = np.concatenate((np.arange(180, 250, 20), np.arange(400, 700, 20)))

    bkg = SplineModel(prefix='bkg_', xknots=knot_xvals)
    params.update(bkg.guess(y, x))

    model = model + bkg

    # Fit the model to the data
    out = model.fit(y, params, x=x)
    comps = out.eval_components()

    # Write the fit report to a log file
    with open('fit_report.log', 'a') as f:
        f.write(f'\nSpectra time_point: {time_point}\n')
        f.write(out.fit_report(min_correl=0.3))

    # Plot the 1st, 10th, and 100th spectrum for checking
    if i in [0, 10, 100]:
        plt.figure()
        plt.plot(x, y, label=f'data at {time_point}')
        plt.plot(x, out.best_fit, label='best fit')
        plt.plot(x, comps['bkg_'], label='background')
        plt.plot(x, comps['peak1_'], label='peak1')
        plt.plot(x, comps['peak2_'], label='peak2')
        plt.plot(x, model.eval(params, x=x), label='initial')
        plt.legend()
        plt.title(f'Spectrum at {time_point}')
        plt.show()