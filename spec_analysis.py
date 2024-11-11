#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:24:07 2024

@author: jake
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define a function for the curve fitting
def curve_func(x, a, b, c):
    return a * np.exp(-b * x) + c

data = pd.read_csv("/home/jake/Desktop/Python_spec/pyspec.csv", index_col=0)
print(data)

# Strip units and convert to float
Time = data.columns.str.replace('s', '').astype(float)
wavelengths = data.index.values.astype(float)

# Specify the wavelengths you want to plot
selected_wavelengths = [412]  # example wavelengths, change as needed

# Function to find the closest wavelength
def find_closest_wavelength(selected_wavelength, available_wavelengths):
    return min(available_wavelengths, key=lambda x: abs(x - selected_wavelength))

# Find the closest recorded wavelengths
closest_wavelengths = [find_closest_wavelength(w, wavelengths) for w in selected_wavelengths]

plt.figure()

# Loop through each selected wavelength and plot
for wavelength in closest_wavelengths:
    if wavelength in wavelengths:
        Absorbance = data.loc[wavelength].values
        
        # Scatter plot
        plt.scatter(Time, Absorbance, label=f'Wavelength {wavelength}')
        
        # Fit the curve with initial parameter guesses and increased maxfev
        initial_guesses = [1, 1e-3, 1]
        popt, _ = curve_fit(curve_func, Time, Absorbance, p0=initial_guesses, maxfev=10000)
        fitted_curve = curve_func(Time, *popt)
        
        # Plot the fitted curve
        plt.plot(Time, fitted_curve, label=f'Fitted Curve {wavelength}')
    else:
        print(f'Wavelength {wavelength} not found in the data.')

plt.xlabel('Time')
plt.ylabel('Absorbance')
plt.legend()
plt.show()
