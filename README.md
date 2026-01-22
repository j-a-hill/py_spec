Python code for tidying and plotting spectroscopy data, particular time-resolved measurements.

Expects data in text formats with whitespaces e.g. .asc  

Run using :

>python spec_main.py -i YOUR_DATA

Accepts multiple arguments and wildcards, can be found using -h 

Arguments and defaults:

-i  --input, Path to the input files (use wildcard for multiple files)

-b  --background,   Path to the background spectrum file for subtraction

-H  --header,       Number of header lines in the file,  default=0

-f  --footer,       Number of footer lines in the file,  default=0

-o  --output, Path to save the processed output files, default="pyspec.csv"

-t  --time, Time for each spectra in S

-bl --baseline, Enable baseline correction, (need to add options, but can change in baseline_correction.py)

-sm --smooth, Enable Savitzky-Golay smoothing, (need to add options, but can change in smoothing.py)

-w  --wavelengths, List of wavelengths to plot over time

-st --Spectra_time, Plot every nth spectrum over time, default=10

## Analysis/fitting can be done with Python scripts or Jupyter notebooks

### Python Scripts (Command Line)
- `spec_analysis.py` - Spectral fitting with multiple Gaussian models
- `spec_time_analysis.py` - Time-series exponential decay fitting
- `dose_analysis_no_fit.py` - Multi-dose comparison plotting

These don't have any fancy options but most parameters are obvious in the code

### Jupyter Notebooks (Interactive Analysis)
For easier interactive analysis and visualization, use the Jupyter notebook versions:
- `spec_analysis.ipynb` - Interactive spectral fitting with Gaussian models and baseline
- `spec_time_analysis.ipynb` - Interactive time-series analysis with exponential fitting
- `dose_analysis_no_fit.ipynb` - Interactive multi-dose comparison and plotting

The notebooks include:
- Step-by-step execution with markdown documentation
- Inline visualizations and plots
- Parameter extraction and summary statistics
- Easy modification of analysis parameters
