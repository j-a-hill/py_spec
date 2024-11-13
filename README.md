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

## Analysis/fitting can be done with spec_analysis.py and spec_time_analysis.py 

These don't have any fancy options but most parameters are obvious in the code
