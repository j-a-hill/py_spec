# py_spec - Spectroscopy Data Analysis

Python code for tidying and plotting spectroscopy data, particularly time-resolved measurements.

## Supported Data Formats

The package now supports multiple data formats:

### Legacy Format
- Text formats with whitespace delimiters (.asc, .txt)
- First column: wavelengths
- Subsequent columns: absorbance values at different time points
- No header row

### Enhanced Format (New!)
The enhanced import functions can now handle:
- **CSV, Excel, TSV files** with headers
- **Flexible column naming** (case-insensitive detection):
  - Wavelength columns: 'wavelength', 'lambda', 'wl', 'nm'
  - Wavenumber columns: 'wavenumber', 'cm-1', 'cm^-1'
  - Measurement columns: 'absorbance', 'transmission', 'fluorescence', 'abs', 'od', 'fluor', etc.
  - Time columns: 'time', 'timestamp'
  - Metadata columns: 'concentration', 'sample', 'power', etc.
- **Both wide and tidy data formats**:
  - Wide format: wavelengths as rows, time points as columns
  - Tidy format: one observation per row with wavelength, time, and measurement columns
- **Automatic conversion** to tidy format for consistent analysis

### Using Enhanced Import Functions

```python
from spec_import import load_tidy_data, load_spectroscopy_data

# Load data with automatic format detection and conversion to tidy format
df = load_tidy_data('your_data.csv')

# Load data with specific column mapping
df = load_tidy_data('your_data.csv', 
                    wavelength_col='nm', 
                    time_interval=0.1)  # 0.1 second intervals

# Use the unified loader (backward compatible)
df = load_spectroscopy_data('your_data.asc', time_point_interval=0.1)
```

## Command Line Usage

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

## Data Format Examples

### Wide Format CSV
```csv
Wavelength,0.0s,0.1s,0.2s,0.3s
400,0.123,0.145,0.167,0.189
410,0.234,0.256,0.278,0.290
420,0.345,0.367,0.389,0.401
```

### Tidy Format CSV
```csv
Wavelength,Time,Absorbance,Sample,Concentration
400,0.0,0.123,Sample1,1.0
400,0.1,0.145,Sample1,1.0
410,0.0,0.234,Sample1,1.0
410,0.1,0.256,Sample1,1.0
```

### Mixed Column Names (Auto-detected)
```csv
nm,Transmission_0ms,Transmission_100ms,Sample_ID,Power
400,0.823,0.745,S1,100
410,0.734,0.656,S1,100
```

All formats are automatically converted to a consistent tidy format for analysis.
