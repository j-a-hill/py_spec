# Spectral Analysis Update Summary

## Changes Made

The `labeled_spec_analysis.ipynb` notebook has been updated to handle all spectral data from both labeled and unlabeled samples across all X-ray transmission levels.

### 1. **Data Loading** 
- Updated to load all CSV files from both directories:
  - **DTNB_Dose/** (labeled samples): 4 files at different X-ray transmission levels (5%, 25%, 50%, 100%)
  - **R37S_dose/** (unlabeled samples): 5 files at different X-ray transmission levels (5%, 10%, 25%, 50%, 100%)
- Total: **9 datasets** loaded automatically
- Each dataset structure retains transmission level information for organization

### 2. **Output Directory Structure**
Created organized output folders to prevent overwhelming displays:
```
spectral_analysis_output/
├── DTNB_Dose_labeled/
│   ├── DTNB_5%/
│   ├── DTNB_25%/
│   ├── DTNB_50%/
│   └── DTNB_100%/
└── R37S_dose_unlabeled/
    ├── R37S_5%/
    ├── R37S_10%/
    ├── R37S_25%/
    ├── R37S_50%/
    └── R37S_100%/
```

### 3. **Plot Export Instead of Display**
All visualization cells have been modified to:
- **Save plots as PNG files** (150 dpi) instead of displaying inline
- **Eliminate clutter** from having 9 datasets × multiple plots per dataset
- **Organize results** by dataset in corresponding output folders

### 4. **Exported Files Per Dataset**
Each dataset folder will contain:
1. `01_first_fit.png` - First timepoint fit visualization
2. `02_peak_positions.png` - Peak position evolution over time (2 peaks)
3. `03_peak_amplitudes.png` - Peak amplitude evolution (indicator of label concentration)
4. `04_multiple_timepoints.png` - 6 representative timepoints
5. `{dataset_name}_results.csv` - Complete peak parameter table (center, amplitude, width, ratio)
6. `{dataset_name}_fit_reports.log` - Detailed fit statistics for all timepoints

### 5. **Data Processing Features**
- **Wavelength filtering**: All data automatically filtered to 280-450 nm (optimal for both label peaks)
- **Peak detection**: Adaptive detection of 315nm and 412nm peaks for labeled samples
- **Amplitude ratio tracking**: 315/412 ratio computed for stoichiometry analysis
- **Summary statistics**: Printed to console for each dataset during execution

### 6. **Data Organization**
All imported data contain 1000 time points, tracking spectral evolution during the pump-probe experiment at different X-ray transmission levels.

## Workflow

When you run the notebook:
1. ✓ All data loads automatically from both directories
2. ✓ Output folders are created automatically
3. ✓ Fits are performed on all datasets (progress printed to console)
4. ✓ All plots are saved to organized folders (no overwhelming display)
5. ✓ CSV results and fit reports saved for analysis

## Next Steps

The notebook is ready to run completely. All plots will be saved to the `spectral_analysis_output/` directory, making it easy to:
- Compare DTNB vs R37S behavior
- Analyze transmission level effects (5% vs 25% vs 50% vs 100%)
- Export results for further analysis
