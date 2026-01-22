#!/usr/bin/env python3
"""
Example Usage of Enhanced Import Functions

This script demonstrates the various ways to use the enhanced import
functions in spec_import.py to handle different data formats.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spec_import import (
    load_absorbance_data, 
    load_tidy_data, 
    load_spectroscopy_data,
    detect_column_type
)

print("=" * 70)
print("Enhanced Import Functions - Usage Examples")
print("=" * 70)
print()

# Example 1: Auto-detect column types
print("Example 1: Column Type Auto-Detection")
print("-" * 70)
columns = ['Wavelength', 'nm', 'Absorbance', 'Time', 'Sample', 'Concentration']
for col in columns:
    col_type = detect_column_type(col)
    print(f"  '{col}' -> {col_type}")
print()

# Example 2: Load wide format CSV
print("Example 2: Load Wide Format CSV and Convert to Tidy")
print("-" * 70)
wide_csv = """Wavelength,0.0s,0.1s,0.2s
400,0.123,0.145,0.167
410,0.234,0.256,0.278
420,0.345,0.367,0.389"""

with open('/tmp/example_wide.csv', 'w') as f:
    f.write(wide_csv)

df = load_tidy_data('/tmp/example_wide.csv')
print("Input: Wide format CSV with time columns")
print("Output: Tidy format DataFrame")
print(df)
print()

# Example 3: Load data with metadata
print("Example 3: Load Data with Metadata (Sample, Concentration)")
print("-" * 70)
metadata_csv = """Wavelength,Sample,Concentration,Abs_0s,Abs_1s,Abs_2s
400,S1,0.5,0.123,0.145,0.167
410,S1,0.5,0.234,0.256,0.278
400,S2,1.0,0.223,0.245,0.267
410,S2,1.0,0.334,0.356,0.378"""

with open('/tmp/example_metadata.csv', 'w') as f:
    f.write(metadata_csv)

df = load_tidy_data('/tmp/example_metadata.csv', time_interval=1.0)
print("Input: CSV with Sample and Concentration metadata")
print("Output: Tidy format with metadata preserved")
print(df)
print()

# Example 4: Different measurement types
print("Example 4: Auto-Detect Different Measurement Types")
print("-" * 70)

# Transmission data
trans_csv = """nm,Transmission_0ms,Transmission_100ms
400,0.823,0.745
410,0.734,0.656"""

with open('/tmp/example_transmission.csv', 'w') as f:
    f.write(trans_csv)

df = load_tidy_data('/tmp/example_transmission.csv')
print("Transmission data:")
print(df.head())
print()

# Fluorescence data
fluor_csv = """Lambda,Fluorescence_t0,Fluorescence_t1
400,123.4,145.6
410,234.5,256.7"""

with open('/tmp/example_fluorescence.csv', 'w') as f:
    f.write(fluor_csv)

df = load_tidy_data('/tmp/example_fluorescence.csv')
print("Fluorescence data:")
print(df.head())
print()

# Example 5: Already tidy data
print("Example 5: Load Already Tidy Data")
print("-" * 70)
tidy_csv = """Wavelength,Time,Absorbance,Sample
400,0.0,0.123,S1
400,0.1,0.145,S1
410,0.0,0.234,S1
410,0.1,0.256,S1"""

with open('/tmp/example_tidy.csv', 'w') as f:
    f.write(tidy_csv)

df = load_tidy_data('/tmp/example_tidy.csv')
print("Input: Already in tidy format")
print("Output: Preserved as-is")
print(df)
print()

# Example 6: Unified function
print("Example 6: Using Unified load_spectroscopy_data()")
print("-" * 70)
df = load_spectroscopy_data('/tmp/example_wide.csv', tidy=True)
print("Automatically detects format and converts to tidy:")
print(df.head())
print()

print("=" * 70)
print("All examples completed successfully!")
print("=" * 70)
