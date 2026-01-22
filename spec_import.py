import numpy as np
import pandas as pd
import os
import re
from typing import Optional, Dict, List, Union

def load_absorbance_data(file_path, header_lines=0, footer_lines=0, time_point_interval=None):
    """
    Function to load absorbance data from an .asc or .txt file, handle header/footer,
    and return the data as a pandas DataFrame.
    
    The first column in the file is assumed to be the wavelength, and the remaining columns are absorbance values.
    
    Parameters:
    - file_path: Path to the .asc or .txt file.
    - header_lines: Number of header lines to skip while reading the file.
    - footer_lines: Number of footer lines to skip while reading the file.
    - time_point_interval: Time interval between each spectrum, in seconds (e.g., 0.1 for 100ms intervals).
    
    Returns:
    - df: DataFrame with 'Wavelength' as the first column and time points as columns.
    """
    
    # Initialize lists to store data
    wavelengths = []
    absorbance_data = []

    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        # Skip the header lines
        for _ in range(header_lines):
            next(file)
        
        # Read the data lines
        previous_line_empty = False  # To detect two consecutive empty lines
        for line in file:
            # Remove any leading/trailing whitespace characters
            line = line.strip()

            # Stop reading if the line contains text (footer content) or two consecutive empty lines
            if line == "":
                if previous_line_empty:
                    break  # Stop if two consecutive empty lines
                previous_line_empty = True
                continue
            
            # If the line contains non-numeric data (assumed to be footer), stop reading
            try:
                # Convert the line to a list of floats
                values = list(map(float, line.split()))
                absorbance_data.append(values)
                wavelengths.append(values[0])  # Assuming the first value is the wavelength
                previous_line_empty = False  # Reset if a valid line is found
            except ValueError:
                # If conversion to float fails, it's likely a footer or non-numeric line
                break

    # Convert absorbance data into a NumPy array
    absorbance_data = np.array(absorbance_data)

    # Create time labels (if not provided, default is 100ms intervals)
    if time_point_interval:
        num_time_points = absorbance_data.shape[1] - 1  # Subtract 1 for the wavelength column
        time_points = [f'{i*time_point_interval:.1f}s' for i in range(num_time_points)]
    else:
        time_points = [f'{i*100}ms' for i in range(absorbance_data.shape[1] - 1)]

    # Create a DataFrame with the data
    df = pd.DataFrame(absorbance_data[:, 1:], columns=time_points)  # Skip the first column (wavelength)
    df.insert(0, 'Wavelength', wavelengths)  # Insert wavelengths as the first column

    return df


def detect_column_type(column_name: str) -> Optional[str]:
    """
    Detect the type of column based on its name.
    
    Parameters:
    - column_name: Name of the column to detect
    
    Returns:
    - Column type: 'wavelength', 'wavenumber', 'absorbance', 'transmission', 
                   'fluorescence', 'time', or None if unknown
    """
    col_lower = column_name.lower().strip()
    
    # Wavelength/wavenumber detection
    if any(term in col_lower for term in ['wavelength', 'lambda', 'wl', 'nm']):
        return 'wavelength'
    if any(term in col_lower for term in ['wavenumber', 'cm-1', 'cm^-1']):
        return 'wavenumber'
    
    # Measurement type detection
    if any(term in col_lower for term in ['absorbance', 'abs', 'od', 'optical density']):
        return 'absorbance'
    if any(term in col_lower for term in ['transmission', 'transmittance', 'trans', '%t']):
        return 'transmission'
    if any(term in col_lower for term in ['fluorescence', 'fluor', 'emission', 'fl']):
        return 'fluorescence'
    
    # Metadata detection (check before time to avoid conflicts)
    if any(term in col_lower for term in ['concentration', 'conc']):
        return 'concentration'
    if any(term in col_lower for term in ['sample', 'id', 'name']):
        return 'sample'
    if any(term in col_lower for term in ['power', 'intensity']):
        return 'power'
    
    # Time detection (check last to avoid false positives with single letter)
    if any(term in col_lower for term in ['time', 'timestamp']):
        return 'time'
    # Only match single 't' if it's the whole column name
    if col_lower == 't':
        return 'time'
    
    return None


def load_tidy_data(file_path: str, 
                   wavelength_col: Optional[str] = None,
                   measurement_col: Optional[str] = None,
                   time_col: Optional[str] = None,
                   metadata_cols: Optional[List[str]] = None,
                   time_interval: Optional[float] = None,
                   **kwargs) -> pd.DataFrame:
    """
    Load spectroscopy data from various file formats and convert to tidy format.
    
    This function can handle:
    - CSV, Excel, TSV, and text files with headers
    - Various column naming conventions (case-insensitive)
    - Missing time information (will be inferred or created)
    - Metadata columns that should be preserved
    
    Parameters:
    - file_path: Path to the data file
    - wavelength_col: Name of wavelength column (auto-detected if None)
    - measurement_col: Name of measurement column (auto-detected if None)
    - time_col: Name of time column (auto-detected if None)
    - metadata_cols: List of metadata column names to preserve
    - time_interval: Time interval between measurements in seconds (for sequential data)
    - **kwargs: Additional arguments passed to pandas read functions
    
    Returns:
    - DataFrame in tidy format with columns: Wavelength, Time, Measurement, [metadata]
    """
    # Determine file type and load data
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.csv', '.txt', '.asc', '.tsv']:
        # Try reading with different delimiters
        try:
            df = pd.read_csv(file_path, **kwargs)
        except Exception:
            try:
                df = pd.read_csv(file_path, delimiter='\t', **kwargs)
            except Exception:
                df = pd.read_csv(file_path, delimiter=r'\s+', **kwargs)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Auto-detect columns if not specified
    column_types = {col: detect_column_type(col) for col in df.columns}
    
    # Find wavelength column
    if wavelength_col is None:
        wavelength_candidates = [col for col, ctype in column_types.items() 
                                if ctype in ['wavelength', 'wavenumber']]
        if wavelength_candidates:
            wavelength_col = wavelength_candidates[0]
        else:
            # Assume first numeric column is wavelength
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                wavelength_col = numeric_cols[0]
    
    # Find measurement column(s)
    if measurement_col is None:
        measurement_candidates = [col for col, ctype in column_types.items() 
                                 if ctype in ['absorbance', 'transmission', 'fluorescence']]
        if not measurement_candidates:
            # If no explicit measurement column, assume all numeric columns except wavelength are measurements
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if wavelength_col in numeric_cols:
                numeric_cols.remove(wavelength_col)
            measurement_candidates = numeric_cols
    else:
        measurement_candidates = [measurement_col]
    
    # Find time column
    if time_col is None:
        time_candidates = [col for col, ctype in column_types.items() if ctype == 'time']
        if time_candidates:
            time_col = time_candidates[0]
    
    # Identify metadata columns
    if metadata_cols is None:
        metadata_cols = []
        for col, ctype in column_types.items():
            if (col != wavelength_col and col not in measurement_candidates and 
                col != time_col and ctype in ['concentration', 'sample', 'power']):
                metadata_cols.append(col)
    
    # Convert to tidy format
    if wavelength_col is None:
        raise ValueError("Could not identify wavelength column. Please specify wavelength_col parameter.")
    
    # Check if data is already in tidy format (one measurement column)
    if len(measurement_candidates) == 1 and time_col is not None:
        # Data is already tidy
        tidy_df = df[[wavelength_col, time_col, measurement_candidates[0]] + metadata_cols].copy()
        tidy_df.columns = ['Wavelength', 'Time', 'Measurement'] + metadata_cols
    else:
        # Data needs to be converted to tidy format
        # Assume multiple measurement columns represent time series
        
        if time_col is None and time_interval is not None:
            # Create time points based on interval
            time_values = [i * time_interval for i in range(len(measurement_candidates))]
        elif time_col is not None:
            # Use existing time column values
            time_values = df[time_col].values if time_col in df.columns else None
        else:
            # Create sequential time points
            time_values = list(range(len(measurement_candidates)))
        
        # Melt the DataFrame to tidy format
        id_vars = [wavelength_col] + metadata_cols
        tidy_df = pd.melt(df, id_vars=id_vars, value_vars=measurement_candidates,
                         var_name='TimePoint', value_name='Measurement')
        
        # Add time values
        if time_values is not None:
            time_mapping = {col: time_values[i] for i, col in enumerate(measurement_candidates)}
            tidy_df['Time'] = tidy_df['TimePoint'].map(time_mapping)
        else:
            tidy_df['Time'] = tidy_df['TimePoint']
        
        # Reorder columns
        cols_order = ['Wavelength', 'Time', 'Measurement'] + metadata_cols
        tidy_df = tidy_df[[wavelength_col, 'Time', 'Measurement'] + metadata_cols]
        tidy_df.columns = cols_order
    
    return tidy_df


def load_spectroscopy_data(file_path: str,
                           format: str = 'auto',
                           tidy: bool = True,
                           **kwargs) -> pd.DataFrame:
    """
    Unified function to load spectroscopy data in various formats.
    
    This is a high-level wrapper that automatically detects data format and
    converts to the appropriate structure.
    
    Parameters:
    - file_path: Path to the data file
    - format: 'auto', 'matrix', or 'tidy' 
             - 'auto': auto-detect format
             - 'matrix': load as matrix (wavelengths x time points)
             - 'tidy': load and convert to tidy format
    - tidy: If True, return data in tidy format (ignored if format='matrix')
    - **kwargs: Additional arguments passed to load functions
    
    Returns:
    - DataFrame in the requested format
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # For legacy .asc/.txt files without headers, use original function
    if file_ext in ['.asc', '.txt'] and format in ['auto', 'matrix']:
        try:
            # Try to detect if file has headers
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                # Check if first line contains non-numeric characters (headers)
                try:
                    float(first_line.split()[0])
                    has_header = False
                except (ValueError, IndexError):
                    has_header = True
            
            if not has_header:
                # Use original function for backward compatibility
                return load_absorbance_data(file_path, **kwargs)
        except Exception:
            pass
    
    # For files with headers or when tidy format is requested
    if tidy or format == 'tidy':
        return load_tidy_data(file_path, **kwargs)
    else:
        # Load as matrix format
        return load_tidy_data(file_path, **kwargs)
