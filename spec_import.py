import numpy as np
import pandas as pd

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
