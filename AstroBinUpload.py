import os
import sys
import pandas as pd
from astropy.io import fits

# Suppress warnings for cleaner notebook output
import warnings
warnings.filterwarnings('ignore')

def extract_fits_headers(directory: str) -> pd.DataFrame:
    """
    Extract headers from all FITS files in a specified directory and its subdirectories.
    
    Args:
        directory (str): The path of the directory containing FITS files.
    
    Returns:
        pandas.DataFrame: A DataFrame where each row represents the header of a FITS file,
                          including the file path as an additional column.
    """
    headers = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.fits'):
                file_path = os.path.join(root, file)
                try:
                    with fits.open(file_path) as hdul:
                        header = hdul[0].header
                        headers.append({**header, 'file_path': file_path})
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return pd.DataFrame(headers)

def create_calibration_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame for calibration data based on specific IMAGETYPs and GAIN.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing FITS header data.
    
    Returns:
        pandas.DataFrame: A DataFrame with columns 'TYPE', 'GAIN', and 'NUMBER'.
    """
    relevant_types = ['DARK', 'BIAS', 'FLAT', 'FLATDARKS']
    filtered_df = df[df['IMAGETYP'].isin(relevant_types)]
    group_counts = filtered_df.groupby(['IMAGETYP', 'GAIN']).size().reset_index(name='NUMBER')
    return group_counts.rename(columns={'IMAGETYP': 'TYPE'})



def create_lights_df(df: pd.DataFrame, calibration_df: pd.DataFrame, filters_df: pd.DataFrame,
                     bortle: int, mean_sqm: float, mean_fwhm: float) -> pd.DataFrame:
    """
    Creates a DataFrame for 'LIGHT' type data with specified columns, 
    replacing filter names with codes and aggregating relevant data.

    Args:
        df (pd.DataFrame): DataFrame containing FITS header data.
        calibration_df (pd.DataFrame): DataFrame containing calibration data.
        filters_df (pd.DataFrame): DataFrame containing filter codes.
        bortle (int): Bortle scale value.
        mean_sqm (float): Mean sqm value.
        mean_fwhm (float): Mean FWHM value.

    Returns:
        pd.DataFrame: Aggregated DataFrame with 'LIGHT' type data.
    """
    light_df = df[df['IMAGETYP'] == 'LIGHT'].copy()
    light_df['date'] = pd.to_datetime(light_df['DATE-LOC']).dt.date
    light_df['gain'] = light_df['GAIN'].astype(calibration_df['GAIN'].dtype)
    light_df.rename(columns={'XBINNING': 'binning', 'EXPOSURE': 'duration'}, inplace=True)
    light_df = light_df.merge(filters_df, left_on='FILTER', right_on='filter')

    aggregated = light_df.groupby(['date', 'code', 'gain', 'binning', 'duration']).agg(
        number=('DATE-LOC', 'count'),
        sensorcooling=('CCD-TEMP', 'mean'),
        temperature=('FOCTEMP', 'mean')
    ).reset_index()

    aggregated['sensorCooling'] = aggregated['sensorcooling'].round().astype(int)
    aggregated['bortle'] = bortle
    aggregated['meanSqm'] = mean_sqm
    aggregated['meanFwhm'] = mean_fwhm

    def get_calibration_data(row: pd.Series, cal_type: str) -> int:
        match = calibration_df[(calibration_df['TYPE'] == cal_type) & (calibration_df['GAIN'] == row['gain'])]
        return match['NUMBER'].sum() if not match.empty else 0

    aggregated['darks'] = aggregated.apply(get_calibration_data, cal_type='DARK', axis=1)
    aggregated['flats'] = aggregated.apply(get_calibration_data, cal_type='FLAT', axis=1)
    aggregated['bias'] = aggregated.apply(get_calibration_data, cal_type='BIAS', axis=1)
    aggregated['flatDarks'] = aggregated.apply(get_calibration_data, cal_type='FLATDARKS', axis=1)
    aggregated['sensorCooling'] = aggregated['sensorCooling'].round(1)
    aggregated['temperature'] = aggregated['temperature'].round(1)
    aggregated.rename(columns={'code': 'filter'}, inplace=True)

    column_order = ['date', 'filter', 'number', 'duration', 'binning', 'gain', 
                    'sensorCooling', 'darks', 'flats', 'flatDarks', 'bias', 'bortle',
                    'meanSqm', 'meanFwhm', 'temperature']

    return aggregated.reindex(columns=column_order)


def consolidate_headers(directories):
    """
    Aggregates FITS file headers from multiple directories into a single pandas DataFrame.

    This function iterates through a list of directories, extracts headers from FITS files in each directory using
    the `extract_fits_headers` function, and then combines them into one DataFrame.

    Args:
        directories (list of str): A list of directory paths to process.

    Returns:
        pandas.DataFrame: A consolidated DataFrame containing all headers from the FITS files in the provided directories.
    """
    all_headers = []
    for directory in directories:
        headers_df = extract_fits_headers(directory)
        all_headers.append(headers_df)
    return pd.concat(all_headers, ignore_index=True)

def main():
    """
    Main function to execute the script. 

    This function performs the following steps:
    1. Checks if the command line arguments meet the expected format (at least one directory path and three numeric arguments).
    2. Validates the existence of the provided directory paths.
    3. Parses and validates the last three command line arguments as numeric values.
    4. Consolidates FITS file headers from the specified directories.
    5. Processes and exports the 'LIGHT' type data to a CSV file.

    The script exits with an error message if the command line arguments do not meet the requirements or if any of the 
    provided directories do not exist.
    """
    # Check if there are at least four arguments (including the script name)
    if len(sys.argv) < 4:
        print("Error: Please provide at least one directory path and three numeric arguments.")
        sys.exit(1)

    # All arguments except the last three are directory paths
    directory_paths = sys.argv[1:-3]

    # Check if directories exist
    for directory in directory_paths:
        if not os.path.isdir(directory):
            print(f"Error: The directory '{directory}' does not exist.")
            sys.exit(1)

    # Try to parse the last three arguments as numeric variables
    try:
        numeric_variables = [float(var) for var in sys.argv[-3:]]
    except ValueError:
        print("Error: The last three arguments must be numeric.")
        sys.exit(1)

    # Process directories and consolidate headers
    fits_headers_df = consolidate_headers(directory_paths)

    calibration = create_calibration_df(fits_headers_df)
    filters = pd.read_csv('filters.csv')
    #lights = create_lights_df(fits_headers_df, calibration, filters, bortle, mean_sqm, mean_fwhm)
    lights = create_lights_df(fits_headers_df, calibration, filters, float(sys.argv[-3]), float(sys.argv[-2]),float(sys.argv[-1]))

    output_csv = os.path.basename(directory_paths[0]) + " acquisition.csv"
    lights.to_csv(output_csv , index=False)
    print(f"Data exported to {output_csv}")


if __name__ == "__main__":
    main()