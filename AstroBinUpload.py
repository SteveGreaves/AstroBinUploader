# coding: utf-8

"""
acqusition.csv uploader see (https://welcome.astrobin.com/importing-acquisitions-from-csv/)

This implementation is not endorsed nor related with AstroBin development team.

Copyright (C) 2023 Steve Greaves

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__version__ = "1.0.3"
 
import os
import sys
import pandas as pd
import datetime
import requests
from astropy.io import fits
import warnings
import re
import math
from datetime import datetime as dt

# Suppressing warnings for specific use-case, but be aware of potential debugging issues
warnings.filterwarnings('ignore')

# Define functions

def extract_fits_headers(directories, default_values):
    """
    Extract headers from FITS files in given directories, using default values for missing fields.

    Parameters:
    directories (list): List of directory paths to search for FITS files.
    default_values (dict): Dictionary of default values for the FITS headers.

    Returns:
    DataFrame: Pandas DataFrame containing the headers from the FITS files.
    """
    headers = []
    required_fields_light = ["EXPOSURE", "DATE-LOC", "XBINNING", "GAIN", "XPIXSZ", "CCD-TEMP", 
                             "FOCALLEN", "SITELAT", "SITELONG", "FILTER", "OBJECT", "FOCTEMP"]
    required_fields_flat = ["FILTER", "GAIN", "XBINNING"]  # Fields specific for FLAT frames
    swcreate_printed = False

    # Extract headers from FITS files
    for directory in directories:
        for root, _, files in os.walk(directory):
            print(f"Reading directory: {root}")  # Print each directory being processed
            for file in files:

                if file.lower().endswith('.fits'):
                    file_path = os.path.join(root, file)
                    try:
                        with fits.open(file_path) as hdul:
                            header = hdul[0].header

                            # Print SWCREATE from the first header file, if not already done
                            if not swcreate_printed:
                                swcreate = header.get('SWCREATE', 'unknown package')
                                swcreate_printed = True

                            # Determine required fields based on IMAGETYP
                            imagetyp = header.get('IMAGETYP')
                            required_fields = required_fields_light if imagetyp == 'LIGHT' else required_fields_flat if imagetyp == 'FLAT' else []
                            for field in required_fields:
                                header.setdefault(field, default_values.get(field))

                            headers.append(dict(header, file_path=file_path))
                    except (OSError, IOError) as e:
                        print(f"Error reading {file_path}: {e}")
    print(f"\nImages captured by {swcreate}")
    return pd.DataFrame(headers)


def format_seconds_to_hms(seconds):
    """
    Convert seconds to a formatted string in hours, minutes, and seconds.
    
    Parameters:
    seconds (int): Total number of seconds.

    Returns:
    str: Formatted time string in 'H hrs M mins S secs' format.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours} hrs")
    if minutes > 0:
        time_parts.append(f"{minutes} mins")
    time_parts.append(f"{seconds:.0F} secs")

    return ' '.join(time_parts)


def summarize_session(df):    
    """
    Summarize observation session by processing LIGHT, FLAT, BIAS, and DARK frames.
    
    Parameters:
    df (DataFrame): The DataFrame containing the FITS headers data.
    """
    if 'IMAGETYP' in df:
        print("\nObservation session Summary:")

        # Process LIGHT, FLAT, BIAS, and DARK frames
        for imagetyp in ['LIGHT', 'FLAT', 'BIAS', 'DARK']:
            if imagetyp in df['IMAGETYP'].values:
                group = df[df['IMAGETYP'] == imagetyp]

                # For LIGHT and FLAT frames
                if imagetyp in ['LIGHT', 'FLAT']:
                    print(f"\n{imagetyp}S:")
                    for filter_type, group_df in group.groupby('FILTER'):
                        frame_count = group_df.shape[0]
                        total_exposure = group_df['EXPOSURE'].sum()
                        formatted_time = format_seconds_to_hms(total_exposure)
                        print(f"  Filter {filter_type}:\t {frame_count} frames, Exposure time: {formatted_time}")

                # For BIAS and DARK frames, sum on GAIN
                elif imagetyp in ['BIAS', 'DARK']:
                    for gain_value, gain_group in group.groupby('GAIN'):
                        frame_count = gain_group.shape[0]
                        total_exposure = gain_group['EXPOSURE'].sum()
                        formatted_time = format_seconds_to_hms(total_exposure)
                        print(f"\n{imagetyp} with GAIN {gain_value}:\t {frame_count} frames, Exposure time: {formatted_time}")
                    
                # Total exposure for LIGHT frames
                if imagetyp == 'LIGHT':
                    total_light_exposure = group['EXPOSURE'].sum()
                    formatted_total_light_time = format_seconds_to_hms(total_light_exposure)
                    print(f"\nTotal session exposure for LIGHTs:\t {formatted_total_light_time}")
    else:
        print("No 'IMAGETYP' column found in headers.")



def create_calibration_df(df):
    """
    Creates a DataFrame for calibration data based on specific IMAGETYPs, GAIN, and FILTER.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing FITS header data.
    
    Returns:
    pandas.DataFrame: A DataFrame with columns 'TYPE', 'GAIN', 'FILTER' (if applicable), and 'NUMBER'.
    """
    relevant_types = ['DARK', 'BIAS', 'FLAT', 'FLATDARKS']
    filtered_df = df[df['IMAGETYP'].isin(relevant_types)]

    # Group by IMAGETYP and GAIN, and additionally by FILTER for FLAT
    if 'FILTER' in df.columns:
        filtered_df.loc[filtered_df['IMAGETYP'] != 'FLAT', 'FILTER'] = ''
        group_counts = filtered_df.groupby(['IMAGETYP', 'GAIN', 'FILTER']).size().reset_index(name='NUMBER')
    else:
        group_counts = filtered_df.groupby(['IMAGETYP', 'GAIN']).size().reset_index(name='NUMBER')

    return group_counts.rename(columns={'IMAGETYP': 'TYPE'})


def create_lights_df(df: pd.DataFrame, calibration_df: pd.DataFrame)-> pd.DataFrame:
                     #bortle: int, mean_sqm: float, mean_fwhm: float) -> pd.DataFrame:
    """
    Creates a DataFrame for 'LIGHT' type data 

    Args:
        df (pd.DataFrame): DataFrame containing FITS header data.
        calibration_df (pd.DataFrame): DataFrame containing calibration data.


    Returns:
        pd.DataFrame: Aggregated DataFrame with 'LIGHT' type data.
    """
    light_df = df[df['IMAGETYP'] == 'LIGHT'].copy()
    light_df['DATE'] = pd.to_datetime(light_df['DATE-LOC']).dt.date


    return pd.DataFrame(light_df)

def get_bortle_sqm(lat, lon, api_key, api_endpoint):
    """
    Retrieves the Bortle scale classification and SQM (Sky Quality Meter) value for a given latitude and longitude.

    This function makes a request to a specified API endpoint using the provided API key. It fetches the artificial
    brightness for the specified coordinates and calculates the SQM. The Bortle scale classification is then determined
    based on the SQM value.

    Args:
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.
    api_key (str): API key for accessing the light pollution data.
    api_endpoint (str): URL of the API endpoint to retrieve light pollution data.

    Returns:
    tuple: A tuple containing the Bortle scale classification (int) and the SQM value (float), rounded to two decimal places.
    """
    params = {
        'ql': 'wa_2015',
        'qt': 'point',
        'qd': f'{lon},{lat}',
        'key': api_key
    }
    response = requests.get(api_endpoint, params=params)
    response.raise_for_status()
    artificial_brightness = float(response.text)
    sqm = (math.log10((artificial_brightness + 0.171168465)/108000000)/-0.4)
    bortle_class = sqm_to_bortle(sqm)
    return bortle_class, round(sqm, 2)

def sqm_to_bortle(sqm):
    """
    Converts an SQM (Sky Quality Meter) value to the corresponding Bortle scale classification.

    The Bortle scale is used to quantify the astronomical observability of celestial objects, affected by light pollution.

    Args:
    sqm (float): The SQM value indicating the level of light pollution.

    Returns:
    int: The Bortle scale classification (ranging from 1 to 9), where 1 indicates the darkest skies and 9 the brightest.
    """
    if sqm > 21.99:
        return 1
    elif 21.50 <= sqm <= 21.99:
        return 2
    elif 21.25 <= sqm <= 21.49:
        return 3
    elif 20.50 <= sqm <= 21.24:
        return 4
    elif 19.50 <= sqm <= 20.49:
        return 5
    elif 18.50 <= sqm <= 19.49:
        return 6
    elif 17.50 <= sqm <= 18.49:
        return 7
    elif 17.00 <= sqm <= 17.49:
        return 8
    else:
        return 9



def calculate_auxiliary_parameters(df,hfr_set):
    """
    Calculates and updates auxiliary parameters for each row in a DataFrame based on astronomical observation data.

    This function processes a DataFrame containing astronomical observation data. It updates each row with Bortle scale,
    SQM (Sky Quality Meter) values, HFR (Half Flux Radius), IMSCALE (Image Scale), and FWHM (Full Width Half Maximum). 
    The Bortle scale and SQM values are either fetched from an API (using latitude and longitude from the DataFrame) 
    or retrieved from a local 'sites.csv' file if already available. The function also calculates HFR, IMSCALE, and FWHM
    based on provided FITS file information in the DataFrame.

    Args:
    df (pd.DataFrame): A DataFrame containing FITS file data. Expected to have columns 'SITELAT', 'SITELONG', 'file_path',
                       'XPIXSZ', and 'FOCALLEN'.
    hfr_set (float)  : A floating point number representing the half-flux radius in pixels entered by the user.

    Returns:
    pd.DataFrame: The input DataFrame with additional columns for Bortle scale, SQM, HFR, IMSCALE, and FWHM.
                  These columns are named 'BORTLE', 'SQM', 'HFR', 'IMSCALE', and 'FWHM', respectively.

    Notes:
    - The function reads API key and endpoint from 'secret.csv'.
    - If 'secret.csv' is not found, it requires manual data entry for Bortle and SQM in 'configuration.csv'.
    - The 'sites.csv' file is used to store and retrieve existing Bortle and SQM values for known locations.
    - HFR, IMSCALE, and FWHM calculations are based on regex matching and basic astronomical formulas.
    """


    try:
        sites = pd.read_csv('sites.csv')
    except FileNotFoundError:
        sites = pd.DataFrame(columns=['Latitude', 'Longitude', 'Bortle', 'SQM'])

    try:
        secret = pd.read_csv('secret.csv')
        api_key = secret['API Key'].iloc[0]
        api_endpoint = secret['API Endpoint'].iloc[0]
    except FileNotFoundError:
        print("Secret file 'secret.csv' not found. Please provide API key and endpoint.")
        api_key = None
        api_endpoint = None

    processed_sites = set()  # Set to keep track of processed lat-lon pairs

    for index, row in df.iterrows():
        lat, lon = round(row['SITELAT'],3), round(row['SITELONG'],3)
        
        site_data = sites[(sites['Latitude'] == lat) & (sites['Longitude'] == lon)]
        if site_data.empty and api_key:
            try:
                bortle, sqm = get_bortle_sqm(lat, lon, api_key,api_endpoint)
                new_site = {'Latitude': lat, 'Longitude': lon, 'Bortle': bortle, 'SQM': sqm}
                new_site_df = pd.DataFrame([new_site])
                sites = pd.concat([sites, new_site_df], ignore_index=True)
                sites.to_csv('sites.csv', index=False)
                df.at[index, 'BORTLE'] = bortle
                df.at[index, 'SQM'] = sqm
                print(f"\nRetrieved Bortle of {bortle} and SQM of {sqm} mag/arc-seconds^2 for lat {lat}, lon {lon} from API at {api_endpoint}.")
            except requests.exceptions.RequestException as e:
                print(f"Failed to retrieve data from the API for lat {lat}, lon {lon}: {e}")
                df.at[index, 'BORTLE'] = 0
                df.at[index, 'SQM'] = 0
        elif not site_data.empty:
            if (lat, lon) not in processed_sites:
                processed_sites.add((lat, lon))
                bortle, sqm = site_data.iloc[0]['Bortle'], site_data.iloc[0]['SQM']
                print(f"\nUsed existing Bortle of {bortle} and SQM of {sqm} mag/arc-seconds^2 for lat {lat}, lon {lon} from sites.csv.")
               
            bortle, sqm = site_data.iloc[0]['Bortle'], site_data.iloc[0]['SQM']
            df.at[index, 'BORTLE'] = bortle
            df.at[index, 'SQM'] = sqm
        else:
            if (lat, lon) not in processed_sites:
                processed_sites.add((lat, lon))
                print(f"\nNo data found for lat {lat}, lon {lon}. Set Bortle and SQM to 0.")
            df.at[index, 'BORTLE'] = 0
            df.at[index, 'SQM'] = 0


        file_path = row['file_path']
        hfr_match = re.search(r'HFR_([0-9.]+)', file_path)
        hfr = float(hfr_match.group(1)) if hfr_match else hfr_set
        imscale = float(row['XPIXSZ']) / float(row['FOCALLEN']) * 206.265
        fwhm = hfr * imscale if hfr >=0.0 else 0.0

        df.at[index, 'HFR'] = hfr
        df.at[index, 'IMSCALE'] = imscale
        df.at[index, 'FWHM'] = fwhm

    print('\nCompleted sky quality extraction')
    return df.round(2)


def aggregate_parameters(lights_df, calibration_df):
    """
    Aggregates observational parameters from lights_df and calibration data from calibration_df.

    The function aggregates data by date, filter, gain, xbinning, and exposure. It computes the mean values
    for sensor cooling, temperature, and mean FWHM. It also integrates calibration data (darks, flats, bias,
    flatDarks) based on the type of filter and gain settings. Additionally, it includes Bortle scale and SQM
    values from the lights_df into the aggregated data.

    Parameters:
    lights_df (pd.DataFrame): DataFrame containing observational data with FITS headers.
    calibration_df (pd.DataFrame): DataFrame containing calibration data like darks, flats, etc.

    Returns:
    pd.DataFrame: Aggregated DataFrame with observational and calibration parameters, 
                  including sensor cooling, temperature, mean FWHM, and more.

    Notes:
    - The function expects 'lights_df' to contain columns like 'date', 'filter', 'gain', etc.
    - The 'calibration_df' is expected to contain calibration data corresponding to 'lights_df'.
    """

    
    lights_df.columns = lights_df.columns.str.lower()

    aggregated_df = lights_df.groupby(['date', 'filter', 'gain', 'xbinning', 'exposure']).agg(
            number=('date', 'count'),
            sensorCooling=('ccd-temp', 'mean'),
            temperature=('foctemp', 'mean'),
            meanFwhm = ('fwhm', 'mean')
        ).reset_index()
    aggregated_df.rename(columns={'xbinning': 'binning',
                                  'exposure': 'duration'}, inplace=True)

    
    
    def get_calibration_data(row: pd.Series, cal_type: str, calibration_df: pd.DataFrame) -> int:
        # For FLAT type, match both 'GAIN' and 'FILTER'
        if cal_type == 'FLAT':
            match = calibration_df[(calibration_df['TYPE'] == cal_type) & 
                                   (calibration_df['GAIN'] == row['gain']) &
                                   (calibration_df['FILTER'].str.upper() == row['filter'].upper())]
        else:
            match = calibration_df[(calibration_df['TYPE'] == cal_type) & 
                                   (calibration_df['GAIN'] == row['gain'])]
    
        return match['NUMBER'].sum() if not match.empty else 0
    
    # Then apply it to your aggregated DataFrame
    aggregated_df['darks'] = aggregated_df.apply(get_calibration_data, args=('DARK', calibration_df), axis=1)
    aggregated_df['flats'] = aggregated_df.apply(get_calibration_data, args=('FLAT', calibration_df), axis=1)
    aggregated_df['bias'] = aggregated_df.apply(get_calibration_data, args=('BIAS', calibration_df), axis=1)
    aggregated_df['flatDarks'] = aggregated_df.apply(get_calibration_data, args=('FLATDARKS', calibration_df), axis=1)

    aggregated_df['bortle'] = lights_df['bortle']
    aggregated_df['meanSqm'] = lights_df['sqm']

    aggregated_df['sensorCooling'] = aggregated_df['sensorCooling'].round().astype(int)

    return aggregated_df.round(2)


def create_astrobin_output(aggregated_df, filter_df):
    """
    Transforms aggregated observational data into a format suitable for AstroBin output.

    This function maps filter names to their corresponding codes using 'filter_df', 
    reorders columns to match the expected AstroBin format, and renames the 'code' 
    column back to 'filter'. The function also drops the original 'filter' column 
    and ensures the final DataFrame is rounded to two decimal places for neatness.

    Parameters:
    aggregated_df (pd.DataFrame): DataFrame containing aggregated observational data.
    filter_df (pd.DataFrame): DataFrame that maps filter names to their corresponding codes.

    Returns:
    pd.DataFrame: Transformed DataFrame in the format suitable for AstroBin, with columns reordered 
                  and filter names replaced by their codes.

    Notes:
    - The function assumes that 'aggregated_df' contains all necessary columns for AstroBin output.
    - The 'filter_df' is used to map filter names to codes.
    """
    #Create a mapping Series
    filter_to_code = filter_df.set_index('filter')['code']
    # Find the position of the 'filter' column
    filter_col_index = aggregated_df.columns.get_loc('filter')
    # Insert the 'code' column right after 'filter'
    aggregated_df.insert(filter_col_index + 1, 'code', aggregated_df['filter'].map(filter_to_code))
    column_order = ['date', 'code', 'number', 'duration', 'binning', 'gain', 
                    'sensorCooling', 'darks', 'flats', 'flatDarks', 'bias', 'bortle',
                    'meanSqm', 'meanFwhm', 'temperature']
    df = aggregated_df.copy().drop('filter',axis=1).reindex(columns=column_order).round(2)
    df.rename(columns={'code': 'filter'}, inplace=True)
    return df 
    
def convert_default(defaults_df):
    """
    Converts specific fields of a DataFrame to appropriate data types and formats.

    This function iterates over each row of the input DataFrame `defaults_df`. It converts
    certain fields specified in `float_keys` to float types, and the 'XBINNING' field to an integer.
    It also formats the 'DATE-LOC' field into a standardized datetime string.

    Parameters:
    defaults_df (pandas.DataFrame): A DataFrame where each row represents a key-value pair.
        The DataFrame should have at least two columns: 'Key' and 'Value'.

    Returns:
    dict: A dictionary where each key-value pair corresponds to the 'Key' and 'Value' 
        columns of the input DataFrame, after applying the necessary conversions and formatting.

    Notes:
    - Any non-convertible values in float and integer fields are coerced to NaN.
    - The 'DATE-LOC' field is expected to be in the format '%Y-%m-%d' and is converted to 
      an ISO 8601 format '%Y-%m-%dT%H:%M:%S.%f', truncated to milliseconds.
    - The function assumes 'defaults_df' is not empty and contains the specified columns.
    """

    float_keys = ['EXPOSURE', 'GAIN', 'XPIXSZ', 'CCD-TEMP', 'FOCALLEN', 'SITELAT', 'SITELONG', 'FOCTEMP','HFR']
    #Iterate over the DataFrame and update the 'Value' column
    for i in range(len(defaults_df)):
        key = defaults_df.at[i, 'Key']
        if key in float_keys:
            defaults_df.at[i, 'Value'] = pd.to_numeric(defaults_df.at[i, 'Value'], errors='coerce')
        elif key == 'XBINNING':
            defaults_df.at[i, 'Value'] = pd.to_numeric(defaults_df.at[i, 'Value'], downcast='integer', errors='coerce')

    # Convert DATE-LOC to datetime object
    defaults_df.loc[defaults_df['Key'] == 'DATE-LOC', 'Value'] = dt.strptime(defaults_df.loc[defaults_df['Key'] == 'DATE-LOC', 'Value'].iloc[0], "%Y-%m-%d").strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    # return the DataFrame as a dictionary
    return dict(zip(defaults_df['Key'], defaults_df['Value']))

def main():
    """
    Main function to execute the script for processing astronomical data.

    Steps:
    1. Validates the existence of 'defaults.csv' file and reads it.
    2. Converts data types in the 'defaults' DataFrame using 'convert_default'.
    3. Validates the existence of 'filters.csv' file and reads it.
    4. Validates command line arguments for at least one directory path.
    5. Validates the existence of specified directory paths.
    6. Processes the directories to consolidate FITS file headers.
    7. Summarizes session data and processes light type data.
    8. Calculates auxiliary parameters and aggregates parameters for analysis.
    9. Creates a final DataFrame for AstroBin output.
    10. Exports the processed data to a CSV file.

    The script exits with an error message if:
    - 'defaults.csv' or 'filters.csv' is not found.
    - No directory paths are provided, or any provided directories do not exist.
    - Any issues occur during data processing.

    Notes:
    - The script assumes 'defaults.csv' contains necessary default values for FITS headers.
    - 'filters.csv' is expected to contain filter-specific data.
    - Directory paths provided as command line arguments should contain FITS files.
    - The output CSV file is named based on the first directory path provided.
    """

     # Validate defaults file
    defaults_file = 'defaults.csv'
    if not os.path.exists(defaults_file):
        print(f'{defaults_file} not found.')
        sys.exit(1)

    defaults_df = pd.read_csv('defaults.csv')
    #Ensure correct data types in data frame
    defaults = convert_default(defaults_df)
    
    #obtain hfr_set
    hfr_set = defaults['HFR']
    
    #Validate filter file
    filter_file = 'filters.csv'
    if not os.path.exists(filter_file):
        print(f'{filter_file} not found.')
        sys.exit(1)
    
    filter_df = pd.read_csv(filter_file)
    

    # Validate command line arguments for directory paths and hfr_set
    if len(sys.argv) < 2:
        print("Error: Please provide at least one directory path.")
        sys.exit(1)

    directory_paths = sys.argv[1:]
    # Validate directory paths
    for directory in directory_paths:
        if not os.path.isdir(directory):
            print(f"Error: The directory '{directory}' does not exist.")
            sys.exit(1)

    # Process directories and consolidate headers
    print('\nReading FITS headers...\n')
    headers_df = extract_fits_headers(directory_paths,defaults)
    summarize_session(headers_df)
    calibration_df = create_calibration_df(headers_df)
    lights_df = create_lights_df(headers_df, calibration_df)
    lights_df = calculate_auxiliary_parameters(lights_df,hfr_set)
    aggregated_df = aggregate_parameters(lights_df, calibration_df)
    astroBin_df = create_astrobin_output(aggregated_df, filter_df)

    # Export data to CSV
    output_csv = os.path.basename(directory_paths[0]) + " acquisition.csv"
    astroBin_df.to_csv(output_csv, index=False)
    print(f"\nData exported to {output_csv}")


if __name__ == "__main__":
    main()


