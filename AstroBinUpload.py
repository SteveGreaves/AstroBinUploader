import os
import sys
import pandas as pd
from astropy.io import fits

#
#Test Change
def extract_fits_headers(directory):
    """
    Extracts headers from all FITS files in a specified directory and its subdirectories.
    
    Parameters:
    directory (str): The path of the directory containing FITS files.
    
    Returns:
    pandas.DataFrame: A DataFrame where each row represents the header of a FITS file,
                      including the file path as an additional column.
    """
    headers = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.fits'):
                file_path = os.path.join(root, file)
                try:
                    with fits.open(file_path) as hdul:
                        header = hdul[0].header
                        headers.append(dict(header, file_path=file_path))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return pd.DataFrame(headers)

def create_calibration_df(df):
    """
    Creates a DataFrame for calibration data based on specific IMAGETYPs and GAIN.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing FITS header data.
    
    Returns:
    pandas.DataFrame: A DataFrame with columns 'TYPE', 'GAIN', and 'NUMBER'.
    """
    relevant_types = ['DARK', 'BIAS', 'FLAT']
    filtered_df = df[df['IMAGETYP'].isin(relevant_types)]
    group_counts = filtered_df.groupby(['IMAGETYP', 'GAIN']).size().reset_index(name='NUMBER')
    return group_counts.rename(columns={'IMAGETYP': 'TYPE'})

def create_lights_df(df, calibration_df, filters_df, bortle, mean_sqm, mean_fwhm):
    """
    Creates a DataFrame for 'LIGHT' type data with specified columns, replacing filter names with codes.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing FITS header data.
    calibration_df (pandas.DataFrame): The DataFrame containing calibration data.
    filters_df (pandas.DataFrame): The DataFrame containing filter codes.
    bortle (int): Bortle scale value.
    mean_sqm (float): Mean sqm value.
    mean_fwhm (float): Mean FWHM value.
    
    Returns:
    pandas.DataFrame: A DataFrame with specified columns for 'LIGHT' type data.
    """
    light_df = df[df['IMAGETYP'] == 'LIGHT'].copy()
    light_df['date'] = pd.to_datetime(light_df['DATE-LOC']).dt.date
    light_df['gain'] = light_df['GAIN'].astype(calibration_df['GAIN'].dtype)

    light_df = light_df.merge(filters_df, left_on='FILTER', right_on='Filter')
    aggregated = light_df.groupby(['date', 'Code', 'gain', 'XBINNING']).agg(
        number=('DATE-LOC', 'count'),
        duration=('EXPOSURE', 'mean'),
        sensorCooling=('CCD-TEMP', 'mean'),
        temperature=('FOCTEMP', 'mean')
    ).reset_index()

    aggregated['sensorCooling'] = aggregated['sensorCooling'].round().astype(int)
    aggregated['flatDarks'] = 0
    aggregated['bortle'] = bortle
    aggregated['meanSqm'] = mean_sqm
    aggregated['meanFWHm'] = mean_fwhm

    def get_calibration_data(row, cal_type):
        match = calibration_df[(calibration_df['TYPE'] == cal_type) & (calibration_df['GAIN'] == row['gain'])]
        return match['NUMBER'].sum() if not match.empty else 0

    aggregated['darks'] = aggregated.apply(get_calibration_data, cal_type='DARK', axis=1)
    aggregated['flats'] = aggregated.apply(get_calibration_data, cal_type='FLAT', axis=1)
    aggregated['bias'] = aggregated.apply(get_calibration_data, cal_type='BIAS', axis=1)

    aggregated['sensorCooling'] = aggregated['sensorCooling'].round(1)
    aggregated['temperature'] = aggregated['temperature'].round(1)

    column_order = ['date', 'filter', 'number', 'duration', 'binning', 'gain', 
                    'sensorCooling', 'darks', 'flats', 'flatDarks', 'bias', 
                    'meanSqm', 'meanFWHm', 'temperature']
    return aggregated.reindex(columns=column_order)

def process_fits_data(directory, filter_file, bortle, mean_sqm, mean_fwhm):
    """
    Processes FITS data to create and export a CSV file with light and calibration data.
    
    Parameters:
    directory (str): Directory containing FITS files.
    filter_file (str): CSV file containing filter codes.
    bortle (int): Bortle scale value.
    mean_sqm (float): Mean sqm value.
    mean_fwhm (float): Mean fwhm value.
    """
    try:
        fits_headers_df = extract_fits_headers(directory)
        if fits_headers_df.empty:
            print("No FITS headers found.")
            return
        calibration = create_calibration_df(fits_headers_df)
        filters = pd.read_csv(filter_file)
        lights = create_lights_df(fits_headers_df, calibration, filters, bortle, mean_sqm, mean_fwhm)
        output_csv = os.path.join(directory, 'AstroBin_Acquisition_File.csv')
        lights.to_csv(output_csv, index=False)
        print(f"Data exported to {output_csv}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    """
    Main function to prompt user input and process FITS data.
    """
    try:
        directory = input("Enter the image directory: ")
        if not os.path.isdir(directory):
            raise ValueError("Directory not found.")
        filter_file = input("Enter the filter file name: ")
        if not os.path.isfile(filter_file):
            raise ValueError("Filter file not found.")
        bortle = int(input("Enter the Bortle scale value: "))
        mean_sqm = float(input("Enter the mean SQM value: "))
        mean_fwhm = float(input("Enter the mean FWHM value: "))
        process_fits_data(directory, filter_file, bortle, mean_sqm, mean_fwhm)
    except ValueError as e:
        print(f"Invalid input: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
