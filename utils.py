import os
import sys
import pandas as pd
from astropy.io import fits

#import a warning suppression library to keep the notebook clean
import warnings
warnings.filterwarnings('ignore')


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
    relevant_types = ['DARK', 'BIAS', 'FLAT','FLATDARKS']
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
    # Create a dataframe with only light image header data extracted from the main data frame
    light_df = df[df['IMAGETYP'] == 'LIGHT'].copy()
    
    # Extract the date from the date-time information
    light_df['date'] = pd.to_datetime(light_df['DATE-LOC']).dt.date
    #group on gain
    light_df['gain'] = light_df['GAIN'].astype(calibration_df['GAIN'].dtype)
    #rename XBINNING column
    light_df.rename(columns={'XBINNING': 'binning'}, inplace=True)
    #rename Exposure column
    light_df.rename(columns={'EXPOSURE': 'duration'}, inplace=True)
    
    #Replace the filter names with Astrobin filter codes 
    #Merge light_df with filters_df by matching the 'FILTER' column in light_df to the 'Filter' column in filters_df
    #This combinines data from both DataFrames based on filter names.
    light_df = light_df.merge(filters_df, left_on='FILTER', right_on='filter')

    #Group light_df by 'date', 'Filter code ', 'gain', and 'XBINNING'.git/
    #then calculate the sum of all frames in the group defined on the specfic 'DATE-LOC',
    #the mean of 'EXPOSURE', 'CCD-TEMP', and 'FOCTEMP',

    aggregated = light_df.groupby(['date', 'code', 'gain', 'binning', 'duration']).agg(
        number=('DATE-LOC', 'count'),
        #duration=('EXPOSURE', 'mean'),
        sensorcooling=('CCD-TEMP', 'mean'),
        temperature=('FOCTEMP', 'mean')
    ).reset_index()
    #aggregate all other parameters for the aggregated dataframe
    aggregated['sensorCooling'] = aggregated['sensorcooling'].round().astype(int)
    aggregated['bortle'] = bortle
    aggregated['meanSqm'] = mean_sqm
    aggregated['meanFwhm'] = mean_fwhm
    #aggregated['filter_name'] = light_df['filter_name']
  

    #define function that collects all calibration frame information
    def get_calibration_data(row, cal_type):
        match = calibration_df[(calibration_df['TYPE'] == cal_type) & (calibration_df['GAIN'] == row['gain'])]
        return match['NUMBER'].sum() if not match.empty else 0

    #add these to the aggregated data frame 
    aggregated['darks'] = aggregated.apply(get_calibration_data, cal_type='DARK', axis=1)
    aggregated['flats'] = aggregated.apply(get_calibration_data, cal_type='FLAT', axis=1)
    aggregated['bias'] = aggregated.apply(get_calibration_data, cal_type='BIAS', axis=1)
    aggregated['flatDarks'] = aggregated.apply(get_calibration_data, cal_type='FLATDARKS', axis=1)
    aggregated['sensorCooling'] = aggregated['sensorCooling'].round(1)
    aggregated['temperature'] = aggregated['temperature'].round(1)
    #rename Code column
    aggregated.rename(columns={'code': 'filter'}, inplace=True)


    #adjust the column order to match that required by AstoBin
    column_order = ['date', 'filter','number', 'duration', 'binning', 'gain', 
                    'sensorCooling', 'darks', 'flats', 'flatDarks', 'bias', 'bortle',
                    'meanSqm', 'meanFwhm', 'temperature']
    #aggregated = aggregated.reindex(columns=column_order)

    return aggregated.reindex(columns=column_order)