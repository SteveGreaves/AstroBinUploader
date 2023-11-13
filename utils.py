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