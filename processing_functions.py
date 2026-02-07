import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse
from typing import Dict, Union
import numpy as np
import logging

processing_version = '1.4.5'

# Changes:
# Date: Thursday 25th September 2024
# Created SDG
# Date: Sunday 28th September 2024
# Function: aggregate_parameters line 257
# Fixed FutureWarning for start_date/end_date dtype mismatch.
# Original lines assigned datetime.date objects via pd.to_datetime(min(dates).date()),
# causing dtype mismatch with datetime64[ns]. Changed to assign YYYY-MM-DD strings
# directly with strftime, removed redundant reassignment, and converted date-obs to strings.
# Preserves .loc[:, 'column'] indexing for consistency.
# Date: Monday 30th September 2024
# Function: aggregate_parameters line 286-290
# Added handling for condition when sitelat/sitelong are missing or all NaN in aggregated_df.
# Now assigns default values from config if available, else NaN.
#
# Date: Sunday 1st February 2026
# Modification : v1.4.2 Restoration & Logic Overhaul.
# 1. Implemented 'Integer Gain Handshake' for calibration matching.
# 2. Restored Heuristic Date Fallback for missing FITS headers.
# 3. Optimized I/O using Pandas engine for RAID 0 performance.
# Author : SDG & Gemini
#
# Date: Tuesday 3rd February 2026
# Modification: v1.4.3 Fix for date collapsing and calibration mismatch.
#
# Date: Saturday 7th February 2026
# Modification: v1.4.5 Robust date-handling strategy to avoid type conflicts.
# Replacement of v1.4.4 logic with safer pd.api.types check and direct assignment.

def initialize_processing(headers_state: Dict, logger: logging.Logger) -> Dict:

    """

    Initializes the processing state with required dependencies.



    Args:

        headers_state (Dict): Dictionary containing headers state (config, counts, etc.).

        logger (logging.Logger): Logger object for application-wide logging.



    Returns:

        Dict: A state dictionary containing 'headers_state', 'logger', 'dp' (precision), 

              and a placeholder for 'df_not_master'.

    """

    # Log initialization of processing state

    logger.info("Processing initialized")

    # Create and return a dictionary with headers_state, logger, decimal precision (dp), and placeholder for non-master DataFrame

    return {

        'headers_state': headers_state,

        'logger': logger,

        'dp': headers_state['dp'],

        'df_not_master': None

    }



def valid_mean(series: pd.Series) -> float:

    """

    Computes the mean of numeric values in a series, strictly excluding invalid data.

    

    This function handles mixed-type series by coercing non-numeric values (None, NaN, 

    empty strings, or text) to NaN before calculating the mean.



    Args:

        series (pd.Series): The data series to average.



    Returns:

        float: The mean of valid numeric values, or np.nan if no valid data exists.

    """

    try:

        # Convert series to numeric, coerce non-numeric to NaN

        numeric_values = pd.to_numeric(series, errors='coerce')

        # Filter out NaN and return mean; return NaN if no valid values

        valid_values = numeric_values[numeric_values.notna()]

        return valid_values.mean() if not valid_values.empty else np.nan

    except Exception:

        return np.nan



def aggregate_parameters(df: pd.DataFrame, state: Dict) -> pd.DataFrame:

    """

    Aggregates observation parameters from the raw header DataFrame into a session-based summary.

    

    This function performs complex data transformations:

    1. Normalizes 'date-obs' using a robust two-step strategy (Pandas 2.0 compatible).

    2. Calculates observation statistics (sessions, days, period).

    3. Handles custom date-obs shifting for overnight sessions (if USEOBSDATE is False).

    4. Groups and aggregates data by site, filter, and exposure settings.

    5. Casts and rounds final values for AstroBin compatibility.



    Args:

        df (pd.DataFrame): The raw DataFrame containing all extracted FITS/XISF headers.

        state (Dict): Processing state dictionary.



    Returns:

        pd.DataFrame: An aggregated DataFrame ready for summary generation or AstroBin output.



    Raises:

        ValueError: If the input is not a DataFrame or if critical date parsing fails.

    """

    # Extract logger from state dictionary

    logger = state['logger']

    # Log start of parameter aggregation

    logger.info("")

    logger.info("STARTING PARAMETER AGGREGATION")

    

    # Define helper function to replace numeric values with 'None' for rotator column

    def replace_numeric_with_none(value):

        if value == 'None':

            return value

        try:

            float(value)

            return 'None'

        except ValueError:

            return value

    try:

        # Validate that input is a pandas DataFrame

        if not isinstance(df, pd.DataFrame):

            raise ValueError("Input must be a pandas DataFrame")

        # Check if DataFrame is empty

        if df.empty:

            logger.info("No data found")

            return pd.DataFrame()

        # Standardizing column names to lower case

        df = df.copy()

        df.columns = df.columns.str.lower()

        logger.info("Standardized column names to lower case")

        

        # Robust Date-Obs Handling (Step 1): String Standardization

        # We only cast to string if it's NOT already a datetime object.

        # This prevents double-conversion errors and handles high-precision arrays (datetime64[us]) 

        # that Pandas 2.0 might have locked in during CSV injection.

        if 'date-obs' in df.columns:

            if not pd.api.types.is_datetime64_any_dtype(df['date-obs']):

                df['date-obs'] = df['date-obs'].astype(str)

            

        # Calculate observation statistics

        # We define a 'session' as a cluster of frames separated by less than 5 hours.

        threshold = timedelta(hours=5)

        date_strings = df[df['imagetyp'] == 'LIGHT']['date-obs']

        logger.info("Extracted date-obs from DataFrame")

        # Parse date strings into datetime objects

        try:

            dates = [parse(s) for s in date_strings]

            logger.info(f"Parsed {len(dates)} dates")

        except ValueError as e:

            logger.error(f"Error parsing dates: {e}")

            logger.error(f"Date strings: {date_strings}")

            return pd.DataFrame()

        # Sort dates to determine observation period

        try:

            dates.sort()

            logger.info("Sorted dates")

        except TypeError as e:

            logger.error(f"Error sorting dates: {e}")

            return pd.DataFrame()

        # Count observation sessions based on time gaps

        sessions = 1

        for i in range(1, len(dates)):

            if dates[i] - dates[i-1] > threshold:

                sessions += 1

        logger.info(f"Number of sessions: {sessions}")

        # Set default date for error cases (1900-01-01 is used as a 'No Data' flag)

        default_date = '1900-01-01'

        # Add session and date columns to DataFrame

        try:

            df.loc[:, 'sessions'] = sessions

            # Fix: Store start_date and end_date as strings to avoid datetime.date/datetime64 conversion issues

            df.loc[:, 'start_date'] = min(dates).strftime('%Y-%m-%d') if dates else default_date

            df.loc[:, 'end_date'] = max(dates).strftime('%Y-%m-%d') if dates else default_date

            df.loc[:, 'num_days'] = (max(dates).date() - min(dates).date()).days + 1 if dates else 0

            logger.info(f"Start date: {df['start_date'].iloc[0]}, End date: {df['end_date'].iloc[0]}, Number of days: {df['num_days'].iloc[0]}")

        except Exception as e:

            logger.error(f"Error setting date parameters: {e}")

            df.loc[:, 'sessions'] = 0

            df.loc[:, 'start_date'] = default_date

            df.loc[:, 'end_date'] = default_date

            df.loc[:, 'num_days'] = 0



        # Robust Date-Obs Handling (Step 2): Safe Datetime Conversion

        try:

            # Direct assignment (df['date-obs'] = ...) without .loc allows Pandas to change 

            # the column dtype freely, which is critical for resolving Dtype conflicts.

            df['date-obs'] = pd.to_datetime(df['date-obs'], errors='coerce')

            if df['date-obs'].isna().all():

                logger.warning("All dates failed parsing. Using default_date.")

                df['date-obs'] = pd.to_datetime(default_date)

        except Exception as e:

            logger.error(f"Error converting date-obs to datetime: {e}")

            df['date-obs'] = pd.to_datetime(default_date)



        # Custom Overnight Date Shifting

        # If USEOBSDATE is False, we group frames taken after midnight with the previous day's session

        # to ensure the whole night of imaging is counted under a single date.

        if not state['headers_state']['useobsdate']:

            logger.info("Use observation date is False. Proceeding with custom date handling")

            try:

                threshold = pd.Timedelta(hours=5)

                df = df.sort_values(by='date-obs', ascending=True).reset_index(drop=True)

                logger.info("Sorted DataFrame by 'date-obs'")

                df.loc[:, 'new-date-obs'] = df['date-obs']

                ref_datetime = df['date-obs'][0]

                midnight = pd.Timestamp("00:00:00").time()

                midday = pd.Timestamp("12:00:00").time()

                ref_date = ref_datetime.date() - timedelta(days=1) if midnight < ref_datetime.time() < midday else ref_datetime.date()

                logger.info(f"Reference date set to: {ref_date}")

                for n in range(1, len(df['date-obs'])):

                    current_date = df['date-obs'][n]

                    previous_date = df['date-obs'][n-1]

                    date_diff = current_date - previous_date

                    logger.debug(f"Processing row {n}: current_date={current_date}, previous_date={previous_date}, date_diff={date_diff}")

                    if date_diff > threshold:

                        ref_date = current_date.date()

                        logger.info(f"Date difference ({date_diff}) exceeded threshold. Updated reference date to {ref_date}")

                    df.loc[n, 'new-date-obs'] = ref_date

                    logger.debug(f"Updated 'new-date-obs' for row {n} to {ref_date}")

                df = df.drop(columns=['date-obs'], errors='ignore')

                df = df.rename(columns={'new-date-obs': 'date-obs'})

                logger.info("Replaced original 'date-obs' with updated values")

            except Exception as e:

                logger.error(f"Error in custom date handling: {e}")



        # Final Formatting for Aggregation

        df.loc[:, 'date-obs'] = pd.to_datetime(df['date-obs']).dt.strftime('%Y-%m-%d')

        logger.info("Converted 'date-obs' column to YYYY-MM-DD string format")

        

        # Split DataFrame into MASTER calibration files and individual frames

        df_master = df[df['imagetyp'].str.contains('MASTER') & ~df['imagetyp'].str.contains('MASTERLIGHT')]

        df_not_master = df[~df['imagetyp'].str.contains('MASTER')]

        state['df_not_master'] = df_not_master

        logger.info(f"Split DataFrame: {len(df_master)} rows in 'MASTER' and {len(df_not_master)} rows in 'not MASTER'")



        # Aggregation Dictionary: Defines how each column is summarized during the groupby operation.

        agg_dict = {

            'ccd-temp': ('ccd-temp', 'mean'),

            'temp_min': ('foctemp', 'min'),

            'temp_max': ('foctemp', 'max'),

            'fwhm': ('fwhm', 'mean'),

            'sitelat': ('sitelat', valid_mean),

            'sitelong': ('sitelong', valid_mean),

            'focratio': ('focratio', 'mean'),

            'foctemp': ('foctemp', 'mean'),

            'focallen': ('focallen', 'first'),

            'bortle': ('bortle', 'mean'),

            'sqm': ('sqm', 'mean'),

            'hfr': ('hfr', 'mean'),

            'xpixsz': ('xpixsz', 'first'),

            'egain': ('egain', 'mean'),

            'instrume': ('instrume', 'first'),

            'imscale': ('imscale', 'mean'),

            'telescop': ('telescop', 'first'),

            'focname': ('focname', 'first'),

            'fwheel': ('fwheel', 'first'),

            'rotname': ('rotname', 'first'),

            'rotantang':('rotantang','first'),

            'target': ('object', 'first'),

            'swcreate': ('swcreate', 'first'),

            'filename': ('filename', 'first'),

            'sessions': ('sessions', 'first'),

            'start_date': ('start_date', 'first'),

            'end_date': ('end_date', 'first'),

            'num_days': ('num_days', 'first')

        }

        # Aggregate non-MASTER DataFrame by key observation parameters

        try:

            agg_not_master_df = df_not_master.groupby(['site', 'date-obs', 'imagetyp', 'filter', 'gain', 'xbinning', 'exposure', 'object']).agg(

                number=('date-obs', 'count'),

                **agg_dict

            ).reset_index()

            logger.info("Aggregated non-MASTER DataFrame")

        except Exception as e:

            logger.error(f"Error aggregating non-MASTER DataFrame: {e}")

            return pd.DataFrame()

        # Combine master and non-master results

        if df_master.empty:

            logger.info("No MASTER data found")

            aggregated_df = agg_not_master_df

        else:

            df_master = df_master.copy()

            df_master.loc[:, 'imscale'] = df_not_master['imscale']

            aggregated_df = pd.concat([df_master, agg_not_master_df])

            logger.info("Concatenated MASTER and non-MASTER results")

        # Rename columns to match internal naming conventions for the summary and AstroBin

        aggregated_df = aggregated_df.rename(columns={

            'imagetyp': 'imageType',

            'ccd-temp': 'sensorCooling',

            'xbinning': 'binning',

            'exposure': 'duration',

            'fwhm': 'meanFwhm',

            'focratio': 'fNumber',

            'foctemp': 'temperature',

            'focname': 'focuser',

            'fwheel': 'filterWheel',

            'telescop': 'telescope',

            'instrume': 'camera',

            'sqm': 'meanSqm',

            'focallen': 'focalLength',

            'xpixsz': 'pixelSize'

        })

        logger.info("Renamed columns for clarity")

        

        # Numeric Data Hardening: Ensures all summarized columns are correct types and rounded.

        try:

            aggregated_df.loc[:, 'sensorCooling'] = aggregated_df['sensorCooling'].astype(float).round().astype(int)

            aggregated_df.loc[:, 'temp_min'] = aggregated_df['temp_min'].astype(float)

            aggregated_df.loc[:, 'temp_max'] = aggregated_df['temp_max'].astype(float)

            aggregated_df.loc[:, 'temperature'] = aggregated_df['temperature'].astype(float)

            aggregated_df.loc[:, 'meanFwhm'] = aggregated_df['meanFwhm'].astype(float)

            aggregated_df.loc[:, 'sitelat'] = aggregated_df['sitelat'].astype(float)

            aggregated_df.loc[:, 'sitelong'] = aggregated_df['sitelong'].astype(float)

            aggregated_df.loc[:, 'fNumber'] = aggregated_df['fNumber'].astype(float)

            aggregated_df.loc[:, 'bortle'] = aggregated_df['bortle'].astype(float)

            aggregated_df.loc[:, 'meanSqm'] = aggregated_df['meanSqm'].astype(float)

            aggregated_df.loc[:, 'egain'] = aggregated_df['egain'].astype(float)

            aggregated_df.loc[:, 'camera'] = aggregated_df['camera'].astype(str)

            aggregated_df.loc[:, 'telescope'] = aggregated_df['telescope'].astype(str)

            aggregated_df.loc[:, 'focuser'] = aggregated_df['focuser'].astype(str)

            aggregated_df.loc[:, 'filterWheel'] = aggregated_df['filterWheel'].astype(str)

            aggregated_df.loc[:, 'rotname'] = aggregated_df['rotname'].astype(str)

            aggregated_df.loc[:, 'rotantang'] = aggregated_df['rotatang'].astype(str)

            aggregated_df.loc[:, 'target'] = aggregated_df['target'].astype(str)

            aggregated_df.loc[:, 'swcreate'] = aggregated_df['swcreate'].astype(str)

            aggregated_df.loc[:, 'duration'] = aggregated_df['duration'].astype(float)

            aggregated_df.loc[:, 'focalLength'] = aggregated_df['focalLength'].astype(float)

            aggregated_df.loc[:, 'pixelSize'] = aggregated_df['pixelSize'].astype(float)

            aggregated_df.loc[:, 'imageType'] = aggregated_df['imageType'].astype(str)

            aggregated_df.loc[:, 'sessions'] = aggregated_df['sessions'].astype(int)

            aggregated_df.loc[:, 'num_days'] = aggregated_df['num_days'].astype(int)

        except Exception as e:

            logger.error(f"Error converting data types: {e}")

        # Round values based on project standards

        aggregated_df = aggregated_df.round({

            'temp_min': 2,

            'temp_max': 2,

            'temperature': 2,

            'meanFwhm': 2,

            'sitelat': state['dp'],

            'sitelong': state['dp'],

            'fNumber': 2,

            'bortle': 2,

            'meanSqm': 2,

            'duration': 2

        })

        # Handle cases where site coordinates were missing across the entire dataset.

        if aggregated_df['sitelat'].isna().all():

            aggregated_df.loc[:, 'sitelat'] = state['headers_state']['config']['defaults'].get('SITELAT', np.nan)

        if aggregated_df['sitelong'].isna().all():

            aggregated_df.loc[:, 'sitelong'] = state['headers_state']['config']['defaults'].get('SITELONG', np.nan)

        logger.info("Rounded and cast values to correct type")

        

        logger.info(f"aggregated_df keys: {list(aggregated_df.keys())}")

        logger.info(f"Number of observation sessions: {aggregated_df['sessions'].iloc[0]}")

        logger.info(f"start_date: {aggregated_df['start_date'].iloc[0]}")

        logger.info(f"end_date: {aggregated_df['end_date'].iloc[0]}")

        logger.info(f"num_days: {aggregated_df['num_days'].iloc[0]}")

        logger.info("")

        return aggregated_df

    except Exception as e:

        logger.error(f"Failed to aggregate parameters: {e}")

        return pd.DataFrame()



def update_filter(filter_value: str, filter_to_code: Dict[str, str], logger: logging.Logger) -> Union[int, str]:

    """

    Maps a filter name (e.g., 'Ha') to its corresponding AstroBin numeric code.



    Args:

        filter_value (str): The filter name found in the FITS header.

        filter_to_code (Dict[str, str]): Mapping dictionary from config.ini.

        logger (logging.Logger): Logger object.



    Returns:

        Union[int, str]: The numeric code if found, otherwise an error string.

    """

    # Log start of filter update

    logger.info("")

    logger.info("STARTING FILTER UPDATE")

    logger.info(f"Updating filter: {filter_value}")



    try:

        # Validate that filter_value is a non-empty string

        if not isinstance(filter_value, str) or not filter_value:

            raise ValueError("Filter value must be a non-empty string")



        # Look up filter code in dictionary

        code = filter_to_code.get(filter_value)

        # Attempt to convert code to integer

        try:

            code = int(code)

        except (ValueError, TypeError):

            code = None

            logger.warning(f"Failed to convert code to integer for filter: {filter_value}")



        # Check if code is a valid integer within range

        if isinstance(code, int) and 1000 <= code <= 99999:

            logger.info(f"Valid code found for filter: {filter_value}")

            return code

        else:

            logger.warning(f"No code found for filter: {filter_value}. Enter a valid code in filters.csv or astrobin upload file")

            return f"{filter_value}: no code found"



    except Exception as e:

        logger.error(f"Error updating filter {filter_value}: {e}")

        return f"{filter_value}: error"



def create_astrobin_output(df: pd.DataFrame, state: Dict) -> pd.DataFrame:

    """

    Transforms the aggregated DataFrame into the final format required by AstroBin's CSV importer.

    

    Processing includes:

    1. Normalization of Gain and Cooling to rounded integers (Integer Handshake).

    2. Mapping of Light frames to their corresponding calibration counts (Darks, Flats, Bias).

    3. Matching calibration frames by Gain (for darks/bias) or Filter (for flats).

    4. Mapping filter names to AstroBin numeric IDs.

    5. Reordering columns to strict CSV specification.



    Args:

        df (pd.DataFrame): The aggregated DataFrame.

        state (Dict): Processing state dictionary.



    Returns:

        pd.DataFrame: A DataFrame formatted exactly for AstroBin upload.

    """

    # Extract logger from state

    logger = state['logger']

    # Log start of AstroBin output creation

    logger.info("")

    logger.info("STARTING ASTROBIN OUTPUT CREATION")



    try:

        # Validate that input is a pandas DataFrame

        if not isinstance(df, pd.DataFrame):

            raise ValueError("Input must be a pandas DataFrame")

        # Check if DataFrame is empty

        if df.empty:

            logger.info("No data found")

            return pd.DataFrame()



        # Normalization: Integer Gain Handshake

        # Force gain and sensorCooling to be integers for matching, resolving float/int mismatch bugs.

        for col in ['gain', 'sensorCooling']:

            if col in df.columns:

                try:

                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round().astype(int)

                    logger.info(f"Normalized {col} to integer values for matching.")

                except Exception as e:

                    logger.warning(f"Could not normalize {col}: {e}")



        # Get filter dictionary from configuration

        filters_dict = state['headers_state']['config']['filters']

        # Initialize empty DataFrame for output

        light_group_agg = pd.DataFrame()

        logger.info("Data found, processing")



        # Extract LIGHT frames as the base for the acquisition table

        light_group = df[df['imageType'] == 'LIGHT'].copy()

        logger.info("Created 'LIGHT' DataFrame copy")



        # Ensure filter column is string type for mapping and matching

        light_group['filter'] = light_group['filter'].astype(str)



        # Calibration Frame Mapping:

        # We look up the count ('number') for each calibration type and match it to Light frames.

        imagetyp_to_column = {

            'BIAS': 'bias',

            'DARK': 'darks',

            'DARKFLAT': 'flatDarks',

            'FLAT': 'flats'

        }



        for imagetyp, column in imagetyp_to_column.items():

            logger.info(f"Processing image type: {imagetyp}")

            try:

                # Extract frames for the current image type

                sub_group = df[df['imageType'] == imagetyp]

                # Match Darks/Bias by Gain, Flats by Filter name.

                light_group[column] = light_group.apply(

                    lambda x: sub_group[sub_group['gain' if imagetyp != 'FLAT' else 'filter'] == x['gain' if imagetyp != 'FLAT' else 'filter']]['number'].sum(),

                    axis=1

                )

                logger.info(f"Updated 'light_group' column: {column}")

            except Exception as e:

                logger.error(f"Error processing image type {imagetyp}: {e}")



        # MASTER Calibration Mapping:

        # Similar to above, but specifically for Master files if they were identified in the scan.

        master_imagetyp_to_column = {

            'MASTERBIAS': 'bias',

            'MASTERDARK': 'darks',

            'MASTERDARKFLAT': 'flatDarks',

            'MASTERFLAT': 'flats'

        }



        for imagetyp, column in master_imagetyp_to_column.items():

            logger.info(f"Processing MASTER frames for image type: {imagetyp}")

            try:

                sub_group = df[df['imageType'] == imagetyp]

                if not sub_group.empty:

                    light_group[column] = light_group.apply(

                        lambda x: sub_group[sub_group['gain' if imagetyp != 'MASTERFLAT' else 'filter'] == x['gain' if imagetyp != 'MASTERFLAT' else 'filter']]['number'].sum(),

                        axis=1

                    )

                    logger.info(f"Updated 'light_group' column with MASTER frame number: {column}")

            except Exception as e:

                logger.error(f"Error processing MASTER frames for {imagetyp}: {e}")



        try:

            # Filter Mapping:

            # Replaces the descriptive filter name (e.g. 'Ha') with the numeric ID (e.g. '4663').

            original_filters = [f.strip() for f in light_group['filter'].unique()]

            light_group['filter'] = light_group['filter'].apply(lambda x: filters_dict.get(x.strip(), x.strip()))

            logger.info("Mapped filter names to codes:")

            for original_filter in original_filters:

                mapped_code = filters_dict.get(original_filter, original_filter)

                logger.info(f"{original_filter} -> {mapped_code}")



            # Strict AstroBin Column Order

            column_order = [

                'date', 'filter', 'number', 'duration', 'binning', 'gain',

                'sensorCooling', 'fNumber', 'darks', 'flats', 'flatDarks', 'bias',

                'bortle', 'meanSqm', 'meanFwhm', 'temperature'

            ]

            # Rename date-obs to date for the header

            light_group.rename(columns={'date-obs': 'date'}, inplace=True)

            light_group_agg = light_group[column_order]

            logger.info("Ordered columns to meet AstroBin requirements")

            logger.info("COMPLETED ASTROBIN OUTPUT CREATION")

        except Exception as e:

            logger.error(f"Error processing filter mappings: {e}")

            return pd.DataFrame()



        return light_group_agg



    except Exception as e:

        logger.error(f"Failed to create AstroBin output: {e}")

        return pd.DataFrame()
