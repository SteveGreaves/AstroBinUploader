import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse
from typing import Dict, Union

def initialize_processing(headers_state: Dict, logger) -> Dict:
    """
    Initializes processing state.

    Parameters:
    headers_state (Dict): Dictionary containing headers state from headers_functions.
    logger: Logger object for logging messages.

    Returns:
    Dict: Dictionary containing initialized processing state.
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

def aggregate_parameters(df: pd.DataFrame, state: Dict) -> pd.DataFrame:
    """
    Aggregates parameters from the given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame to aggregate parameters from.
    state (Dict): Dictionary containing processing state.

    Returns:
    pd.DataFrame: DataFrame with aggregated parameters.

    Raises:
    ValueError: If DataFrame is invalid or date parsing fails.
    """
    # Extract logger from state dictionary
    logger = state['logger']
    # Log start of parameter aggregation
    logger.info("")
    logger.info("STARTING PARAMETER AGGREGATION")

    # Define helper function to replace numeric values with 'None' for rotator column
    def replace_numeric_with_none(value):
        # Return 'None' if value is already 'None'
        if value == 'None':
            return value
        # Try to convert value to float; if successful, return 'None'
        try:
            float(value)
            return 'None'
        except ValueError:
            # Return original value if not numeric
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
        # Create a copy to avoid modifying the input DataFrame
        df = df.copy()
        # Convert all column names to lowercase for consistency
        df.columns = df.columns.str.lower()
        logger.info("Standardized column names to lower case")

        # Calculate observation statistics
        # Define threshold for separating observation sessions (5 hours)
        threshold = timedelta(hours=5)
        # Extract date-obs values for LIGHT frames
        date_strings = df[df['imagetyp'] == 'LIGHT']['date-obs']
        logger.info("Extracted date-obs from DataFrame")

        # Parse date strings into datetime objects
        try:
            dates = [parse(s) for s in date_strings]
            logger.info(f"Parsed {len(dates)} dates")
        except ValueError as e:
            # Log error and date strings if parsing fails
            logger.error(f"Error parsing dates: {e}")
            logger.error(f"Date strings: {date_strings}")
            return pd.DataFrame()

        # Sort dates to determine observation period
        try:
            dates.sort()
            logger.info("Sorted dates")
        except TypeError as e:
            # Log error if sorting fails (e.g., invalid date types)
            logger.error(f"Error sorting dates: {e}")
            return pd.DataFrame()

        # Count observation sessions based on time gaps
        sessions = 1
        for i in range(1, len(dates)):
            if dates[i] - dates[i-1] > threshold:
                sessions += 1
        logger.info(f"Number of sessions: {sessions}")

        # Set default date for error cases
        default_date = pd.to_datetime('1900-01-01')
        # Add session and date columns to DataFrame
        try:
            df.loc[:, 'sessions'] = sessions
            df.loc[:, 'start_date'] = pd.to_datetime(min(dates).date())
            df.loc[:, 'end_date'] = pd.to_datetime(max(dates).date())
            df.loc[:, 'num_days'] = (max(dates).date() - min(dates).date()).days + 1
            logger.info(f"Start date: {df['start_date'].iloc[0]}, End date: {df['end_date'].iloc[0]}, Number of days: {df['num_days'].iloc[0]}")
        except Exception as e:
            # Log error and set default values if date setting fails
            logger.error(f"Error setting date parameters: {e}")
            df.loc[:, 'sessions'] = 0
            df.loc[:, 'start_date'] = default_date
            df.loc[:, 'end_date'] = default_date
            df.loc[:, 'num_days'] = 0

        # Convert date-obs to datetime, handling multiple formats
        try:
            # Try standard format first
            df.loc[:, 'date-obs'] = pd.to_datetime(df['date-obs'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
            if df['date-obs'].isna().any():
                # Try format with microseconds if standard format fails
                try:
                    df.loc[:, 'date-obs'] = pd.to_datetime(df['date-obs'], format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce')
                except ValueError:
                    # Fall back to inferred format if both fail
                    df.loc[:, 'date-obs'] = pd.to_datetime(df['date-obs'], infer_datetime_format=True, errors='coerce')
        except Exception as e:
            # Log error and set default date if conversion fails
            logger.error(f"Error converting date-obs to datetime: {e}")
            df.loc[:, 'date-obs'] = default_date

        # Handle custom date processing if useobsdate is False
        if not state['headers_state']['useobsdate']:
            logger.info("Use observation date is False. Proceeding with custom date handling")
            try:
                # Define threshold for date grouping
                threshold = pd.Timedelta(hours=5)
                # Sort DataFrame by date-obs
                df = df.sort_values(by='date-obs', ascending=True).reset_index(drop=True)
                logger.info("Sorted DataFrame by 'date-obs'")

                # Initialize new-date-obs with original dates
                df.loc[:, 'new-date-obs'] = df['date-obs']
                # Get reference datetime from first row
                ref_datetime = df['date-obs'][0]
                # Define time boundaries for date adjustment
                midnight = pd.Timestamp("00:00:00").time()
                midday = pd.Timestamp("12:00:00").time()

                # Adjust reference date based on time of day
                ref_date = ref_datetime.date() - timedelta(days=1) if midnight < ref_datetime.time() < midday else ref_datetime.date()
                logger.info(f"Reference date set to: {ref_date}")

                # Update dates based on session gaps
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

                # Replace original date-obs with new values
                df = df.drop(columns=['date-obs'], errors='ignore')
                df = df.rename(columns={'new-date-obs': 'date-obs'})
                logger.info("Replaced original 'date-obs' with updated values")
            except Exception as e:
                logger.error(f"Error in custom date handling: {e}")

        # Convert date-obs to date format
        df.loc[:, 'date-obs'] = pd.to_datetime(df['date-obs']).dt.date
        logger.info("Converted 'date-obs' column to datetime.date format")

        # Split DataFrame into MASTER and non-MASTER
        df_master = df[df['imagetyp'].str.contains('MASTER') & ~df['imagetyp'].str.contains('MASTERLIGHT')]
        df_not_master = df[~df['imagetyp'].str.contains('MASTER')]
        # Store non-master DataFrame in state
        state['df_not_master'] = df_not_master
        logger.info(f"Split DataFrame: {len(df_master)} rows in 'MASTER' and {len(df_not_master)} rows in 'not MASTER'")

        # Define aggregation dictionary for non-master DataFrame
        agg_dict = {
            'ccd-temp': ('ccd-temp', 'mean'),
            'temp_min': ('foctemp', 'min'),
            'temp_max': ('foctemp', 'max'),
            'fwhm': ('fwhm', 'mean'),
            'sitelat': ('sitelat', 'mean'),
            'sitelong': ('sitelong', 'mean'),
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
            'rotator': ('rotator', 'first'),
            'target': ('object', 'first'),
            'swcreate': ('swcreate', 'first'),
            'filename': ('filename', 'first'),
            'sessions': ('sessions', 'first'),
            'start_date': ('start_date', 'first'),
            'end_date': ('end_date', 'first'),
            'num_days': ('num_days', 'first')
        }

        # Aggregate non-MASTER DataFrame
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

        # Rename columns for output
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

        # Force numeric data types
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
            aggregated_df.loc[:, 'rotator'] = aggregated_df['rotator'].astype(str)
            aggregated_df.loc[:, 'target'] = aggregated_df['target'].astype(str)
            aggregated_df.loc[:, 'swcreate'] = aggregated_df['swcreate'].astype(str)
            aggregated_df.loc[:, 'duration'] = aggregated_df['duration'].astype(float)
            aggregated_df.loc[:, 'focalLength'] = aggregated_df['focalLength'].astype(float)
            aggregated_df.loc[:, 'pixelSize'] = aggregated_df['pixelSize'].astype(float)
            aggregated_df.loc[:, 'imageType'] = aggregated_df['imageType'].astype(str)
            aggregated_df.loc[:, 'sessions'] = aggregated_df['sessions'].astype(int)
            aggregated_df.loc[:, 'start_date'] = pd.to_datetime(aggregated_df['start_date']).dt.date
            aggregated_df.loc[:, 'end_date'] = pd.to_datetime(aggregated_df['end_date']).dt.date
            aggregated_df.loc[:, 'num_days'] = aggregated_df['num_days'].astype(int)
        except Exception as e:
            logger.error(f"Error converting data types: {e}")

        # Round values
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
        logger.info("Rounded and cast values to correct type")

        aggregated_df.loc[:, 'rotator'] = aggregated_df['rotator'].apply(replace_numeric_with_none)
        logger.info("Processed rotator column")

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

def update_filter(filter_value: str, filter_to_code: Dict[str, str], logger) -> Union[int, str]:
    """
    Maps a filter name to its corresponding code.

    Parameters:
    filter_value (str): Name of the filter.
    filter_to_code (Dict[str, str]): Dictionary mapping filter names to codes.
    logger: Logger object for logging messages.

    Returns:
    Union[int, str]: Filter code or error message.

    Raises:
    ValueError: If filter_value is not a string or empty.
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
    Creates an output DataFrame for AstroBin.

    Parameters:
    df (pd.DataFrame): Input DataFrame to process.
    state (Dict): Dictionary containing processing state.

    Returns:
    pd.DataFrame: Processed DataFrame for AstroBin.

    Raises:
    ValueError: If DataFrame is invalid.
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

        # Get filter dictionary from configuration
        filters_dict = state['headers_state']['config']['filters']
        # Initialize empty DataFrame for output
        light_group_agg = pd.DataFrame()
        logger.info("Data found, processing")

        # Extract LIGHT frames
        light_group = df[df['imageType'] == 'LIGHT'].copy()
        logger.info("Created 'LIGHT' DataFrame copy")

        # Define mapping of image types to calibration columns
        imagetyp_to_column = {
            'BIAS': 'bias',
            'DARK': 'darks',
            'DARKFLAT': 'flatDarks',
            'FLAT': 'flats'
        }

        # Process calibration frame counts
        for imagetyp, column in imagetyp_to_column.items():
            logger.info(f"Processing image type: {imagetyp}")
            try:
                # Extract frames for the current image type
                sub_group = df[df['imageType'] == imagetyp]
                # Add calibration frame counts to LIGHT DataFrame
                light_group[column] = light_group.apply(
                    lambda x: sub_group[sub_group['gain' if imagetyp != 'FLAT' else 'filter'] == x['gain' if imagetyp != 'FLAT' else 'filter']]['number'].sum(),
                    axis=1
                )
                logger.info(f"Updated 'light_group' column: {column}")
            except Exception as e:
                logger.error(f"Error processing image type {imagetyp}: {e}")

        # Define mapping of master image types to calibration columns
        master_imagetyp_to_column = {
            'MASTERBIAS': 'bias',
            'MASTERDARK': 'darks',
            'MASTERDARKFLAT': 'flatDarks',
            'MASTERFLAT': 'flats'
        }

        # Process master calibration frame counts
        for imagetyp, column in master_imagetyp_to_column.items():
            logger.info(f"Processing MASTER frames for image type: {imagetyp}")
            try:
                # Extract master frames for the current image type
                sub_group = df[df['imageType'] == imagetyp]
                if not sub_group.empty:
                    # Add master frame counts to LIGHT DataFrame
                    light_group[column] = light_group.apply(
                        lambda x: sub_group[sub_group['gain' if imagetyp != 'MASTERFLAT' else 'filter'] == x['gain' if imagetyp != 'MASTERFLAT' else 'filter']]['number'].sum(),
                        axis=1
                    )
                    logger.info(f"Updated 'light_group' column with MASTER frame number: {column}")
            except Exception as e:
                logger.error(f"Error processing MASTER frames for {imagetyp}: {e}")

        # Check if filter column is of object type
        if light_group['filter'].dtype == 'object':
            try:
                # Get unique filters
                original_filters = [f.strip() for f in light_group['filter'].unique()]
                # Map filter names to codes using dictionary
                light_group['filter'] = light_group['filter'].apply(lambda x: filters_dict.get(x.strip(), x.strip()))
                logger.info("Mapped filter names to codes:")
                for original_filter in original_filters:
                    mapped_code = filters_dict.get(original_filter, original_filter)
                    logger.info(f"{original_filter} -> {mapped_code}")

                # Define column order for AstroBin output
                column_order = [
                    'date', 'filter', 'number', 'duration', 'binning', 'gain',
                    'sensorCooling', 'fNumber', 'darks', 'flats', 'flatDarks', 'bias',
                    'bortle', 'meanSqm', 'meanFwhm', 'temperature'
                ]
                # Rename date-obs to date
                light_group.rename(columns={'date-obs': 'date'}, inplace=True)
                # Select ordered columns
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