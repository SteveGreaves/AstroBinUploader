import pandas as pd
import numpy as np
import requests
import math
import sys
from geopy.geocoders import Nominatim
from typing import Tuple, Optional, Dict
#
# Date: Sunday 1st February 2026
# Modification : v1.4.2 Restoration & Logic Overhaul.
# 1. Implemented 'Integer Gain Handshake' for calibration matching.
# 2. Restored Heuristic Date Fallback for missing FITS headers.
# 3. Optimized I/O using Pandas engine for RAID 0 performance.
# Author : SDG & Gemini

def initialize_sites(headers_state: Dict, logger) -> Dict:
    """Initializes the sites processing state for handling location data.

    Sets up a dictionary containing the headers state, logger, existing sites DataFrame, geolocator,
    and placeholder for location information.

    Args:
        headers_state (Dict): Dictionary containing headers state from headers_functions, including
            configuration with 'secret' and 'sites' sections.
        logger: Logger object for logging messages.

    Returns:
        Dict: Dictionary containing initialized sites state with keys:
            - headers_state: The input headers state.
            - logger: The logger object.
            - existing_sites_df: DataFrame of existing sites from configuration.
            - geolocator: Nominatim geolocator instance for reverse geocoding.
            - location_info: Placeholder for location data (initially None).

    Raises:
        ValueError: If headers_state is not a dictionary or logger is invalid.
        KeyError: If required configuration keys (e.g., 'secret', 'sites') are missing.
    """
    # Log start of sites initialization
    logger.info("")
    logger.info("INITIALIZING SITES")

    try:
        # Extract email address from configuration for geolocator user agent
        email_address = headers_state['config']['secret']['EMAIL_ADDRESS'].strip()
        # Create DataFrame from existing sites in configuration
        existing_sites_df = pd.DataFrame(headers_state['config']['sites'])
        logger.info("Existing sites DataFrame created")
    except Exception as e:
        # Handle errors (e.g., missing 'secret' or 'sites' keys) by initializing an empty DataFrame
        existing_sites_df = pd.DataFrame()
        logger.warning(f"Failed to create existing sites DataFrame, initialized as empty: {str(e)}")

    try:
        # Initialize Nominatim geolocator with user agent based on email address
        geolocator = Nominatim(user_agent=f"AstroBinUpload.py_{email_address}")
        logger.info("Geolocator initialized")
    except Exception as e:
        # Handle geolocator initialization errors (e.g., invalid email) and set to None
        logger.error(f"Failed to initialize geolocator: {str(e)}")
        geolocator = None

    # Return initialized sites state dictionary
    return {
        'headers_state': headers_state,
        'logger': logger,
        'existing_sites_df': existing_sites_df,
        'geolocator': geolocator,
        'location_info': None
    }

def find_location_by_coords(sites_df: pd.DataFrame, lat_long_pair: Tuple[float, float], logger) -> Optional[pd.DataFrame]:
    """Finds a location in the sites DataFrame matching the given latitude and longitude pair.

    Matches coordinates by rounding to one decimal place for approximate equality.

    Args:
        sites_df (pd.DataFrame): DataFrame containing site data with 'latitude' and 'longitude' columns.
        lat_long_pair (Tuple[float, float]): Tuple of (latitude, longitude) to match.
        logger: Logger object for logging messages.

    Returns:
        Optional[pd.DataFrame]: DataFrame of matching rows, or None if no match is found.

    Raises:
        ValueError: If sites_df is not a pandas DataFrame or lat_long_pair is not a valid tuple.
    """
    try:
        # Validate that sites_df is a pandas DataFrame
        if not isinstance(sites_df, pd.DataFrame):
            raise ValueError("sites_df must be a pandas DataFrame")
        # Validate that lat_long_pair is a tuple with two elements
        if not isinstance(lat_long_pair, tuple) or len(lat_long_pair) != 2:
            raise ValueError("lat_long_pair must be a tuple of two floats")

        # Extract target latitude and longitude
        target_lat, target_long = lat_long_pair
        # Find rows where latitude and longitude match within one decimal place
        matching_rows = sites_df[
            (np.ceil(sites_df['latitude'] * 10) / 10 == np.ceil(target_lat * 10) / 10) &
            (np.ceil(sites_df['longitude'] * 10) / 10 == np.ceil(target_long * 10) / 10)
        ]

        # Check if matching rows were found
        if not matching_rows.empty:
            logger.debug(f"Matching location found: Site {matching_rows.index[0]}, "
                       f"Latitude {matching_rows['latitude'].iloc[0]}, "
                       f"Longitude {matching_rows['longitude'].iloc[0]}")
            return matching_rows
        # Return None if no match is found
        logger.info(f"No matching location found for coordinates: {lat_long_pair}")
        return None

    except Exception as e:
        # Log any errors during location matching (e.g., invalid DataFrame columns)
        logger.error(f"Error finding location by coordinates {lat_long_pair}: {str(e)}")
        return None

def get_bortle_sqm(lat: float, lon: float, api_key: str, api_endpoint: str, logger) -> Tuple[int, float, Optional[str], bool, bool]:
    """Retrieves Bortle scale and SQM value for given coordinates using an external API.

    Makes an API request to retrieve artificial brightness, calculates SQM, and converts to Bortle scale.

    Args:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
        api_key (str): API key for the geolocation service.
        api_endpoint (str): API endpoint URL for the geolocation service.
        logger: Logger object for logging messages.

    Returns:
        Tuple[int, float, Optional[str], bool, bool]: Tuple containing:
            - Bortle scale (int): Bortle class (1-9).
            - SQM value (float): Sky Quality Meter value in mag/arcsec².
            - Error message (Optional[str]): Error message if API call fails, else None.
            - API key validity (bool): Whether the API key is valid.
            - Endpoint validity (bool): Whether the API endpoint is valid.

    Raises:
        ValueError: If lat, lon, api_key, or api_endpoint are invalid.
    """
    # Log start of Bortle and SQM retrieval
    logger.info("")
    logger.info("GETTING BORTLE SCALE AND SQM VALUE")
    logger.info(f"Retrieving Bortle scale and SQM for coordinates ({lat}, {lon})")

    # Define helper function to validate API key (16 alphanumeric characters)
    def is_valid_api_key(api_key: str) -> bool:
        return api_key is not None and len(api_key) == 16 and api_key.isalnum()

    # Define helper function to validate API endpoint (non-empty string)
    def is_valid_api_endpoint(api_endpoint: str) -> bool:
        return bool(api_endpoint and api_endpoint.strip())

    try:
        # Validate input types
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise ValueError("Latitude and longitude must be numbers")
        if not isinstance(api_key, str) or not isinstance(api_endpoint, str):
            raise ValueError("API key and endpoint must be strings")

        # Check API key and endpoint validity
        api_valid = is_valid_api_key(api_key)
        api_endpoint_valid = is_valid_api_endpoint(api_endpoint)

        # Handle invalid API key and endpoint cases
        if not api_valid and not api_endpoint_valid:
            logger.error("Both API key and API endpoint are invalid")
            return 0, 0, "Both API key and API endpoint are invalid", api_valid, api_endpoint_valid
        elif not api_valid:
            logger.error("API key is malformed")
            return 0, 0, "API key is malformed", api_valid, api_endpoint_valid
        elif not api_endpoint_valid:
            logger.error("API endpoint is empty")
            return 0, 0, "API endpoint is empty", api_valid, api_endpoint_valid

        # Prepare API request parameters
        params = {'ql': 'wa_2015', 'qt': 'point', 'qd': f'{lon},{lat}', 'key': api_key}
        logger.info("Sending request to API endpoint")
        # Make API request
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()

        # Check for authentication error
        if response.text.strip() == 'Invalid authentication.':
            logger.error("Authentication error: Missing or invalid API key")
            return 0, 0, "Authentication error: Missing or invalid API key", False, api_endpoint_valid

        # Parse artificial brightness from response
        artificial_brightness = float(response.text)
        # Calculate SQM (Sky Quality Meter) value using formula
        sqm = round((math.log10((artificial_brightness + 0.171168465) / 108000000) / -0.4), 2)
        # Convert SQM to Bortle scale
        bortle_class = round(sqm_to_bortle(sqm, logger), 2)
        logger.info(f"Retrieved Bortle scale: {bortle_class}, SQM value: {sqm}")
        return bortle_class, sqm, None, api_valid, api_endpoint_valid

    except requests.exceptions.HTTPError as err:
        # Handle HTTP errors (e.g., 404, 500)
        logger.error(f"HTTP Error: {err}")
        return 0, 0, f"HTTP Error: {err}", api_valid, False
    except ValueError as e:
        # Handle errors converting response to float
        logger.error(f"Could not convert response to float: {str(e)}")
        return 0, 0, f"Could not convert response to float: {str(e)}", api_valid, api_endpoint_valid
    except requests.exceptions.RequestException as e:
        # Handle other request errors (e.g., connection issues)
        logger.error(f"Request Error: {str(e)}")
        return 0, 0, f"Request Error: {str(e)}", api_valid, api_endpoint_valid
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"An error occurred: {str(e)}")
        return 0, 0, f"An error occurred: {str(e)}", api_valid, api_endpoint_valid

def sqm_to_bortle(sqm: float, logger) -> int:
    """Converts an SQM value to a Bortle scale classification.

    Maps the Sky Quality Meter (SQM) value to a Bortle scale class (1-9) based on predefined ranges.

    Args:
        sqm (float): Sky Quality Meter value in mag/arcsec².
        logger: Logger object for logging messages.

    Returns:
        int: Bortle scale classification (1-9).

    Raises:
        ValueError: If sqm is not a number.
    """
    # Log start of SQM to Bortle conversion
    logger.info("")
    logger.info("CONVERTING SQM VALUE TO BORTLE SCALE")
    logger.info(f"Converting SQM value {sqm} to Bortle scale")

    try:
        # Validate that SQM is a number
        if not isinstance(sqm, (int, float)):
            raise ValueError("SQM value must be a number")

        # Map SQM value to Bortle scale based on ranges
        if sqm > 21.99:
            logger.debug("SQM value corresponds to Bortle scale 1")
            return 1
        elif 21.50 <= sqm <= 21.99:
            logger.debug("SQM value corresponds to Bortle scale 2")
            return 2
        elif 21.25 <= sqm <= 21.49:
            logger.debug("SQM value corresponds to Bortle scale 3")
            return 3
        elif 20.50 <= sqm <= 21.24:
            logger.debug("SQM value corresponds to Bortle scale 4")
            return 4
        elif 19.50 <= sqm <= 20.49:
            logger.debug("SQM value corresponds to Bortle scale 5")
            return 5
        elif 18.50 <= sqm <= 19.49:
            logger.debug("SQM value corresponds to Bortle scale 6")
            return 6
        elif 17.50 <= sqm <= 18.49:
            logger.debug("SQM value corresponds to Bortle scale 7")
            return 7
        elif 17.00 <= sqm <= 17.49:
            logger.debug("SQM value corresponds to Bortle scale 8")
            return 8
        else:
            logger.debug("SQM value corresponds to Bortle scale 9")
            return 9

    except Exception as e:
        # Log error and return default Bortle class (9) for urban skies
        logger.error(f"Error converting SQM to Bortle scale: {str(e)}")
        return 9

def is_new_location(existing_sites: pd.DataFrame, lat_long_pair: Tuple[float, float], logger) -> bool:
    """Checks if a given location is new by comparing with existing sites.

    Compares the latitude and longitude pair against existing sites, rounding to one decimal place.

    Args:
        existing_sites (pd.DataFrame): DataFrame of existing sites with 'latitude' and 'longitude' columns.
        lat_long_pair (Tuple[float, float]): Tuple of (latitude, longitude) to check.
        logger: Logger object for logging messages.

    Returns:
        bool: True if the location is new (no match found), False otherwise.

    Raises:
        ValueError: If existing_sites is not a DataFrame or lat_long_pair is not a valid tuple.
    """
    try:
        # Validate input types
        if not isinstance(existing_sites, pd.DataFrame):
            raise ValueError("existing_sites must be a pandas DataFrame")
        if not isinstance(lat_long_pair, tuple) or len(lat_long_pair) != 2:
            raise ValueError("lat_long_pair must be a tuple of two floats")

        # Check if DataFrame is empty or no matching coordinates are found
        is_new = existing_sites.empty or existing_sites[
            (np.ceil(existing_sites['latitude'] * 10) / 10 == np.ceil(lat_long_pair[0] * 10) / 10) &
            (np.ceil(existing_sites['longitude'] * 10) / 10 == np.ceil(lat_long_pair[1] * 10) / 10)
        ].empty
        logger.debug(f"Location {lat_long_pair} is {'new' if is_new else 'existing'}")
        return is_new

    except Exception as e:
        # Log error and assume location is new to proceed with processing
        logger.error(f"Error checking if location is new: {str(e)}")
        return True

def process_new_location(lat_long_pair: Tuple[float, float], api_key: str, api_endpoint: str, state: Dict) -> Tuple[Tuple[float, float], str, int, float]:
    """Processes a new location using reverse geocoding and retrieves Bortle/SQM values.

    Performs reverse geocoding to get a location string and uses an API to fetch Bortle and SQM values.
    Falls back to default configuration values on errors.

    Args:
        lat_long_pair (Tuple[float, float]): Tuple of (latitude, longitude) for the new location.
        api_key (str): API key for the geolocation service.
        api_endpoint (str): API endpoint URL for the geolocation service.
        state (Dict): Sites state dictionary containing logger, headers_state, and geolocator.

    Returns:
        Tuple[Tuple[float, float], str, int, float]: Tuple containing:
            - Updated lat/long pair (Tuple[float, float]): Original or default coordinates.
            - Location string (str): Geocoded address or default site name.
            - Bortle scale (int): Bortle class (1-9).
            - SQM value (float): Sky Quality Meter value in mag/arcsec².

    Raises:
        ValueError: If lat_long_pair, api_key, or api_endpoint are invalid.
    """
    # Extract logger and state components
    logger = state['logger']
    headers_state = state['headers_state']
    geolocator = state['geolocator']

    # Log start of new location processing
    logger.info("")
    logger.info("PROCESSING NEW LOCATION")
    logger.info(f"Site location does not exist in existing sites: {lat_long_pair}")

    try:
        # Validate input types
        if not isinstance(lat_long_pair, tuple) or len(lat_long_pair) != 2:
            raise ValueError("lat_long_pair must be a tuple of two floats")
        if not isinstance(api_key, str) or not isinstance(api_endpoint, str):
            raise ValueError("API key and endpoint must be strings")

        # Initialize with default site name from configuration
        location_str = headers_state['config']['defaults']['SITE']
        try:
            # Check if geolocator is initialized
            if geolocator is None:
                raise ValueError("Geolocator not initialized")
            # Perform reverse geocoding to get address
            location = geolocator.reverse(lat_long_pair, exactly_one=True)
            if location is None:
                raise Exception("No location found for the given lat_long_pair")
            # Update location string with geocoded address
            location_str = location.address
            logger.info(f"Using location string: {location_str}")
            logger.info(f"Using SITELAT: {lat_long_pair[0]}, SITELONG: {lat_long_pair[1]}")

            # Retrieve Bortle and SQM values
            bortle, sqm, error_msg, _, _ = get_bortle_sqm(lat_long_pair[0], lat_long_pair[1], api_key, api_endpoint, logger)
            if error_msg:
                logger.warning(f"API error: {error_msg}")
            if bortle == 0 and sqm == 0:
                # Use default values if API returns zeros
                bortle = headers_state['config']['defaults']['BORTLE']
                sqm = headers_state['config']['defaults']['SQM']
                logger.warning(f"Bortle and SQM returned 0, using defaults: Bortle {bortle}, SQM {sqm}")
            logger.info(f"Processed new location: {location_str}, Bortle: {bortle}, SQM: {sqm}")
        except Exception as e:
            # Handle geocoding errors and fall back to default values
            logger.warning(f"Geocoding error: {str(e)}. Using default site information")
            logger.info(f"Default location: {headers_state['config']['defaults']['SITE']}, "
                       f"Bortle: {headers_state['config']['defaults']['BORTLE']}, "
                       f"SQM: {headers_state['config']['defaults']['SQM']}")
            location_str = headers_state['config']['defaults']['SITE']
            lat_long_pair = (
                round(float(headers_state['config']['defaults']['SITELAT']), headers_state['dp']),
                round(float(headers_state['config']['defaults']['SITELONG']), headers_state['dp'])
            )
            bortle = headers_state['config']['defaults']['BORTLE']
            sqm = headers_state['config']['defaults']['SQM']

        # Return processed location data
        return lat_long_pair, location_str, bortle, sqm

    except Exception as e:
        # Log error and return default values
        logger.error(f"Error processing new location: {str(e)}")
        return (
            lat_long_pair,
            headers_state['config']['defaults']['SITE'],
            headers_state['config']['defaults']['BORTLE'],
            headers_state['config']['defaults']['SQM']
        )

def save_new_location(location_str: str, bortle: int, sqm: float, lat_long_pair: Tuple[float, float], state: Dict) -> None:
    """Saves a new location to the configuration and updates the existing sites DataFrame.

    Updates the configuration with the new location’s details and refreshes the sites DataFrame.

    Args:
        location_str (str): Location string (e.g., geocoded address).
        bortle (int): Bortle scale value (1-9).
        sqm (float): Sky Quality Meter value in mag/arcsec².
        lat_long_pair (Tuple[float, float]): Tuple of (latitude, longitude).
        state (Dict): Sites state dictionary containing logger and headers_state.

    Raises:
        ValueError: If location_str, bortle, sqm, or lat_long_pair are invalid.
        KeyError: If headers_state or configuration keys are missing.
    """
    # Extract logger and headers_state
    logger = state['logger']
    headers_state = state['headers_state']

    try:
        # Validate input types
        if not isinstance(location_str, str) or not location_str:
            raise ValueError("location_str must be a non-empty string")
        if not isinstance(bortle, (int, float)) or not isinstance(sqm, (int, float)):
            raise ValueError("Bortle and SQM must be numbers")
        if not isinstance(lat_long_pair, tuple) or len(lat_long_pair) != 2:
            raise ValueError("lat_long_pair must be a tuple of two floats")

        # Create dictionary of location data
        keys_values = {'latitude': lat_long_pair[0], 'longitude': lat_long_pair[1], 'bortle': bortle, 'sqm': sqm}
        # Import and call update_config to save new location
        from config_functions import update_config
        update_config(headers_state['config'], 'sites', location_str, keys_values, logger)
        logger.info(f"Saved new site: {location_str} with Latitude {lat_long_pair[0]}, "
                   f"Longitude {lat_long_pair[1]}, Bortle: {bortle}, SQM: {sqm}")

        # Update existing sites DataFrame
        state['existing_sites_df'] = pd.DataFrame(headers_state['config']['sites'])
        logger.info(f"Updated existing sites DataFrame: {state['existing_sites_df'].to_dict()}")

    except Exception as e:
        # Log error if saving fails
        logger.error(f"Error saving new location {location_str}: {str(e)}")

def get_site_data_old(df: pd.DataFrame, state: Dict) -> pd.DataFrame:
    """Retrieves and updates site data for each row in the DataFrame.

    Processes each row to assign site name, Bortle scale, and SQM value, either from existing sites
    or by geocoding new locations.

    Args:
        df (pd.DataFrame): Input DataFrame with 'SITELAT' and 'SITELONG' columns.
        state (Dict): Sites state dictionary containing logger, headers_state, existing_sites_df, and geolocator.

    Returns:
        pd.DataFrame: Updated DataFrame with 'SITE', 'BORTLE', and 'SQM' columns.

    Raises:
        ValueError: If df is not a pandas DataFrame or state is invalid.
        KeyError: If required configuration or state keys are missing.
    """
    # Extract logger and headers_state
    logger = state['logger']
    headers_state = state['headers_state']

    # Log start of site data retrieval
    logger.info("")
    logger.info("GETTING SITE DATA")
    logger.info("Starting site data retrieval")

    try:
        # Validate input DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        # Check for empty DataFrame
        if df.empty:
            logger.error("No images found in session. Terminating program")
            sys.exit("Error: No images found in session")

        # Retrieve API key and endpoint from configuration
        try:
            api_key = list(headers_state['config']['secret'].keys())[0].strip()
            api_endpoint = list(headers_state['config']['secret'].values())[0].strip()
            logger.info(f"Retrieved API key and endpoint: {api_key}, {api_endpoint}")
        except (IndexError, KeyError) as e:
            # Handle missing or invalid secret configuration
            logger.error(f"Error retrieving API key/endpoint: {str(e)}")
            api_key = ""
            api_endpoint = ""

        # Transpose existing sites DataFrame for easier access
        est = state['existing_sites_df'].transpose()
        # Track seen locations to avoid redundant processing
        seen_locations = {}

    

        # Process each row in the DataFrame
        for index, row in df.iterrows():
            try:
                # Round latitude and longitude to match configuration precision
                lat_long_pair = (round(row['SITELAT'], headers_state['dp']),
                               round(row['SITELONG'], headers_state['dp']))
                # Round to one decimal place for matching
                r_lat_long_pair = (np.ceil(row['SITELAT'] * 10) / 10,
                                 np.ceil(row['SITELONG'] * 10) / 10)

                # Check if location is new
                if is_new_location(est, lat_long_pair, logger):
                    logger.info(f"New location found: {lat_long_pair}")
                    # Process new location to get site data
                    lat_long_pair_return, location_str, bortle, sqm = process_new_location(
                        lat_long_pair, api_key, api_endpoint, state
                    )

                    # Update coordinates if defaults were used
                    if lat_long_pair_return != lat_long_pair:
                        logger.warning("Default site used instead of given lat_long_pair")
                        df.at[index, 'SITELAT'] = lat_long_pair_return[0]
                        df.at[index, 'SITELONG'] = lat_long_pair_return[1]
                        lat_long_pair = lat_long_pair_return

                    logger.info(f"New location: {location_str}, Bortle: {bortle}, SQM: {sqm}")
                    # Save new location to configuration
                    save_new_location(location_str, bortle, sqm, lat_long_pair, state)
                    # Update transposed sites DataFrame
                    est = state['existing_sites_df'].transpose()
                else:
                    # Track existing location
                    if r_lat_long_pair not in seen_locations:
                        seen_locations[r_lat_long_pair] = r_lat_long_pair
                        logger.info(f"Existing location seen in config: {lat_long_pair}")
                        logger.info(f"Seen locations: {seen_locations}")

                # Find location data in existing sites
                state['location_info'] = find_location_by_coords(est, lat_long_pair, logger)

                # Assign site data to row
                if state['location_info'] is None:
                    logger.error(f"No location found in ['sites'] for lat_long_pair: {lat_long_pair}")
                    logger.info("Using default site information")
                    site = headers_state['config']['defaults']['SITE']
                    bortle = headers_state['config']['defaults']['BORTLE']
                    sqm = headers_state['config']['defaults']['SQM']
                else:
                    site = state['location_info'].index[0]
                    bortle = state['location_info']['bortle'].iloc[0]
                    sqm = state['location_info']['sqm'].iloc[0]

                df.at[index, 'BORTLE'] = bortle
                df.at[index, 'SQM'] = sqm
                df.at[index, 'SITE'] = site

            except Exception as e:
                # Handle errors for individual rows and use default values
                logger.error(f"Error processing row {index} for site data: {str(e)}")
                df.at[index, 'BORTLE'] = headers_state['config']['defaults']['BORTLE']
                df.at[index, 'SQM'] = headers_state['config']['defaults']['SQM']
                df.at[index, 'SITE'] = headers_state['config']['defaults']['SITE']

        logger.info("Completed site data retrieval")
        return df

    except Exception as e:
        # Log error and return unmodified DataFrame
        logger.error(f"Failed to retrieve site data: {str(e)}")
        return df
    
    import pandas as pd

def is_valid_coordinate(value):
    """Check if a value is a valid numeric coordinate."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def get_site_data(df: pd.DataFrame, state: Dict) -> pd.DataFrame:
    """Retrieves and updates site data for each row in the DataFrame.

    Processes each row to assign site name, Bortle scale, and SQM value, either from existing sites
    or by geocoding new locations.

    Args:
        df (pd.DataFrame): Input DataFrame with 'SITELAT' and 'SITELONG' columns.
        state (Dict): Sites state dictionary containing logger, headers_state, existing_sites_df, and geolocator.

    Returns:
        pd.DataFrame: Updated DataFrame with 'SITE', 'BORTLE', and 'SQM' columns.

    Raises:
        ValueError: If df is not a pandas DataFrame or state is invalid.
        KeyError: If required configuration or state keys are missing.
    """
    # Extract logger and headers_state
    logger = state['logger']
    headers_state = state['headers_state']

    # Log start of site data retrieval
    logger.info("")
    logger.info("GETTING SITE DATA")
    logger.info("Starting site data retrieval")

    try:
        # Validate input DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        # Check for empty DataFrame
        if df.empty:
            logger.error("No images found in session. Terminating program")
            sys.exit("Error: No images found in session")

        # Retrieve API key and endpoint from configuration
        try:
            api_key = list(headers_state['config']['secret'].keys())[0].strip()
            api_endpoint = list(headers_state['config']['secret'].values())[0].strip()
            logger.info(f"Retrieved API key and endpoint: {api_key}, {api_endpoint}")
        except (IndexError, KeyError) as e:
            # Handle missing or invalid secret configuration
            logger.error(f"Error retrieving API key/endpoint: {str(e)}")
            api_key = ""
            api_endpoint = ""

        # Transpose existing sites DataFrame for easier access
        est = state['existing_sites_df'].transpose()
        # Track seen locations to avoid redundant processing
        seen_locations = {}

        # Process each row in the DataFrame
        for index, row in df.iterrows():
            try:

                if not (is_valid_coordinate(row['SITELAT']) and is_valid_coordinate(row['SITELONG'])):
                    logger.info(f"No valid numeric coordinates for row {index}, using default values")
                    state['location_info'] = None
                    site = headers_state['config']['defaults']['SITE']
                    bortle = headers_state['config']['defaults']['BORTLE']
                    # Check if SQM is 0 or empty
                    logger.info(f"SQM value for row {index} is {row['SQM']}")
                    if pd.isna(row['SQM']) or row['SQM'] == 0 or row['SQM'] == "":
                        sqm = headers_state['config']['defaults']['SQM']
                    else:
                        # This case should not occur since location_info is None, logging for clarity
                        logger.warning(f"Unexpected SQM value {row['SQM']} with no valid coordinates")
                        sqm = row['SQM']

                # Check if coordinates are empty
                #logger.info(f"SITELAT = {row['SITELAT']}, SITELONG = {row['SITELONG']}")
                #if pd.isna(row['SITELAT']) or pd.isna(row['SITELONG']):
                #    logger.info(f"No coordinates for row {index}, using default values")
                #    state['location_info'] = None
                #    site = headers_state['config']['defaults']['SITE']
                #    bortle = headers_state['config']['defaults']['BORTLE']
                #    # Check if SQM is 0 or empty
                #    if pd.isna(row['SQM']) or row['SQM'] == 0:
                #        sqm = headers_state['config']['defaults']['SQM']
                #    else:
                #        # This case should not occur since location_info is None, logging for clarity
                #        logger.warning(f"Unexpected SQM value {row['SQM']} with no coordinates")
                #        sqm = headers_state['config']['defaults']['SQM']
                else:
                    # Round latitude and longitude to match configuration precision
                    lat_long_pair = (round(row['SITELAT'], headers_state['dp']),
                                   round(row['SITELONG'], headers_state['dp']))
                    # Round to one decimal place for matching
                    r_lat_long_pair = (np.ceil(row['SITELAT'] * 10) / 10,
                                     np.ceil(row['SITELONG'] * 10) / 10)

                    # Check if location is new
                    if is_new_location(est, lat_long_pair, logger):
                        logger.info(f"New location found: {lat_long_pair}")
                        # Process new location to get site data
                        lat_long_pair_return, location_str, bortle, sqm = process_new_location(
                            lat_long_pair, api_key, api_endpoint, state
                        )

                        # Update coordinates if defaults were used
                        if lat_long_pair_return != lat_long_pair:
                            logger.warning("Default site used instead of given lat_long_pair")
                            df.at[index, 'SITELAT'] = lat_long_pair_return[0]
                            df.at[index, 'SITELONG'] = lat_long_pair_return[1]
                            lat_long_pair = lat_long_pair_return

                        logger.info(f"New location: {location_str}, Bortle: {bortle}, SQM: {sqm}")
                        # Save new location to configuration
                        save_new_location(location_str, bortle, sqm, lat_long_pair, state)
                        # Update transposed sites DataFrame
                        est = state['existing_sites_df'].transpose()
                    else:
                        # Track existing location
                        if r_lat_long_pair not in seen_locations:
                            seen_locations[r_lat_long_pair] = r_lat_long_pair
                            logger.info(f"Existing location seen in config: {lat_long_pair}")
                            logger.info(f"Seen locations: {seen_locations}")

                    # Find location data in existing sites
                    state['location_info'] = find_location_by_coords(est, lat_long_pair, logger)

                    # Assign site data to row
                    if state['location_info'] is None:
                        logger.error(f"No location found in ['sites'] for lat_long_pair: {lat_long_pair}")
                        logger.info("Using default site information")
                        site = headers_state['config']['defaults']['SITE']
                        bortle = headers_state['config']['defaults']['BORTLE']
                        sqm = headers_state['config']['defaults']['SQM']
                    else:
                        site = state['location_info'].index[0]
                        bortle = state['location_info']['bortle'].iloc[0]
                        # Check if SQM is 0 or empty
                        if pd.isna(row['SQM']) or row['SQM'] == 0:
                            sqm = headers_state['config']['defaults']['SQM']
                        else:
                            sqm = state['location_info']['sqm'].iloc[0]

                df.at[index, 'BORTLE'] = bortle
                df.at[index, 'SQM'] = sqm
                df.at[index, 'SITE'] = site

            except Exception as e:
                # Handle errors for individual rows and use default values
                logger.error(f"Error processing row {index} for site data: {str(e)}")
                df.at[index, 'BORTLE'] = headers_state['config']['defaults']['BORTLE']
                df.at[index, 'SQM'] = headers_state['config']['defaults']['SQM']
                df.at[index, 'SITE'] = headers_state['config']['defaults']['SITE']

        logger.info("Completed site data retrieval")
        return df

    except Exception as e:
        # Log error and return unmodified DataFrame
        logger.error(f"Failed to retrieve site data: {str(e)}")
        return df