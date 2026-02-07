import logging
import os
import struct
import xml.etree.ElementTree as ET
import re
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List, Tuple
from configobj import ConfigObj
from astropy.io import fits
from datetime import datetime
import logging
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor
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
# Author : SDG & Gemini

class WorkerLogger(logging.Logger):
    def __init__(self):
        super().__init__('worker_logger')
    def info(self, msg, *args, **kwargs): pass
    def debug(self, msg, *args, **kwargs): pass
    def warning(self, msg, *args, **kwargs): pass
    def error(self, msg, *args, **kwargs): pass

def worker_process_header(file_path, defaults, override, wanted_keys, useobsdate, dp):
    try:
        # Reconstruct minimal state
        config = {'defaults': defaults, 'override': override}
        logger = WorkerLogger()
        
        state = {
            'config': config,
            'logger': logger,
            'dp': dp,
            'number_of_images_processed': 0,
            'header': {},
            'headers': pd.DataFrame(),
            'headers_reduced': [],
            'wanted_keys': wanted_keys,
            'useobsdate': useobsdate,
            'header_filename': os.path.basename(file_path),
            'number': 1
        }
        
        return process_headers(file_path, state)
    except Exception as e:
        # Failsafe logging to stderr since main logger is not available
        import sys
        sys.stderr.write(f"Worker failed for {file_path}: {e}\n")
        return None



def initialize_headers(config: ConfigObj, logger: logging.Logger, dp: int) -> Dict[str, Any]:
    """Initializes the header processing state for FITS/XISF file handling.

    Creates a dictionary containing configuration, logger, precision, and state variables
    for processing astronomical image headers.

    Args:
        config (ConfigObj): Configuration object containing default settings for header processing.
        logger (logging.Logger): Logger instance for logging messages.
        dp (int): Decimal precision for floating-point values.

    Returns:
        Dict[str, Any]: Dictionary containing initialized header processing state with keys:
            - config: The input configuration object.
            - logger: The logger instance.
            - dp: Decimal precision for numerical values.
            - number_of_images_processed: Counter for processed images (initially 0).
            - header: Placeholder for current header dictionary.
            - headers: Empty DataFrame for accumulating headers.
            - headers_reduced: List for reduced headers.
            - useobsdate: Boolean indicating whether to use observation dates (from config).

    Raises:
        ValueError: If config is not a ConfigObj, logger is not a logging.Logger, or dp is negative.
    """
    try:
        # Validate input types
        if not isinstance(config, ConfigObj):
            raise ValueError(f"config must be a ConfigObj, got {type(config).__name__}")
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(dp, int) or dp < 0:
            raise ValueError("dp must be a non-negative integer")

        # Log initialization of header state
        logger.info("Initialising headers state")
        # Create headers state dictionary
        headers_state = {
            'config': config,
            'logger': logger,
            'dp': dp,
            'number_of_images_processed': 0,
            'header': {},
            'headers': pd.DataFrame(),
            'headers_reduced': [],
            'useobsdate': config['defaults'].get('USEOBSDATE', 'TRUE').lower() != 'false'
        }
        # Log useobsdate setting for debugging
        logger.debug(f"Headers state initialized: useobsdate={headers_state['useobsdate']}")
        return headers_state

    except Exception as e:
        # Log and raise any initialization errors
        logger.error(f"Error initialising headers state: {str(e)}")
        raise

def read_xisf_header(file_path: str, logger: logging.Logger) -> Optional[str]:
    """Reads the XML header from an XISF file.

    Opens the file, verifies the XISF signature, and extracts the XML header string.

    Args:
        file_path (str): Path to the XISF file.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        Optional[str]: XML header string if successfully read, or None if an error occurs.

    Raises:
        ValueError: If file_path is not a non-empty string or logger is invalid.
        OSError: If the file cannot be read or has an invalid format.
    """
    try:
        # Validate input types
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")

        # Log attempt to read XISF header
        logger.debug(f"Attempting to read XISF header from {file_path}")
        # Check if file exists
        if not os.path.isfile(file_path):
            logger.error(f"File {file_path} does not exist")
            return None

        # Open file in binary mode
        with open(file_path, 'rb') as file:
            # Read and verify XISF signature
            signature = file.read(8).decode('ascii', errors='ignore')
            if signature != 'XISF0100':
                logger.error(f"Invalid XISF file format for {file_path}: expected XISF0100, got {signature}")
                return None
            # Read header length (4 bytes, unsigned int)
            header_length = struct.unpack('<I', file.read(4))[0]
            # Skip reserved field (4 bytes)
            file.read(4)
            # Read and decode XML header
            xml_header = file.read(header_length).decode('utf-8', errors='ignore')
            logger.debug(f"Successfully read XISF header from {file_path}")
            return xml_header

    except OSError as e:
        # Log file access errors
        logger.error(f"Error reading XISF file {file_path}: {str(e)}")
        return None
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error reading XISF header from {file_path}: {str(e)}")
        return None

def xml_to_data(xml_header: str, logger: logging.Logger) -> Optional[Tuple[Dict[str, Any], int]]:
    """Parses XML header data from an XISF file to extract FITS keywords and image count.

    Processes XML to extract header key-value pairs and the number of images from processing history.

    Args:
        xml_header (str): XML header string from an XISF file.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        Optional[Tuple[Dict[str, Any], int]]: Tuple of header dictionary and number of images,
            or None if parsing fails.

    Raises:
        ValueError: If xml_header is not a string or logger is invalid.
        ET.ParseError: If the XML is invalid.
    """
    try:
        # Validate input types
        if not isinstance(xml_header, str):
            raise ValueError(f"xml_header must be a string, got {type(xml_header).__name__}")
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")

        # Log start of XML parsing
        logger.debug("Parsing XML header data")
        # Define XISF namespace for XML parsing
        ns = {'xisf': 'http://www.pixinsight.com/xisf'}
        ET.register_namespace('', ns['xisf'])
        # Initialize header dictionary and image count
        header: Dict[str, Any] = {}
        number = 1

        # Parse XML header
        root = ET.fromstring(xml_header)
        logger.debug("Successfully parsed XML header")

        # Extract FITS keywords from XML
        for fits_keyword in root.findall('.//xisf:FITSKeyword', namespaces=ns):
            name = fits_keyword.get('name')
            value = fits_keyword.get('value')
            comment = fits_keyword.get('comment') or ''
            if name:
                # Store keyword-value pair
                header[name] = value
                # Extract SWCREATE from PixInsight comment
                if 'Integration with PixInsight' in comment:
                    header['SWCREATE'] = ' '.join(comment.split()[2:])
                    logger.debug(f"Extracted SWCREATE from comment: {header['SWCREATE']}")

        # Extract number of images from processing history
        property_elem = root.find(".//xisf:Property[@id='PixInsight:ProcessingHistory']", namespaces=ns)
        if property_elem is not None and property_elem.text and property_elem.text.strip():
            try:
                # Parse nested XML in processing history
                nested_root = ET.fromstring(property_elem.text)
                table_elem = nested_root.find(".//table[@id='images']")
                if table_elem is not None and table_elem.get('rows'):
                    # Extract number of images from 'rows' attribute
                    number = int(table_elem.get('rows'))
                    logger.debug(f"Found {number} images in ProcessingHistory")
                else:
                    logger.debug("No 'rows' attribute or table found in ProcessingHistory")
            except ET.ParseError:
                logger.error("Failed to parse nested XML in ProcessingHistory")
        else:
            # Check if it is a master frame before warning
            imagetyp = header.get('IMAGETYP', '').upper()
            if 'MASTER' in imagetyp:
                logger.warning("No PixInsight:ProcessingHistory property found")
            else:
                logger.debug("No PixInsight:ProcessingHistory property found")

        # Log extracted header for debugging
        logger.debug(f"Extracted header: {header}")
        return header, number

    except ET.ParseError as e:
        # Log XML parsing errors
        logger.error(f"Invalid XML format: {str(e)}")
        return None
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error parsing XML header: {str(e)}")
        return None

def get_number_of_FIT_cal_frames(header: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Extracts the number of calibration frames from FITS header HISTORY entries.

    Parses HISTORY entries to extract the number of images and software creator for calibration frames.

    Args:
        header (Dict[str, Any]): Header dictionary containing HISTORY entries.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        Dict[str, Any]: Updated header dictionary with NUMBER and SWCREATE keys if found.

    Raises:
        ValueError: If header is not a dictionary or logger is invalid.
    """
    try:
        # Validate input types
        if not isinstance(header, dict):
            raise ValueError(f"header must be a dictionary, got {type(header).__name__}")
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")

        # Log start of calibration frame count extraction
        logger.debug("Extracting calibration frame count from FITS header HISTORY")
        # Check if HISTORY key exists
        if 'HISTORY' not in header:
            logger.debug("No HISTORY key found in header")
            return header

        # Ensure HISTORY is a list for iteration
        history_lines = header['HISTORY'] if isinstance(header['HISTORY'], list) else [header['HISTORY']]
        logger.debug("Found HISTORY in header")

        # Parse HISTORY lines for relevant data
        for line in history_lines:
            if 'Integration with PixInsight' in line:
                # Extract SWCREATE from PixInsight integration comment
                header['SWCREATE'] = ' '.join(line.split()[2:])
                logger.debug(f"Found SWCREATE in HISTORY: {header['SWCREATE']}")
            if 'ImageIntegration.numberOfImages:' in line:
                try:
                    # Extract number of images from HISTORY
                    number = int(line.split()[-1])
                    header['NUMBER'] = number
                    logger.debug(f"Found {number} calibration frames in HISTORY")
                except ValueError as e:
                    # Log error if number parsing fails
                    logger.error(f"Error parsing number of images from HISTORY: {str(e)}")

        return header

    except Exception as e:
        # Log and return header on error
        logger.error(f"Error extracting calibration frame count: {str(e)}")
        return header

def clean_object_column(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Cleans the OBJECT column in the DataFrame for LIGHT frames.

    Normalizes object names by splitting on spaces or underscores and handling panel numbers
    (e.g., appending 'Panel' before numeric suffixes).

    Args:
        df (pd.DataFrame): DataFrame containing header data with 'OBJECT' and 'IMAGETYP' columns.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        pd.DataFrame: DataFrame with cleaned 'OBJECT' column.

    Raises:
        ValueError: If df is not a DataFrame, logger is invalid, or required columns are missing.
    """
    try:
        # Validate input types
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        # Check for required columns
        required_columns = {'IMAGETYP', 'OBJECT'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Log start of OBJECT column cleaning
        logger.info("Cleaning OBJECT column for LIGHT frames")
        # Create a copy to avoid modifying the input DataFrame
        df = df.copy()
        # Process each row
        for i in df.index:
            if df.loc[i, 'IMAGETYP'] == 'LIGHT':
                try:
                    # Split OBJECT name on spaces or underscores
                    parts = re.split(r'\s|_', str(df.loc[i, 'OBJECT']))
                    # Remove empty parts and strip whitespace
                    parts = [part.strip() for part in parts if part]
                    # If last part is numeric, insert 'Panel' before it
                    if len(parts) >= 3 and parts[-1].isdigit():
                        parts[-2] = 'Panel'
                    # Join parts back into cleaned OBJECT name
                    df.loc[i, 'OBJECT'] = ' '.join(parts)
                    logger.debug(f"Cleaned OBJECT for index {i}: {df.loc[i, 'OBJECT']}")
                except Exception as e:
                    # Log error for individual row processing
                    logger.error(f"Error cleaning OBJECT for index {i}: {str(e)}")
        logger.info("Completed cleaning OBJECT column")
        return df

    except Exception as e:
        # Log and return DataFrame on error
        logger.error(f"Error cleaning OBJECT column: {str(e)}")
        return df

def process_headers(file_path: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Directly extracts and normalizes the header from a single FITS or XISF file.

    The function determines the file type, opens it, and performs initial normalization:
    1. Maps alternative keys (e.g., CREATOR -> SWCREATE) using config 'override' rules.
    2. Identifies Master files and their internal image counts (NUMBER).
    3. Normalizes IMAGETYP strings (e.g., 'Light Frame' -> 'LIGHT').
    4. Drops non-essential metadata to reduce memory footprint.

    Args:
        file_path (str): The absolute path to the FITS or XISF file.
        state (Dict[str, Any]): Dictionary containing config and logger.

    Returns:
        Optional[Dict[str, Any]]: A reduced dictionary of normalized header keys, 
                                  or None if the file is invalid or a 'master light'.
    """
    try:
        # Validate input types
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dictionary, got {type(state).__name__}")
        if not all(key in state for key in ['config', 'logger']):
            raise ValueError("state must contain 'config' and 'logger' keys")
        if not isinstance(state['logger'], logging.Logger):
            raise ValueError("state['logger'] must be a logging.Logger instance")

        # Extract config and logger from state
        config, logger = state['config'], state['logger']
        # Initialize number of images to 1 (default for non-master files)
        state['number'] = 1
        # Define wanted keys from config defaults and additional mandatory tracking keys
        state['wanted_keys'] = list(config['defaults'].keys()) + ['FILENAME', 'NUMBER']
        
        # USEOBSDATE override logic
        if config['defaults'].get('USEOBSDATE', 'TRUE').lower() == 'false':
            logger.debug("USEOBSDATE is set to False")
            state['useobsdate'] = False
            if 'USEOBSDATE' in state['wanted_keys']:
                state['wanted_keys'].remove('USEOBSDATE')
                logger.debug("Removed USEOBSDATE from wanted_keys")

        state['header_filename'] = os.path.basename(file_path)
        logger.debug(f"Processing headers for file: {state['header_filename']}")

        if not os.path.isfile(file_path):
            logger.error(f"File {file_path} does not exist")
            return None

        # Initialize empty header dictionary
        hdr: Dict[str, Any] = {}
        
        # Type-Specific Processing
        if file_path.lower().endswith(('.fits', '.fit', '.fts')):
            try:
                # FITS: Uses Astropy to open the file and extract the primary HDU header.
                with fits.open(file_path) as hdul:
                    hdr = dict(hdul[0].header)
                    # Extract calibration frame count from HISTORY (common in PixInsight masters)
                    hdr = get_number_of_FIT_cal_frames(hdr, logger)
                    logger.debug(f"Extracted FITS header from {file_path}")
            except OSError as e:
                logger.error(f"Error reading FITS file {file_path}: {str(e)}")
                return None
        elif file_path.lower().endswith('.xisf'):
            # XISF: Standard PixInsight format. Requires reading the XML header block.
            xml_header = read_xisf_header(file_path, logger)
            if xml_header:
                result = xml_to_data(xml_header, logger)
                if result:
                    hdr, state['number'] = result
                    logger.debug(f"Parsed XISF header from {file_path}")
                else:
                    logger.error(f"Failed to parse XISF header from {file_path}")
                    return None
            else:
                logger.error(f"No XISF header found for {file_path}")
                return None
        else:
            logger.warning(f"Unsupported file extension for {file_path}")
            return None

        # Basic Validation
        if 'IMAGETYP' not in hdr:
            logger.warning(f"IMAGETYP key not found in header for {file_path}")
            return None
        
        # Hard Filter: Drop 'master light' frames.
        # These are final integrated images and contain no useful session-level metadata.
        if 'master light' in hdr['IMAGETYP'].lower():
            logger.debug(f"Dropping header for {file_path} as it contains 'master light' in IMAGETYP")
            return None

        # Normalize IMAGETYP for LIGHT frames: Standardizes 'Light Frame', 'LIGHTFRAME', etc. to 'LIGHT'.
        if 'light' in hdr['IMAGETYP'].lower():
            hdr['IMAGETYP'] = 'LIGHT'
            logger.debug("Converted IMAGETYP to LIGHT")
        
        #replaced code
        #if hdr['IMAGETYP'] in ['LIGHTFRAME', "'LIGHTFRAME'", "'Light Frame'", 'Light Frame']:
        #    hdr['IMAGETYP'] = 'LIGHT'
        #    logger.info("Converted IMAGETYP from LIGHTFRAME to LIGHT")

        # Log if file is a master frame other than a master light frame
        if state['number'] > 1:
            logger.debug(f"File {file_path} is a master with {state['number']} images")
        else:
            logger.debug(f"File {file_path} is not a master")

        # Log raw header for debugging
        logger.debug(f"Raw header: {hdr}")
        # Reduce header to wanted keys
        reduced_header = reduce_headers(hdr, state)
        logger.debug(f"Reduced header: {reduced_header}")
        return reduced_header

    except Exception as e:
        # Log and return None on error
        logger.error(f"Error processing headers for {file_path}: {str(e)}")
        return None

def process_directory(directory: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Processes all FITS/XISF files in a directory to extract headers.

    Iterates through files in the directory and its subdirectories, processing headers for supported file types.

    Args:
        directory (str): Path to the directory to process.
        state (Dict[str, Any]): Dictionary containing headers state with logger, etc.

    Returns:
        List[Dict[str, Any]]: List of processed header dictionaries.

    Raises:
        ValueError: If directory is not a non-empty string, state is not a dictionary, or required state keys are missing.
    """
    try:
        # Validate input types
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("directory must be a non-empty string")
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dictionary, got {type(state).__name__}")
        if not all(key in state for key in ['logger']):
            raise ValueError("state must contain 'logger' key")
        if not isinstance(state['logger'], logging.Logger):
            raise ValueError("state['logger'] must be a logging.Logger instance")

        # Extract logger
        logger = state['logger']
        # Log start of directory processing
        logger.info(f"Processing directory: {directory}")

        # Initialize list for header dictionaries
        dir_headers: List[Dict[str, Any]] = []

        # Check if directory exists
        if not os.path.isdir(directory):
            logger.error(f"Directory {directory} does not exist or is not a directory")
            return dir_headers

        file_paths = []
        # Walk through directory and subdirectories
        for root, _, files in os.walk(directory, followlinks=True):
            print(f"Processing directory: {root}")
            for file in files:
                file_path = os.path.join(root, file)
                # Skip non-FITS/XISF files
                if not file_path.lower().endswith(('.fits', '.fit', '.fts', '.xisf')):
                    logger.debug(f"Skipping non-FITS/XISF file: {file_path}")
                    continue
                file_paths.append(file_path)

        # Prepare args for parallel processing
        # Convert ConfigObj sections to dicts to ensure picklability across environments
        defaults = dict(state['config']['defaults'])
        override = dict(state['config']['override'])
        
        # Note: wanted_keys might not be fully populated in state if process_headers hasn't run yet, 
        # but initialize_headers doesn't set it. process_headers sets it locally. 
        # However, we can construct the initial list here or let worker do it.
        # Worker reconstructs state, so we just need to pass None for wanted_keys and let process_headers logic handle it?
        # Actually, process_headers says: state['wanted_keys'] = list(config['defaults'].keys()) + ['FILENAME', 'NUMBER']
        # So it overwrites whatever is in state. So we can pass None or empty list.
        # But wait, initialize_headers sets 'wanted_keys'? No, it doesn't.
        # Let's check initialize_headers.
        # It creates state with 'config', 'logger', ... but NOT 'wanted_keys'.
        # process_headers adds it.
        # So passing [] is fine as long as worker_process_header passes it to state, and process_headers overwrites it.
        
        # So we don't need to pass wanted_keys to worker, worker can set it.
        # But my worker_process_header signature has wanted_keys. I'll pass empty list.
        
        useobsdate = state['useobsdate']
        dp = state['dp']

        with ProcessPoolExecutor() as executor:
            # Map returns an iterator that preserves order
            results = executor.map(worker_process_header, 
                                   file_paths, 
                                   [defaults] * len(file_paths), 
                                   [override] * len(file_paths),
                                   [[]] * len(file_paths), # wanted_keys placeholder
                                   [useobsdate] * len(file_paths),
                                   [dp] * len(file_paths))
            
            for header in results:
                if header:
                    dir_headers.append(header)
                    logger.debug(f"Processed header") # Can't log filename easily here without tracking
                else:
                    logger.warning(f"No valid header found") # Can't log filename easily

        # Log number of headers extracted
        logger.info(f"Extracted {len(dir_headers)} headers from directory {directory}")
        print(f"\nExtracted {len(dir_headers)} headers from directory {directory}\n\n")
        return dir_headers

    except Exception as e:
        # Log and return empty list on error
        logger.error(f"Error processing directory {directory}: {str(e)}")
        return []

def apply_light_values(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Applies SITE, SITELAT, and SITELONG from LIGHT frames to non-LIGHT frames.

    Uses the first LIGHT frame’s site data to standardize non-LIGHT frames.

    Args:
        df (pd.DataFrame): DataFrame containing header data with 'IMAGETYP', 'SITE', 'SITELAT', and 'SITELONG' columns.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        pd.DataFrame: Updated DataFrame with applied site values.

    Raises:
        ValueError: If df is not a DataFrame, logger is invalid, or required columns are missing.
    """
    try:
        # Validate input types
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        # Check for required columns
        required_columns = {'IMAGETYP', 'SITE', 'SITELAT', 'SITELONG'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Log start of site value application
        logger.info("Applying LIGHT frame SITE, SITELAT, SITELONG to non-LIGHT frames")
        # Create a copy to avoid modifying the input DataFrame
        df = df.copy()
        # Check for LIGHT frames
        if not df[df['IMAGETYP'] == 'LIGHT'].empty:
            # Get site data from first LIGHT frame
            light_row = df[df['IMAGETYP'] == 'LIGHT'].iloc[0]
            site, sitelat, sitelong = light_row['SITE'], light_row['SITELAT'], light_row['SITELONG']
            # Apply site data to non-LIGHT frames
            df.loc[df['IMAGETYP'] != 'LIGHT', ['SITE', 'SITELAT', 'SITELONG']] = site, sitelat, sitelong
            logger.info("Applied LIGHT frame values to non-LIGHT frames")
        else:
            logger.warning("No LIGHT frames found to apply SITE, SITELAT, SITELONG")
        return df

    except Exception as e:
        # Log and return DataFrame on error
        logger.error(f"Error applying LIGHT frame values: {str(e)}")
        return df

def process_and_accumulate_headers(headers: List[Dict[str, Any]], state: Dict[str, Any]) -> None:
    """Accumulates processed headers into the state’s headers DataFrame.

    Converts header dictionaries to a DataFrame, applies site values, and concatenates to the state’s headers.

    Args:
        headers (List[Dict[str, Any]]): List of header dictionaries to process.
        state (Dict[str, Any]): Dictionary containing headers state with 'headers' and 'logger' keys.

    Raises:
        ValueError: If headers is not a list, state is not a dictionary, or required state keys are missing.
    """
    try:
        # Validate input types
        if not isinstance(headers, list):
            raise ValueError(f"headers must be a list, got {type(headers).__name__}")
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dictionary, got {type(state).__name__}")
        if not all(key in state for key in ['headers', 'logger']):
            raise ValueError("state must contain 'headers' and 'logger' keys")
        if not isinstance(state['logger'], logging.Logger):
            raise ValueError("state['logger'] must be a logging.Logger instance")

        # Extract logger
        logger = state['logger']
        # Log start of header accumulation
        logger.info(f"Accumulating {len(headers)} headers into state")
        # Convert headers list to DataFrame
        df = pd.DataFrame(headers)
        if df.empty:
            logger.warning("No headers to accumulate")
            return

        # Apply LIGHT frame site values to non-LIGHT frames
        processed_df = apply_light_values(df, logger)
        # Concatenate processed headers to state’s headers DataFrame
        state['headers'] = pd.concat([state['headers'], processed_df], ignore_index=True)
        logger.info("Accumulated headers into state DataFrame")

    except Exception as e:
        # Log and raise error
        logger.error(f"Error accumulating headers: {str(e)}")
        raise

def count_subdirectories(root: str, logger: logging.Logger) -> int:
    """Counts the number of subdirectories in a given root directory.

    Uses os.walk to iterate through the directory tree and count subdirectories.

    Args:
        root (str): Path to the root directory.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        int: Number of subdirectories found.

    Raises:
        ValueError: If root is not a non-empty string or logger is invalid.
        OSError: If the directory cannot be accessed.
    """
    try:
        # Validate input types
        if not isinstance(root, str) or not root.strip():
            raise ValueError("root must be a non-empty string")
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")

        # Log start of subdirectory counting
        logger.info(f"Counting subdirectories in {root}")
        # Initialize subdirectory counter
        subdirectory_count = 0
        # Walk through directory tree
        for _, dirs, _ in os.walk(root):
            subdirectory_count += len(dirs)
        logger.info(f"Found {subdirectory_count} subdirectories in {root}")
        return subdirectory_count

    except OSError as e:
        # Log file system errors
        logger.error(f"Error accessing directory {root}: {str(e)}")
        return 0
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error counting subdirectories in {root}: {str(e)}")
        return 0

def process_directories(directories: List[str], state: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crawls multiple directories to extract, normalize, and condition FITS/XISF headers.

    This is the high-level entry point for data acquisition. It handles:
    1. Recursive directory walking.
    2. Parallelized header extraction using ProcessPoolExecutor (optimized for RAID 0).
    3. Initial conditioning (deduplication, coordinate alignment).

    Args:
        directories (List[str]): List of absolute directory paths to scan.
        state (Dict[str, Any]): Dictionary containing header state, logger, and config.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - Conditioned DataFrame: Cleaned and ready for aggregation.
            - Raw DataFrame: Unfiltered results for debug logging.

    Raises:
        ValueError: If input arguments are malformed or missing required keys.
    """
    try:
        # Validate input types
        if not isinstance(directories, list) or not all(isinstance(d, str) for d in directories):
            raise ValueError("directories must be a list of strings")
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dictionary, got {type(state).__name__}")
        if not all(key in state for key in ['logger']):
            raise ValueError("state must contain 'logger' key")
        if not isinstance(state['logger'], logging.Logger):
            raise ValueError("state['logger'] must be a logging.Logger instance")

        # Extract logger
        logger = state['logger']
        # Log start of directory processing
        logger.info("Processing directories")
        # Initialize list for headers
        headers: List[Dict[str, Any]] = []
        # Process each directory
        for directory in directories:
            # Get real path to handle symlinks (standard practice for PixInsight/NINA storage)
            real_dir = os.path.realpath(directory)
            if not os.path.isdir(real_dir):
                logger.error(f"Directory {real_dir} does not exist or is not a directory")
                continue
            # Extract headers from directory
            headers.extend(process_directory(real_dir, state))
            logger.info(f"Processed directory {real_dir}")

        # Create raw headers DataFrame for auditing/debug purposes
        raw_headers_df = pd.DataFrame(headers)
        if raw_headers_df.empty:
            logger.warning("No headers extracted from directories")
            return pd.DataFrame(), raw_headers_df

        # Condition headers: Applies the core cleaning pipeline (deduplication, gain handshake, etc.)
        conditioned_headers = condition_headers(headers, state)
        logger.info("Completed processing directories")

        return conditioned_headers, raw_headers_df

    except Exception as e:
        # Log and return empty DataFrames on error to prevent application crash
        logger.error(f"Error processing directories: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def condition_headers(headers: List[Dict[str, Any]], state: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies a series of data-cleansing filters to the extracted header list.

    The pipeline includes:
    1. Calibration matching (filtering out irrelevant Darks/Bias).
    2. Flat frame consolidation (summing Master Flat counts).
    3. WBPP Deduplication (removing temporary calibrated/registered files).
    4. Coordinate alignment (propagating Light coordinates to Calibration frames).
    5. Gain standardization (Integer Gain Handshake).

    Args:
        headers (List[Dict[str, Any]]): List of raw header dictionaries.
        state (Dict[str, Any]): Dictionary containing config and logger.

    Returns:
        pd.DataFrame: A highly normalized DataFrame ready for parameter aggregation.
    """
    try:
        # Validate input types
        if not isinstance(headers, list):
            raise ValueError(f"headers must be a list, got {type(headers).__name__}")
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dictionary, got {type(state).__name__}")
        if not all(key in state for key in ['config', 'logger']):
            raise ValueError("state must contain 'config' and 'logger' keys")
        if not isinstance(state['logger'], logging.Logger):
            raise ValueError("state['logger'] must be a logging.Logger instance")

        # Extract logger and config
        logger, config = state['logger'], state['config']
        # Log start of header conditioning
        logger.info("Conditioning headers")
        # Convert headers list to DataFrame
        headers_df = pd.DataFrame(headers)

        # Check for empty DataFrame
        if headers_df.empty:
            logger.warning("No headers to condition")
            return headers_df

        # Check for IMAGETYP column (essential for all downstream logic)
        if 'IMAGETYP' not in headers_df.columns:
            logger.error("IMAGETYP column missing in headers DataFrame")
            return headers_df

        # Check for LIGHT frames: If no lights are found, the session is considered invalid.
        if 'LIGHT' not in headers_df['IMAGETYP'].values:
            logger.warning("No LIGHT frames found in headers")
            return headers_df
        
        # Step 1: Calibration Frame Filtering
        # Drops Dark/Bias frames if their exposure doesn't match any Light frame.
        logger.info("Processing calibration frames")
        headers_df = deal_with_calibration_frames(headers_df, state)

        # Step 2: Flat Consolidation
        # Identifies Master Flats and sums their internal NUMBER counts if they match active Light filters.
        logger.info("Processing flat frames")
        headers_df = deal_with_flat_frames(headers_df, state)

        # Step 3: WBPP Deduplication
        # PixInsight's WBPP script creates multiple versions of the same frame (_c, _cc, _r).
        # We filter these to ensure each raw capture is only counted once.
        logger.info("Filtering duplicate LIGHT frames")
        headers_df = filter_and_remove_duplicates(headers_df, logger)

        # Step 4: Normalization
        headers_df['IMAGETYP'] = headers_df['IMAGETYP'].str.upper()
        logger.info("Converted IMAGETYP to uppercase")

        # Step 5: Gain Handshake
        # Updates GAIN values based on EGAIN matches, resolving vendor-specific reporting differences.
        headers_df = check_camera_gains(headers_df, state)
        logger.info("Checked and updated camera gains")

        # Step 6: GAIN Defaulting
        # Handles cases where GAIN was reported as -1 (signaling no data).
        headers_df.loc[headers_df['EGAIN'] == -1, 'EGAIN'] = config['defaults']['EGAIN']
        headers_df['EGAIN'] = headers_df['EGAIN'].abs()
        headers_df.loc[headers_df['GAIN'] == -1, 'GAIN'] = config['defaults']['GAIN']
        headers_df['GAIN'] = headers_df['GAIN'].abs()
        logger.info("Set GAIN to absolute value and applied defaults for negative gains")

        # Step 7: Optical Parameter Extraction
        # Calculates FWHM and Image Scale based on HFR extracted from filenames.
        headers_df = get_HFR(headers_df, state)
        logger.info("Extracted HFR, IMSCALE, and FWHM values")

        # Step 8: Coordinate Propagation
        # Calibration frames often lack coordinates. We find the closest Light frame
        # (spatially) and copy its coordinates to ensure reverse geocoding works.
        headers_df = modify_lat_long(headers_df, logger)
        logger.info("Aligned calibration frame lat/long with closest LIGHT frame")

        # Step 9: Object Cleaning
        # Normalizes target names (e.g. 'M 31' vs 'M31') for consistent grouping.
        headers_df = clean_object_column(headers_df, logger)
        logger.info("Cleaned OBJECT column")
        return headers_df

    except Exception as e:
        logger.error(f"Error conditioning headers: {str(e)}")
        return pd.DataFrame()

def modify_lat_long(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Aligns latitude and longitude of non-LIGHT frames to the closest LIGHT frame.

    Uses Euclidean distance to find the closest LIGHT frame for each non-LIGHT frame.

    Args:
        df (pd.DataFrame): DataFrame with 'IMAGETYP', 'SITELAT', and 'SITELONG' columns.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        pd.DataFrame: Updated DataFrame with aligned latitude and longitude values.

    Raises:
        ValueError: If df is not a DataFrame, logger is invalid, or required columns are missing.
    """
    try:
        # Validate input types
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        # Check for required columns
        required_columns = {'IMAGETYP', 'SITELAT', 'SITELONG'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Log start of coordinate alignment
        logger.info("Aligning non-LIGHT frame lat/long with closest LIGHT frame")
        # Create a copy to avoid modifying the input DataFrame
        df = df.copy()
        # Extract LIGHT frames
        light_df = df[df['IMAGETYP'] == 'LIGHT'].copy()
        if light_df.empty:
            logger.warning("No LIGHT frames found for lat/long alignment")
            return df

        # Convert coordinates to float for distance calculation
        light_df['SITELAT'] = light_df['SITELAT'].astype(float)
        light_df['SITELONG'] = light_df['SITELONG'].astype(float)

        # Process non-LIGHT frames
        for i, row in df[df['IMAGETYP'] != 'LIGHT'].iterrows():
            try:
                # Convert current row’s coordinates to float
                sitelat = float(row['SITELAT'])
                sitelong = float(row['SITELONG'])
                # Calculate Euclidean distances to LIGHT frames
                distances = np.sqrt((light_df['SITELAT'] - sitelat) ** 2 + (light_df['SITELONG'] - sitelong) ** 2)
                # Find index of closest LIGHT frame
                closest_light_index = distances.idxmin()
                # Update coordinates
                df.loc[i, 'SITELAT'] = light_df.loc[closest_light_index, 'SITELAT']
                df.loc[i, 'SITELONG'] = light_df.loc[closest_light_index, 'SITELONG']
                logger.debug(f"Aligned lat/long for index {i} to LIGHT frame at index {closest_light_index}")
            except (ValueError, TypeError) as e:
                # Log error for individual row processing
                logger.error(f"Error aligning lat/long for index {i}: {str(e)}")

        logger.info("Completed lat/long alignment")
        return df

    except Exception as e:
        # Log and return DataFrame on error
        logger.error(f"Error modifying lat/long: {str(e)}")
        return df

def reduce_headers(hdr: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Reduces a header dictionary to a subset of wanted keys with normalized values.
    Maps alternative keys, applies defaults, and handles specific header transformations.
    Uses state['config']['defaults'] values for SITELAT and SITELONG if they are empty strings in hdr.

    Args:
        hdr (Dict[str, Any]): Header dictionary to reduce.
        state (Dict[str, Any]): Dictionary containing headers state with config, logger, wanted_keys, etc.

    Returns:
        Dict[str, Any]: Reduced header dictionary with normalized values.

    Raises:
        ValueError: If hdr or state is not a dictionary, or required state keys are missing.
    """
    try:
        # Validate input types
        if not isinstance(hdr, dict):
            raise ValueError(f"hdr must be a dictionary, got {type(hdr).__name__}")
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dictionary, got {type(state).__name__}")
        if not all(key in state for key in ['config', 'logger', 'wanted_keys', 'header_filename', 'number']):
            raise ValueError("state must contain 'config', 'logger', 'wanted_keys', 'header_filename', and 'number' keys")
        if not isinstance(state['logger'], logging.Logger):
            raise ValueError("state['logger'] must be a logging.Logger instance")
        
        # Extract logger and config
        logger, config = state['logger'], state['config']
        # Log start of header reduction
        logger.debug("Reducing headers")

        for fits_keyword, hdr_keyword in config['override'].items():
            # Handle case where hdr_keyword is a list
            if isinstance(hdr_keyword, list):
                for key in hdr_keyword:
                    if key in hdr:
                        hdr[fits_keyword] = hdr[key]
                        logger.debug(f"Mapped {key} to {fits_keyword}")
            # Handle case where hdr_keyword is a single string
            elif hdr_keyword in hdr:
                hdr[fits_keyword] = hdr[hdr_keyword]
                logger.debug(f"Mapped {hdr_keyword} to {fits_keyword}")
                
        # Map alternative keys to standard keys
        #if 'CREATOR' in hdr:
        #    hdr['SWCREATE'] = hdr['CREATOR']
        #    logger.debug("Mapped CREATOR to SWCREATE")
        #if 'EXPTIME' in hdr and hdr['EXPTIME'] != 'No Data':
        #    hdr['EXPOSURE'] = hdr['EXPTIME']
        #    logger.debug("Mapped EXPTIME to EXPOSURE")
        if 'LAT-OBS' in hdr and hdr['LAT-OBS'] != 'No Data':
            hdr['SITELAT'] = hdr['LAT-OBS']
            hdr['SITELONG'] = hdr['LONG-OBS']
            hdr['FWHM'] = 0
            logger.debug("Mapped LAT-OBS and LONG-OBS for master frame")
        #if 'SITENAME' in hdr:
        #    hdr['SITE'] = hdr['SITENAME']
        #    logger.debug("Mapped SITENAME to SITE")
        #if 'FOCUSER' in hdr:
        #    hdr['FOCNAME'] = hdr['FOCUSER']
        #    logger.debug("Mapped FOCUSER to FOCNAME")
        if 'IMAGETYP' in hdr:
            # Normalize IMAGETYP by removing spaces and converting to uppercase
            hdr['IMAGETYP'] = hdr['IMAGETYP'].replace(' ', '').upper()
            logger.debug(f"Normalized IMAGETYP to {hdr['IMAGETYP']}")
        if 'FILENAME' not in hdr:
            # Set FILENAME from state if missing
            hdr['FILENAME'] = state['header_filename']
            logger.debug(f"Set FILENAME to {state['header_filename']}")
        # Set NUMBER from state
        hdr['NUMBER'] = state['number']
        logger.debug(f"Set NUMBER to {state['number']}")
        if 'INSTRUME' in hdr:
            # Convert INSTRUME to uppercase
            hdr['INSTRUME'] = hdr['INSTRUME'].upper()
            logger.debug(f"Normalized INSTRUME to {hdr['INSTRUME']}")
        # Set default GAIN and EGAIN if missing
        hdr.setdefault('GAIN', -1)
        hdr.setdefault('EGAIN', 1)
        logger.debug("Applied default GAIN and EGAIN if missing")
        # Remove quotes from string values
        hdr = {k: v.strip('"').strip("'") if isinstance(v, str) else v for k, v in hdr.items()}
        logger.debug("Removed quotes from string values")
        
        # Replace empty SITELAT and SITELONG with config['defaults'] values
        #if 'SITELAT' in hdr and hdr['SITELAT'] == '':
        #    hdr['SITELAT'] = config['defaults'].get('SITELAT', '')
        #    logger.debug(f"Replaced empty SITELAT with config['defaults']['SITELAT']: {hdr['SITELAT']}")
        #if 'SITELONG' in hdr and hdr['SITELONG'] == '':
        #    hdr['SITELONG'] = config['defaults'].get('SITELONG', '')
        #    logger.debug(f"Replaced empty SITELONG with config['defaults']['SITELONG']: {hdr['SITELONG']}")

        logger.debug(f"Reduced header before limiting to wanted keys: {hdr}")
        # Reduce header to wanted keys, using defaults if missing
        reduced_hdr = {k: hdr.get(k, config['defaults'].get(k, '')) for k in state['wanted_keys'] if k in config['defaults'] or k in ['FILENAME', 'NUMBER']}
        logger.debug(f"Reduced header to wanted keys: {reduced_hdr}")
        
        # Convert data types
        return check_and_convert_data_types(reduced_hdr, state)
    except Exception as e:
        # Log and return empty dictionary on error
        logger.error(f"Error reducing headers: {str(e)}")
        return {}

def filter_and_remove_duplicates(headers_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Filters and removes duplicate LIGHT frames with WBPP postfixes.

    Removes duplicates by selecting preferred file extensions and handling processed files.

    Args:
        headers_df (pd.DataFrame): DataFrame containing header data with 'FILENAME' and 'IMAGETYP' columns.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        pd.DataFrame: Filtered DataFrame with duplicates removed.

    Raises:
        ValueError: If headers_df is not a DataFrame or required columns are missing.
    """
    try:
        # Validate input types
        if not isinstance(headers_df, pd.DataFrame):
            raise ValueError("headers_df must be a pandas DataFrame")
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")

        # Log start of duplicate filtering
        logger.info("Filtering and removing duplicate LIGHT frames")
        # Create a copy to avoid modifying the input DataFrame
        headers_df = headers_df.copy()
        # Remove masterlight frames
        headers_df = headers_df[headers_df['IMAGETYP'] != 'masterlight']

        # Check for required columns
        required_columns = {'FILENAME', 'IMAGETYP'}
        if not required_columns.issubset(headers_df.columns):
            missing = required_columns - set(headers_df.columns)
            logger.error(f"Missing required columns: {missing}")
            return headers_df

        # Extract base filename without calibration postfixes. Updated 2025-9-26 to make the selection case insensitive
        headers_df['base_filename'] = headers_df['FILENAME'].str.extract(r'(.+?)(?:_c.*)?(\.xisf|\.fits|\.fit|\.fts)', flags=re.IGNORECASE)[0]
        
        #headers_df['base_filename'] = headers_df['FILENAME'].str.extract(r'(.+?)(?:_c.*)?(\.xisf|\.fits|\.fit|\.fts)')[0]
        
        if headers_df['base_filename'].isna().all():
            logger.warning("No valid base filenames extracted")
            return headers_df

        # Check for duplicates
        if headers_df['base_filename'].duplicated().any():
            logger.info("Duplicates found and will be removed")
        else:
            logger.info("No duplicates found")

        # Initialize final DataFrame
        final_df = pd.DataFrame()
        # Process each unique base filename
        for base_filename in headers_df['base_filename'].unique():
            if pd.isna(base_filename):
                logger.warning("Skipping NaN base_filename")
                continue
            # Check preferred extensions in order
            for ext in ['.xisf', '.fits', '.fit', '.fts']:
                match = headers_df[headers_df['FILENAME'] == f'{base_filename}{ext}']
                if not match.empty:
                    final_df = pd.concat([final_df, match])
                    logger.debug(f"Selected {base_filename}{ext} for inclusion")
                    break
            else:
                # Fallback to _c.xisf if no standard extension found
                match_c = headers_df[headers_df['FILENAME'] == f'{base_filename}_c.xisf']
                if not match_c.empty:
                    final_df = pd.concat([final_df, match_c])
                    logger.debug(f"Selected {base_filename}_c.xisf as fallback")
                else:
                    # Use default filename if no match found
                    default_filename = f'{base_filename}.xisf'
                    default_row = headers_df[headers_df['base_filename'] == base_filename].iloc[0].copy()
                    default_row['FILENAME'] = default_filename
                    final_df = pd.concat([final_df, pd.DataFrame([default_row])])
                    logger.debug(f"Set default filename {default_filename}")

        # Drop temporary base_filename column
        final_df = final_df.drop(columns=['base_filename'], errors='ignore')
        logger.info("Completed duplicate filtering")
        return final_df.reset_index(drop=True)

    except Exception as e:
        # Log and return DataFrame on error
        logger.error(f"Error filtering and removing duplicates: {str(e)}")
        return headers_df

def deal_with_calibration_frames(df: pd.DataFrame, state: Dict[str, Any]) -> pd.DataFrame:
    """Removes unnecessary calibration frames based on exposure matching.

    Drops dark and bias frames if master versions exist or if exposures don’t match LIGHT frames.

    Args:
        df (pd.DataFrame): DataFrame containing header data with 'IMAGETYP' and 'EXPOSURE' columns.
        state (Dict[str, Any]): Dictionary containing headers state with logger, etc.

    Returns:
        pd.DataFrame: Filtered DataFrame with unnecessary calibration frames removed.

    Raises:
        ValueError: If df is not a DataFrame, state is not a dictionary, or required state keys are missing.
    """
    try:
        # Validate input types
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dictionary, got {type(state).__name__}")
        if not all(key in state for key in ['logger']):
            raise ValueError("state must contain 'logger' key")
        if not isinstance(state['logger'], logging.Logger):
            raise ValueError("state['logger'] must be a logging.Logger instance")

        # Extract logger
        logger = state['logger']
        # Log start of calibration frame processing
        logger.info("Processing calibration frames")
        # Create a copy to avoid modifying the input DataFrame
        df = df.copy()

        # Check for required columns
        required_columns = {'IMAGETYP', 'EXPOSURE'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            logger.error(f"Missing required columns: {missing}")
            return df

        # Convert IMAGETYP to lowercase for consistency
        df['IMAGETYP'] = df['IMAGETYP'].str.lower()
        # Get unique exposures for LIGHT and dark frames
        light_exposures = df[df['IMAGETYP'] == 'light']['EXPOSURE'].unique()
        dark_exposures = df[df['IMAGETYP'].str.contains('dark', case=False)]['EXPOSURE'].unique()

        # Remove dark frames with exposures not matching LIGHT frames
        for exposure_value in dark_exposures:
            if exposure_value not in light_exposures:
                logger.info(f"No light frame with exposure {exposure_value} found, dropping dark frames")
                df = df[~(df['IMAGETYP'].str.contains('dark', case=False) & (df['EXPOSURE'] == exposure_value))]

        # Remove non-master calibration frames if master versions exist
        master_types = ['masterbias', 'masterdark', 'masterflat', 'masterflatdark']
        for master_type in master_types:
            if any(master_type in s.lower() for s in df['IMAGETYP']):
                removeimgtype = master_type.replace("master", "")
                logger.info(f"{master_type} found, removing {removeimgtype} frames")
                df = df[df['IMAGETYP'] != removeimgtype]

        logger.info("Completed calibration frame processing")
        return df

    except Exception as e:
        # Log and return DataFrame on error
        logger.error(f"Error processing calibration frames: {str(e)}")
        return df

def deal_with_flat_frames(df: pd.DataFrame, state: Dict[str, Any]) -> pd.DataFrame:
    """Handles flat frames by removing unnecessary ones and consolidating master flats.

    Drops flat frames if master flats exist or if filters don’t match LIGHT frames, and sums master flat counts.

    Args:
        df (pd.DataFrame): DataFrame containing header data with 'IMAGETYP' and 'FILTER' columns.
        state (Dict[str, Any]): Dictionary containing headers state with logger, etc.

    Returns:
        pd.DataFrame: Filtered DataFrame with processed flat frames.

    Raises:
        ValueError: If df is not a DataFrame, state is not a dictionary, or required state keys are missing.
    """
    try:
        # Validate input types
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dictionary, got {type(state).__name__}")
        if not all(key in state for key in ['logger']):
            raise ValueError("state must contain 'logger' key")
        if not isinstance(state['logger'], logging.Logger):
            raise ValueError("state['logger'] must be a logging.Logger instance")

        # Extract logger
        logger = state['logger']
        # Log start of flat frame processing
        logger.info("Processing flat frames")
        # Create a copy to avoid modifying the input DataFrame
        df = df.copy()

        # Check for required columns
        required_columns = {'IMAGETYP', 'FILTER'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            logger.error(f"Missing required columns: {missing}")
            return df

        # Convert IMAGETYP to lowercase for consistency
        df['IMAGETYP'] = df['IMAGETYP'].str.lower()
        # Get unique filters for LIGHT and flat frames
        light_filters = df[df['IMAGETYP'] == 'light']['FILTER'].unique()
        flat_filters = df[df['IMAGETYP'].str.contains('flat', case=False)]['FILTER'].unique()

        # Remove flat frames with filters not matching LIGHT frames
        for filter_value in flat_filters:
            if filter_value not in light_filters:
                logger.info(f"No light frame with filter {filter_value} found, dropping flat frames")
                df = df[~(df['IMAGETYP'].str.contains('flat', case=False) & (df['FILTER'] == filter_value))]

        # Identify master flat filters
        master_flat_filters = df[(df['IMAGETYP'].str.contains('master', case=False) & df['IMAGETYP'].str.contains('flat', case=False))]['FILTER'].unique()
        for filter_value in master_flat_filters:
            logger.info(f"Master flat frame with filter {filter_value} found, dropping flat frames")

        # Remove non-master flat frames if master flats exist
        for filter_value in master_flat_filters:
            df = df[~((df['IMAGETYP'] == 'flat') & (df['FILTER'] == filter_value))]

        # Consolidate master flat frames by summing counts
        master_flat_df = df[df['IMAGETYP'].str.contains('master', case=False) & df['IMAGETYP'].str.contains('flat', case=False)].copy()
        if not master_flat_df.empty:
            # Sum NUMBER for each filter
            master_flat_df.loc[:, 'NUMBER'] = master_flat_df.groupby('FILTER')['NUMBER'].transform('sum')
            # Remove duplicates, keeping one row per filter
            master_flat_df = master_flat_df.drop_duplicates('FILTER')
            logger.info("Consolidated master flat frames")

        # Remove original master flat frames from DataFrame
        df = df[~(df['IMAGETYP'].str.contains('master', case=False) & df['IMAGETYP'].str.contains('flat', case=False))]
        if not master_flat_df.empty:
            # Reintegrate consolidated master flat frames
            df = pd.concat([df, master_flat_df], ignore_index=True)
            logger.info("Reintegrated consolidated master flat frames")

        logger.info("Completed flat frame processing")
        return df

    except Exception as e:
        # Log and return DataFrame on error
        logger.error(f"Error processing flat frames: {str(e)}")
        return df

def check_and_convert_data_types(d: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Converts data types of header dictionary values to match expected types.

    Handles DMS coordinates and date formatting, applying rounding where needed.

    Args:
        d (Dict[str, Any]): Header dictionary to process.
        state (Dict[str, Any]): Dictionary containing headers state with logger and dp keys.

    Returns:
        Dict[str, Any]: Dictionary with converted data types.

    Raises:
        ValueError: If d or state is not a dictionary, or required state keys are missing.
    """
    try:
        # Validate input types
        if not isinstance(d, dict):
            raise ValueError(f"d must be a dictionary, got {type(d).__name__}")
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dictionary, got {type(state).__name__}")
        if not all(key in state for key in ['logger', 'dp']):
            raise ValueError("state must contain 'logger' and 'dp' keys")
        if not isinstance(state['logger'], logging.Logger):
            raise ValueError("state['logger'] must be a logging.Logger instance")

        # Extract logger and decimal precision
        logger, dp = state['logger'], state['dp']
        # Log start of data type conversion
        logger.debug("Checking and converting data types")

        # Define helper function to convert DMS to decimal degrees
        def dms_to_decimal(dms_str: str) -> float:
            try:
                # Check for DMS format (e.g., "10d20m30s N")
                if 'd' in dms_str and 'm' in dms_str and 's' in dms_str:
                    parts = dms_str.replace('d', ' ').replace('m', ' ').replace('s', ' ').split()
                    degrees, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
                    direction = parts[3] if len(parts) == 4 else ''
                else:
                    # Assume space-separated DMS (e.g., "10 20 30")
                    degrees, minutes, seconds = map(float, dms_str.split())
                    direction = ''
                # Convert to decimal degrees
                decimal = degrees + minutes / 60 + seconds / 3600
                if direction in ['W', 'S']:
                    decimal = -decimal
                logger.debug(f"Converted DMS {dms_str} to decimal: {decimal}")
                return decimal
            except Exception as e:
                # Log error and return default value
                logger.error(f"Error converting DMS to decimal: {str(e)}")
                return 0.0

        # Define expected data types for header keys
        data_types = {
            'IMAGETYP': str,
            'EXPOSURE': float,
            'DATE-OBS': str,
            'XBINNING': int,
            'GAIN': int,
            'EGAIN': float,
            'XPIXSZ': float,
            'CCD-TEMP': float,
            'FOCALLEN': int,
            'FOCRATIO': float,
            'FILTER': str,
            'FOCTEMP': float,
            'SWCREATE': str,
            'HFR': float,
            'FWHM': float,
            'SITELAT': float,
            'SITELONG': float,
            'OBJECT': str,
            'FILENAME': str,
            'NUMBER': int,
            'INSTRUME': str,
            'SQM' : float,
            'BORTLE': float,
            'FOCTEMP': float,
            'FWHEEL' : str,
            'ROTNAME': str,
            'ROTATANG': float
        }

        # Process each key
        for key, expected_type in data_types.items():
            if key in d:
                v = d[key]
                if isinstance(v, str):
                    # Remove quotes from string values
                    v = v.replace('"', '').strip("'")
                    if key == 'DATE-OBS':
                        try:
                            # Remove microseconds and format date
                            if '.' in v:
                                v = v.split('.')[0]
                            v = datetime.strptime(v, "%Y-%m-%dT%H:%M:%S").strftime('%Y-%m-%dT%H:%M:%S')
                            logger.debug(f"Formatted DATE-OBS: {v}")
                        except ValueError as e:
                            logger.error(f"Error formatting DATE-OBS {v}: {str(e)}")
                    if key in ['SITELAT', 'SITELONG'] and ' ' in v:
                        # Convert DMS coordinates to decimal
                        v = dms_to_decimal(v)

                try:
                    # Convert value to expected type
                    if expected_type == float:
                        d[key] = round(float(v), dp)
                    elif expected_type == int:
                        d[key] = int(float(v))
                    elif expected_type == str:
                        d[key] = str(v)
                    logger.debug(f"Converted {key} to {expected_type.__name__}: {d[key]}")
                except (ValueError, TypeError) as e:
                    # Log error for type conversion
                    logger.error(f"Error converting {key} to {expected_type.__name__}: {str(e)}")
            else:
                # Log missing key
                logger.warning(f"Key {key} not found in dictionary")
        logger.debug("Completed data type conversion")
        return d

    except Exception as e:
        # Log and return dictionary on error
        logger.error(f"Error converting data types: {str(e)}")
        return d

def check_camera_gains(hdrs: pd.DataFrame, state: Dict[str, Any]) -> pd.DataFrame:
    """Checks and updates camera gains in the headers DataFrame.

    Ensures consistent GAIN and EGAIN values, applying defaults for invalid or missing entries.

    Args:
        hdrs (pd.DataFrame): DataFrame containing header data with 'INSTRUME', 'GAIN', and 'EGAIN' columns.
        state (Dict[str, Any]): Dictionary containing headers state with config, logger, etc.

    Returns:
        pd.DataFrame: Updated DataFrame with corrected gains.

    Raises:
        ValueError: If hdrs is not a DataFrame, state is not a dictionary, or required state keys are missing.
    """
    try:
        # Validate input types
        if not isinstance(hdrs, pd.DataFrame):
            raise ValueError("hdrs must be a pandas DataFrame")
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dictionary, got {type(state).__name__}")
        if not all(key in state for key in ['config', 'logger']):
            raise ValueError("state must contain 'config' and 'logger' keys")
        if not isinstance(state['logger'], logging.Logger):
            raise ValueError("state['logger'] must be a logging.Logger instance")

        # Extract logger and config
        logger, config = state['logger'], state['config']
        # Log start of gain checking
        logger.info("Checking camera gains")
        # Create a copy to avoid modifying the input DataFrame
        hdrs = hdrs.copy()

        # Check for required columns
        required_columns = {'INSTRUME', 'GAIN', 'EGAIN'}
        if not required_columns.issubset(hdrs.columns):
            missing = required_columns - set(hdrs.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Replace 'No Data' with 0 for GAIN
        hdrs['GAIN'] = hdrs['GAIN'].replace('No Data', 0)
        # Get unique valid gain combinations
        unique_df = hdrs[hdrs['GAIN'] >= 0][['INSTRUME', 'GAIN', 'EGAIN']].drop_duplicates()
        logger.info(f"Unique cameras: {unique_df['INSTRUME'].unique()}")
        logger.info(f"Unique gains: {unique_df['GAIN'].unique()}")
        logger.info(f"Unique egains: {unique_df['EGAIN'].unique()}")

        # Apply default values for invalid or missing INSTRUME, GAIN, EGAIN
        unique_df['INSTRUME'] = unique_df['INSTRUME'].apply(lambda x: config['defaults']['INSTRUME'] if pd.isna(x) or x == '' else x)
        unique_df['EGAIN'] = unique_df['EGAIN'].apply(lambda x: config['defaults']['EGAIN'] if pd.isna(x) or x == '' or x < 0 else x)
        unique_df['GAIN'] = unique_df['GAIN'].apply(lambda x: config['defaults']['GAIN'] if pd.isna(x) or x == '' or x < 0 else x)
        logger.info("Applied default values for empty or invalid INSTRUME, GAIN, and EGAIN")

        # Define function to update GAIN based on EGAIN
        def update_gain(row):
            try:
                if row['GAIN'] <= 0:
                    # Find matching EGAIN in unique_df
                    match = unique_df[unique_df['EGAIN'] == row['EGAIN']]
                    if not match.empty:
                        return match['GAIN'].iloc[0]
                return row['GAIN']
            except Exception as e:
                logger.error(f"Error updating GAIN for row {row.name}: {str(e)}")
                return row['GAIN']

        # Apply gain updates
        hdrs['GAIN'] = hdrs.apply(update_gain, axis=1)
        logger.info("Completed camera gain updates")
        return hdrs

    except Exception as e:
        # Log and return DataFrame on error
        logger.error(f"Error checking camera gains: {str(e)}")
        return hdrs

def get_HFR(df: pd.DataFrame, state: Dict[str, Any]) -> pd.DataFrame:
    """Extracts HFR, IMSCALE, and FWHM from LIGHT frames in the DataFrame.

    Calculates Half Flux Radius (HFR), image scale, and Full Width at Half Maximum (FWHM)
    for LIGHT frames based on filename or default values.

    Args:
        df (pd.DataFrame): DataFrame containing header data with 'IMAGETYP', 'FILENAME', 'FOCALLEN', and 'XPIXSZ' columns.
        state (Dict[str, Any]): Dictionary containing headers state with config, logger, etc.

    Returns:
        pd.DataFrame: Updated DataFrame with 'HFR', 'IMSCALE', and 'FWHM' columns.

    Raises:
        ValueError: If df is not a DataFrame, state is not a dictionary, or required state keys are missing.
    """
    try:
        # Validate input types
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dictionary, got {type(state).__name__}")
        if not all(key in state for key in ['config', 'logger']):
            raise ValueError("state must contain 'config' and 'logger' keys")
        if not isinstance(state['logger'], logging.Logger):
            raise ValueError("state['logger'] must be a logging.Logger instance")

        # Extract logger and config
        logger, config = state['logger'], state['config']
        # Log start of HFR extraction
        logger.info("Starting HFR extraction")
        # Create a copy to avoid modifying the input DataFrame
        df = df.copy()

        # Check for required columns
        required_columns = {'IMAGETYP', 'FILENAME', 'FOCALLEN', 'XPIXSZ'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Get default HFR from config
        hfr_set = float(config['defaults']['HFR'])
        # Count total LIGHT frames
        total_frames = len(df[df['IMAGETYP'] == 'LIGHT'])
        counter = 0

        # Process each LIGHT frame
        for index, row in df[df['IMAGETYP'] == 'LIGHT'].iterrows():
            try:
                counter += 1
                file_path = row['FILENAME']
                # Extract HFR from filename if present
                hfr_match = re.search(r'HFR_([0-9.]+)', file_path)
                hfr = float(hfr_match.group(1)) if hfr_match and float(hfr_match.group(1)) > 0 else hfr_set
                # Convert focal length and pixel size to float
                focal_length = float(row['FOCALLEN'])
                pixel_size = float(row['XPIXSZ'])
                # Calculate image scale (arcseconds per pixel)
                imscale = pixel_size / focal_length * 206.265
                # Calculate FWHM (arcseconds)
                fwhm = hfr * imscale * 2 if hfr >= 0.0 else 0.0

                # Update DataFrame with calculated values
                df.loc[index, 'HFR'] = round(hfr, 2)
                df.loc[index, 'IMSCALE'] = round(imscale, 2)
                df.loc[index, 'FWHM'] = round(fwhm, 2)
                logger.debug(f"Processed LIGHT frame {counter}/{total_frames}: HFR={hfr:.2f}, IMSCALE={imscale:.2f}, FWHM={fwhm:.2f}")
            except (ValueError, TypeError) as e:
                # Log error for individual row processing
                logger.error(f"Error processing HFR for index {index} ({file_path}): {str(e)}")

        # Update state with processed frame count
        state['number_of_images_processed'] = counter
        logger.info(f"Completed HFR extraction: mean HFR={df['HFR'].mean():.2f}, "
                   f"mean IMSCALE={df['IMSCALE'].mean():.2f}, mean FWHM={df['FWHM'].mean():.2f}")
        return df

    except Exception as e:
        # Log and return DataFrame on error
        logger.error(f"Error extracting HFR: {str(e)}")
        return df