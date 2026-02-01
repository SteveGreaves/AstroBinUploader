import logging
import os
from io import StringIO
from typing import Any, Dict, Optional, Tuple
from configobj import ConfigObj

config_version = '1.4.0'
#
# Date: Sunday 1st February 2026
# Modification : v1.4.2 Restoration & Logic Overhaul.
# 1. Implemented 'Integer Gain Handshake' for calibration matching.
# 2. Restored Heuristic Date Fallback for missing FITS headers.
# 3. Optimized I/O using Pandas engine for RAID 0 performance.
# Author : SDG & Gemini

def cast_value(value: Any, logger: Optional[logging.Logger] = None) -> Any:
    """Casts a value to an appropriate type (list, float, int, or string) based on its format.

    Attempts to convert comma-separated strings to lists, strings with periods to floats,
    numeric strings to integers, or retains the original value if casting fails.
    Recursively processes lists to handle nested values.

    Args:
        value (Any): The value to cast, which can be any type (string, list, number, etc.).
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.

    Returns:
        Any: The cast value, which may be a list, float, int, or the original value if casting fails.

    Raises:
        ValueError: If logger is provided but is not a valid logging.Logger instance.
    """
    # Validate logger if provided
    if logger and not isinstance(logger, logging.Logger):
        error_msg = "logger must be a logging.Logger instance"
        logger.error(error_msg) if logger else None
        raise ValueError(error_msg)

    try:
        # Handle list values by recursively casting each item
        if isinstance(value, list):
            logger.info(f"Casting list value {value}") if logger else None
            return [cast_value(item, logger) for item in value]
        # Handle comma-separated strings by splitting into a list
        if isinstance(value, str) and ',' in value:
            logger.info(f"Casting comma-separated string: {value}") if logger else None
            return [cast_value(item.strip(), logger) for item in value.split(',')]
        # Handle strings with a period as potential floats
        if isinstance(value, str) and '.' in value or (isinstance(value, float) and not value.is_integer()):
            try:
                cast_result = float(value)
                logger.info(f"Cast {value} to float: {cast_result}") if logger else None
                return cast_result
            except ValueError:
                logger.info(f"Failed to cast {value} to float, returning as string") if logger else None
                return value
        # Attempt to cast to integer for numeric strings
        try:
            cast_result = int(value)
            logger.info(f"Cast {value} to int: {cast_result}") if logger else None
            return cast_result
        except (ValueError, TypeError):
            # Return original value if casting fails
            logger.info(f"Returning {value} as is") if logger else None
            return value
    except Exception as e:
        # Log and return original value on unexpected errors
        logger.error(f"Error casting value {value}: {str(e)}") if logger else None
        return value

def get_default_ini_string(logger: Optional[logging.Logger] = None) -> str:
    """Generates a default configuration string for the INI file.

    Provides a template configuration with default values for FITS keywords, filter mappings,
    API secrets, and sites, used when initializing a new configuration file.

    Args:
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.

    Returns:
        str: A string containing the default INI configuration.

    Raises:
        ValueError: If logger is provided but is not a valid logging.Logger instance.
    """
    # Validate logger if provided
    if logger and not isinstance(logger, logging.Logger):
        error_msg = "logger must be a logging.Logger instance"
        logger.error(error_msg) if logger else None
        raise ValueError(error_msg)

    # Log start of default INI string generation
    logger.info("Generating default INI configuration string") if logger else None
    # Define default INI configuration string
    ini_string = """
    [defaults]
    IMAGETYP = LIGHT
    EXPOSURE = 0.0
    DATE-OBS = 2023-01-01
    XBINNING = 1
    GAIN = -1
    EGAIN = -1
    INSTRUME = None
    TELESCOP = None
    FOCNAME = None
    FWHEEL = None
    ROTNAME = None
    ROTANTANG = 0
    XPIXSZ = 3.76
    CCD-TEMP = -10
    FOCALLEN = 540
    FOCRATIO = 5.4
    SITE = None
    SITELAT = 0.0
    SITELONG = 0.0
    BORTLE = 0.0
    SQM = 0.0
    FILTER = No Filter
    OBJECT = No target
    FOCTEMP = 20
    HFR = 1.6
    FWHM = 0
    SWCREATE = Unknown package
    USEOBSDATE = TRUE
    [override]
    [filters]
    Ha = 4663
    SII = 4844
    OIII = 4752
    Red = 4649
    Green = 4643
    Blue = 4637
    Lum = 2906
    CLS = 4632
    [secret]
    xxxxxxxxxxxxxxxx = https://www.lightpollutionmap.info/QueryRaster/
    EMAIL_ADDRESS = id@provider.com
    [sites]
    """
    # Log completion of INI string generation
    logger.debug("Default INI configuration string generated") if logger else None
    return ini_string

def cast_section(section: Dict[Any, Any], logger: Optional[logging.Logger] = None) -> None:
    """Recursively casts values in a configuration section dictionary to appropriate types.

    Processes nested dictionaries and applies type casting (list, float, int, or string) to each value
    using the cast_value function.

    Args:
        section (Dict[Any, Any]): Configuration section dictionary to cast values for.
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.

    Raises:
        ValueError: If section is not a dictionary or logger is invalid.
    """
    try:
        # Validate input types
        if logger and not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(section, dict):
            raise ValueError(f"section must be a dictionary, got {type(section).__name__}")

        # Log start of section casting
        logger.info("Casting configuration section values") if logger else None
        # Process each key-value pair in the section
        for key, value in section.items():
            if isinstance(value, dict):
                # Recursively cast nested dictionaries
                logger.debug(f"Recursively casting subsection: {key}") if logger else None
                cast_section(value, logger)
            else:
                # Cast individual value
                section[key] = cast_value(value, logger)
                logger.debug(f"Cast value for key {key}: {section[key]}") if logger else None
        logger.info("Completed casting configuration section") if logger else None

    except Exception as e:
        # Log and raise error
        logger.error(f"Error casting section: {str(e)}") if logger else None
        raise

def initialise_config(filename: str, logger: Optional[logging.Logger] = None) -> Tuple[ConfigObj, bool]:
    """Initializes a configuration file, creating it with default values if it does not exist.

    Reads an existing configuration file or creates a new one using the default INI string.
    Normalizes and casts configuration values to appropriate types.

    Args:
        filename (str): Path to the configuration file (e.g., config.ini).
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.

    Returns:
        Tuple[ConfigObj, bool]: A tuple containing:
            - ConfigObj: The initialized configuration object.
            - bool: True if the configuration file was created or modified, False otherwise.

    Raises:
        ValueError: If filename is not a non-empty string or logger is invalid.
        OSError: If the configuration file cannot be read or written.
    """
    try:
        # Validate input types
        if logger and not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string")

        # Log start of configuration initialization
        logger.info("Initializing configuration file") if logger else None
        # Initialize change flag
        change = False
        # Get default INI configuration string
        ini_string = get_default_ini_string(logger)

        try:
            # Check if configuration file exists
            if not os.path.exists(filename):
                logger.info(f"No configuration file found at {filename}, creating new one") if logger else None
                # Create new ConfigObj from default INI string
                config = ConfigObj(StringIO(ini_string), encoding='utf-8')
                config.filename = filename
                try:
                    # Write new configuration file
                    config.write()
                    change = True
                    logger.info(f"Created configuration file: {filename}") if logger else None
                except OSError as e:
                    logger.error(f"Failed to write configuration file {filename}: {str(e)}") if logger else None
                    raise OSError(f"Failed to write configuration file {filename}: {str(e)}")
            else:
                # Read existing configuration file
                config = ConfigObj(filename, encoding='utf-8')
                logger.info(f"Configuration file found at {filename}") if logger else None

            # Correct configuration structure and normalize keys
            correct_config(config, logger)
            # Cast configuration values to appropriate types
            cast_section(config, logger)
            logger.info(f"Configuration object: {dict(config)}") if logger else None
            return config, change

        except OSError as e:
            # Log file access errors
            logger.error(f"Error accessing configuration file {filename}: {str(e)}") if logger else None
            raise

    except Exception as e:
        # Log and raise unexpected errors
        logger.error(f"Error initializing configuration: {str(e)}") if logger else None
        raise

def update_config(config: ConfigObj, section: str, subsection: Optional[str], keys_values: Dict[str, Any], logger: Optional[logging.Logger] = None) -> bool:
    """Updates a configuration section or subsection with new key-value pairs.

    Adds or updates a section and optional subsection in the configuration, writes changes to the file,
    and casts values to appropriate types.

    Args:
        config (ConfigObj): Configuration object to update.
        section (str): Name of the configuration section to update (e.g., 'sites').
        subsection (Optional[str]): Name of the subsection to update, or None to generate a default name.
        keys_values (Dict[str, Any]): Dictionary of key-value pairs to add or update.
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.

    Returns:
        bool: True if changes were made to the configuration, False otherwise.

    Raises:
        ValueError: If config is not a ConfigObj, section/subsection is invalid, or keys_values is not a dictionary.
        OSError: If the configuration file cannot be written.
    """
    try:
        # Validate input types
        if logger and not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(config, ConfigObj):
            raise ValueError(f"config must be a ConfigObj, got {type(config).__name__}")
        if not isinstance(section, str) or not section.strip():
            raise ValueError("section must be a non-empty string")
        if subsection is not None and (not isinstance(subsection, str) or not subsection.strip()):
            raise ValueError("subsection must be a non-empty string or None")
        if not isinstance(keys_values, dict):
            raise ValueError(f"keys_values must be a dictionary, got {type(keys_values).__name__}")

        # Log start of configuration update
        logger.info(f"Updating configuration file: section={section}, subsection={subsection}") if logger else None
        # Initialize change flag
        change = False

        # Add section if it does not exist
        if section not in config:
            logger.info(f"Adding new section: {section}") if logger else None
            config[section] = {}
            change = True

        # Generate default subsection name if not provided
        if not subsection:
            subsection = f"site{len(config[section]) + 1}"
            logger.info(f"No subsection provided, using default: {subsection}") if logger else None
            change = True

        # Add empty line before section for formatting
        config.comments[section] = ['']

        # Add or update subsection with key-value pairs
        if subsection not in config[section]:
            config[section][subsection] = keys_values
            change = True
            logger.info(f"Added subsection {subsection} to section {section} with keys: {keys_values}") if logger else None
        else:
            # Update existing keys if values differ
            for key, value in keys_values.items():
                if key not in config[section][subsection] or config[section][subsection][key] != value:
                    config[section][subsection][key] = value
                    change = True
                    logger.info(f"Updated {key} in {section}/{subsection} to {value}") if logger else None

        # Save changes and cast values if modified
        if change:
            cast_section(config, logger)
            try:
                # Write updated configuration to file
                config.write()
                logger.info(f"Saved updated configuration to {config.filename}") if logger else None
            except OSError as e:
                logger.error(f"Failed to write configuration file {config.filename}: {str(e)}") if logger else None
                raise OSError(f"Failed to write configuration file {config.filename}: {str(e)}")
        else:
            logger.info(f"No changes made to configuration file {config.filename}") if logger else None

        return change

    except Exception as e:
        # Log and raise error
        logger.error(f"Error updating configuration: {str(e)}") if logger else None
        raise

def get_config(config: ConfigObj, logger: Optional[logging.Logger] = None) -> Dict:
    """Converts a ConfigObj to a dictionary representation.

    Extracts the configuration data into a standard Python dictionary for easier access.

    Args:
        config (ConfigObj): Configuration object to convert.
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.

    Returns:
        Dict: Dictionary representation of the configuration.

    Raises:
        ValueError: If config is not a ConfigObj or logger is invalid.
    """
    try:
        # Validate input types
        if logger and not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(config, ConfigObj):
            raise ValueError(f"config must be a ConfigObj, got {type(config).__name__}")

        # Log start of configuration retrieval
        logger.info("Retrieving configuration as dictionary") if logger else None
        # Convert ConfigObj to dictionary
        config_dict = dict(config)
        # Log resulting dictionary for debugging
        logger.debug(f"Configuration dictionary: {config_dict}") if logger else None
        return config_dict

    except Exception as e:
        # Log and raise error
        logger.error(f"Error retrieving configuration: {str(e)}") if logger else None
        raise

def read_config(filename: str, logger: Optional[logging.Logger] = None) -> Dict:
    """Reads a configuration file into a dictionary.

    Loads the INI file and converts it to a Python dictionary.

    Args:
        filename (str): Path to the configuration file.
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.

    Returns:
        Dict: Dictionary representation of the configuration file.

    Raises:
        ValueError: If filename is not a non-empty string or logger is invalid.
        OSError: If the configuration file cannot be read.
    """
    try:
        # Validate input types
        if logger and not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string")

        # Log start of configuration reading
        logger.info(f"Reading configuration file: {filename}") if logger else None
        try:
            # Read configuration file into ConfigObj
            config = ConfigObj(filename, encoding='utf-8')
            # Convert to dictionary
            config_dict = dict(config)
            logger.debug(f"Configuration dictionary: {config_dict}") if logger else None
            return config_dict
        except OSError as e:
            # Log file access errors
            logger.error(f"Failed to read configuration file {filename}: {str(e)}") if logger else None
            raise OSError(f"Failed to read configuration file {filename}: {str(e)}")

    except Exception as e:
        # Log and raise unexpected errors
        logger.error(f"Error reading configuration: {str(e)}") if logger else None
        raise

def correct_config(config: ConfigObj, logger: Optional[logging.Logger] = None) -> None:
    """Normalizes and corrects the configuration by ensuring section names and required keys.

    Converts section names to lowercase without spaces, ensures required keys in 'defaults' and 'sites',
    and saves changes to the configuration file.

    Args:
        config (ConfigObj): Configuration object to correct.
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.

    Raises:
        ValueError: If config is not a ConfigObj or logger is invalid.
        OSError: If the configuration file cannot be written.
    """
    try:
        # Validate input types
        if logger and not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(config, ConfigObj):
            raise ValueError(f"config must be a ConfigObj, got {type(config).__name__}")

        # Log start of configuration correction
        logger.info("Starting configuration correction") if logger else None

        # Parse default INI string for reference
        ini_string_config = ConfigObj(StringIO(get_default_ini_string(logger)), encoding='utf-8')
        logger.info("Parsed default INI string into ConfigObj") if logger else None

        # Normalize section names
        normalize_section_names(config, logger)
        # Reconstruct configuration to match reference structure
        reconstruct_config(config, ini_string_config, logger)

        # Normalize keys in 'defaults' section
        if 'defaults' in config:
            config['defaults'] = {
                key.upper().replace(' ', ''): value
                for key, value in config['defaults'].items()
            }
            logger.info("Normalized keys in 'defaults' section to uppercase without spaces") if logger else None
            # Ensure order and presence of default keys
            ensure_default_keys_order(config['defaults'], ini_string_config['defaults'], logger)

        # Process 'sites' section
        process_sites_section(config, logger)

        try:
            # Save corrected configuration
            config.write()
            logger.info(f"Saved corrected configuration to {config.filename}") if logger else None
        except OSError as e:
            logger.error(f"Failed to write configuration file {config.filename}: {str(e)}") if logger else None
            raise OSError(f"Failed to write configuration file {config.filename}: {str(e)}")

        logger.info("Completed configuration correction") if logger else None

    except Exception as e:
        # Log and raise error
        logger.error(f"Error correcting configuration: {str(e)}") if logger else None
        raise

def normalize_section_names(section: ConfigObj, logger: Optional[logging.Logger] = None) -> None:
    """Normalizes section names in a configuration by converting to lowercase and removing spaces.

    Updates section names in-place to ensure consistency.

    Args:
        section (ConfigObj): Configuration section to normalize.
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.

    Raises:
        ValueError: If section is not a ConfigObj or logger is invalid.
    """
    try:
        # Validate input types
        if logger and not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(section, ConfigObj):
            raise ValueError(f"section must be a ConfigObj, got {type(section).__name__}")

        # Log start of section name normalization
        logger.info("Normalizing section names") if logger else None
        # Process each section key
        for key in list(section.keys()):
            if isinstance(section[key], ConfigObj):
                # Normalize section name to lowercase without spaces
                normalized_key = key.lower().replace(' ', '')
                if key != normalized_key:
                    # Rename section if changed
                    section[normalized_key] = section.pop(key)
                    logger.info(f"Normalized section name from {key} to {normalized_key}") if logger else None
        logger.info("Completed normalizing section names") if logger else None

    except Exception as e:
        # Log and raise error
        logger.error(f"Error normalizing section names: {str(e)}") if logger else None
        raise

def reconstruct_config(config: ConfigObj, reference_config: ConfigObj, logger: Optional[logging.Logger] = None) -> None:
    """Reconstructs the configuration to match the structure of a reference configuration.

    Copies sections from the current or reference configuration, preserving existing data
    and ensuring the configuration has the expected structure.

    Args:
        config (ConfigObj): Configuration object to reconstruct.
        reference_config (ConfigObj): Reference configuration providing the desired structure.
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.

    Raises:
        ValueError: If config or reference_config is not a ConfigObj, or logger is invalid.
    """
    try:
        # Validate input types
        if logger and not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(config, ConfigObj) or not isinstance(reference_config, ConfigObj):
            raise ValueError("config and reference_config must be ConfigObj instances")

        # Log start of configuration reconstruction
        logger.info("Reconstructing configuration") if logger else None
        # Initialize new configuration
        new_config = ConfigObj()

        # Copy sections from reference or current configuration
        for section in reference_config:
            if section in config:
                # Use existing section if present
                new_config[section] = config[section]
                logger.info(f"Copied section {section} from current configuration") if logger else None
            else:
                # Use reference section if missing
                new_config[section] = reference_config[section]
                logger.info(f"Copied section {section} from reference configuration") if logger else None

        # Clear and update current configuration
        config.clear()
        logger.info("Cleared current configuration") if logger else None
        config.update(new_config)
        logger.info("Updated configuration with new structure") if logger else None

    except Exception as e:
        # Log and raise error
        logger.error(f"Error reconstructing configuration: {str(e)}") if logger else None
        raise

def process_sites_section(config: ConfigObj, logger: Optional[logging.Logger] = None) -> None:
    """Normalizes and processes the 'sites' section of the configuration.

    Normalizes subsection names and keys to lowercase without spaces, ensures required keys
    (latitude, longitude, bortle, sqm), and adds missing keys with default values.

    Args:
        config (ConfigObj): Configuration object containing the 'sites' section.
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.

    Raises:
        ValueError: If config is not a ConfigObj or logger is invalid.
    """
    try:
        # Validate input types
        if logger and not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(config, ConfigObj):
            raise ValueError(f"config must be a ConfigObj, got {type(config).__name__}")

        # Log start of sites section processing
        logger.info("Processing 'sites' section") if logger else None
        # Parse default INI string for reference values
        ini_string_config = ConfigObj(StringIO(get_default_ini_string(logger)), encoding='utf-8')

        # Process sites section if present
        if 'sites' in config:
            # Iterate over subsections
            for subsection in list(config['sites'].keys()):
                # Strip whitespace from subsection name
                stripped_subsection = subsection.strip()
                if stripped_subsection != subsection:
                    config['sites'][stripped_subsection] = config['sites'].pop(subsection)
                    logger.info(f"Stripped spaces from subsection name: {stripped_subsection}") if logger else None

                # Normalize keys to lowercase without spaces
                config['sites'][stripped_subsection] = {
                    key.lower().replace(' ', ''): value
                    for key, value in config['sites'][stripped_subsection].items()
                }
                logger.info(f"Normalized keys in subsection: {stripped_subsection}") if logger else None

                # Define required keys and default values
                required_keys = ['latitude', 'longitude', 'bortle', 'sqm']
                default_values = {
                    'latitude': ini_string_config['defaults'].get('SITELAT', 0.0),
                    'longitude': ini_string_config['defaults'].get('SITELONG', 0.0),
                    'bortle': ini_string_config['defaults'].get('BORTLE', 4),
                    'sqm': ini_string_config['defaults'].get('SQM', 20.5)
                }

                # Ensure required keys are present
                new_dict = {}
                for key in required_keys:
                    new_dict[key] = config['sites'][stripped_subsection].get(key, default_values[key])
                # Preserve additional keys
                for key, value in config['sites'][stripped_subsection].items():
                    if key not in required_keys:
                        new_dict[key] = value

                # Update subsection with normalized keys
                config['sites'][stripped_subsection] = new_dict
                logger.info(f"Updated subsection {stripped_subsection} with required keys") if logger else None

        logger.info("Completed processing 'sites' section") if logger else None

    except Exception as e:
        # Log and raise error
        logger.error(f"Error processing 'sites' section: {str(e)}") if logger else None
        raise

def ensure_default_keys_order(defaults_section: Dict, reference_defaults: Dict, logger: Optional[logging.Logger] = None) -> None:
    """Ensures the order and presence of keys in the 'defaults' section of the configuration.

    Reorders keys to match the reference configuration and adds missing keys with default values.

    Args:
        defaults_section (Dict): 'defaults' section of the configuration to process.
        reference_defaults (Dict): Reference 'defaults' section for key order and default values.
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.

    Raises:
        ValueError: If defaults_section or reference_defaults is not a dictionary, or logger is invalid.
    """
    try:
        # Validate input types
        if logger and not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(defaults_section, dict) or not isinstance(reference_defaults, dict):
            raise ValueError("defaults_section and reference_defaults must be dictionaries")

        # Log start of default keys ordering
        logger.info("Ensuring default keys order") if logger else None
        # Initialize ordered keys dictionary
        ordered_keys = {}

        # Copy keys from reference configuration
        for key in reference_defaults:
            # Normalize key to uppercase without spaces
            normalized_key = key.upper().replace(' ', '')
            if normalized_key in defaults_section:
                # Use existing value if present
                ordered_keys[normalized_key] = defaults_section[normalized_key]
                logger.info(f"Copied key {normalized_key} from defaults section") if logger else None
            else:
                # Use reference value if missing
                ordered_keys[normalized_key] = reference_defaults[key]
                logger.info(f"Added missing key {normalized_key} from reference defaults") if logger else None

        # Preserve additional keys in defaults section
        for key in defaults_section:
            if key not in ordered_keys:
                ordered_keys[key] = defaults_section[key]
                logger.info(f"Added extra key {key} to ordered keys") if logger else None

        # Update defaults section with ordered keys
        defaults_section.clear()
        defaults_section.update(ordered_keys)
        logger.info("Updated defaults section with ordered keys") if logger else None

    except Exception as e:
        # Log and raise error
        logger.error(f"Error ensuring default keys order: {str(e)}") if logger else None
        raise