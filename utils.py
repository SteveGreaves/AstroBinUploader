import logging
import os
import pandas as pd
from datetime import datetime
from typing import Tuple, Union
import numpy as np

utils_version = '1.4.1'

def initialise_logging(log_filename: str, logger: logging.Logger = None) -> logging.Logger:
    """Initializes logging for the application with robust error handling.

    Creates a logger that writes to a specified file with a consistent format.
    Clears the existing log file and configures the logger with a file handler.

    Args:
        log_filename (str): Path to the log file.
        logger (logging.Logger, optional): Logger for pre-initialization errors. Defaults to None

    Returns:
        logging.Logger: Configured logger object.

    Raises:
        ValueError: If log_filename is empty or not a string.
        OSError: If unable to create or write to the log file.
    """
    errors = []

    try:
        if not isinstance(log_filename, str) or not log_filename.strip():
            error_msg = "log_filename must be a non-empty string"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)

        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_filename)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                if logger:
                    logger.info(f"Created log directory: {log_dir}")
            except OSError as e:
                error_msg = f"Failed to create log directory {log_dir}: {str(e)}"
                if logger:
                    logger.error(error_msg)
                raise OSError(error_msg)

        # Clear existing log file
        try:
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write('')
            if logger:
                logger.info(f"Cleared log file: {log_filename}")
        except OSError as e:
            error_msg = f"Failed to clear log file {log_filename}: {str(e)}"
            if logger:
                logger.error(error_msg)
            raise OSError(error_msg)

        # Configure logger
        new_logger = logging.getLogger(__name__)
        new_logger.handlers.clear()  # Clear existing handlers
        new_logger.setLevel(logging.INFO)

        try:
            handler = logging.FileHandler(log_filename, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - Line: %(lineno)d - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            new_logger.addHandler(handler)
        except OSError as e:
            error_msg = f"Failed to set up file handler for {log_filename}: {str(e)}"
            if logger:
                logger.error(error_msg)
            raise OSError(error_msg)

        # Log any errors that occurred before logger was fully initialized
        for error in errors:
            new_logger.error(str(error))
        new_logger.info("Logging initialized successfully")
        return new_logger

    except Exception as e:
        error_msg = f"Critical error initializing logging: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise type(e)(error_msg) from e

def seconds_to_hms(seconds: Union[int, float], logger: logging.Logger, aligned: bool = False) -> str:
    """Converts seconds to a string formatted as hours, minutes, and seconds.

    Args:
        seconds (Union[int, float]): Number of seconds to convert.
        logger (logging.Logger): Logger for logging messages.
        aligned (bool, optional): If True, returns aligned format with fixed-width fields.
            Defaults to False.

    Returns:
        str: Formatted string in hours, minutes, and seconds.

    Raises:
        ValueError: If seconds is negative or not a number, or if logger is invalid.
    """
    try:
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(seconds, (int, float)):
            raise ValueError(f"seconds must be a number, got {type(seconds).__name__}")
        if seconds < 0:
            raise ValueError("seconds cannot be negative")

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = float(seconds % 60)

        formatted_time = (
            f"{hours:>6} hrs {minutes:>6} mins {secs:>6.2f} secs" if aligned
            else f"{hours} hrs {minutes} mins {secs:.2f} secs"
        )
        logger.debug(f"Converted {seconds} seconds to HMS: {formatted_time}")
        return formatted_time

    except Exception as e:
        logger.error(f"Error converting {seconds} seconds to HMS: {str(e)}")
        return "Invalid time format"

def process_image_type(group: pd.DataFrame, imagetype: str, logger: logging.Logger) -> Tuple[str, float]:
    """Processes a specific image type and calculates total exposure time.

    Generates a formatted summary of the image type's details, including frame count,
    exposure time, and other relevant parameters, grouped by target or gain.

    Args:
        group (pd.DataFrame): DataFrame containing grouped observation data.
        imagetype (str): Type of image to process (e.g., 'LIGHT', 'DARK', 'FLAT').
        logger (logging.Logger): Logger for logging messages.

    Returns:
        Tuple[str, float]: Formatted summary string and total exposure time in seconds.

    Raises:
        ValueError: If group is not a DataFrame, imagetype is invalid, or required columns are missing.
    """
    try:
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(group, pd.DataFrame):
            raise ValueError("group must be a pandas DataFrame")
        if not isinstance(imagetype, str) or not imagetype.strip():
            raise ValueError("imagetype must be a non-empty string")

        valid_imagetypes = {'LIGHT', 'LIGHTFRAME', 'DARK', 'MASTERDARK', 'BIAS', 'MASTERBIAS', 'FLAT', 'MASTERFLAT'}
        if imagetype not in valid_imagetypes:
            raise ValueError(f"Invalid imagetype: {imagetype}. Must be one of {valid_imagetypes}")

        required_columns = {'imageType', 'number', 'duration'}
        if not required_columns.issubset(group.columns):
            missing = required_columns - set(group.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Processing image type: {imagetype}")
        lines = []
        total_exposure_time = 0.0

        image_group = group[group['imageType'] == imagetype].copy()
        if image_group.empty:
            logger.warning(f"No frames found for image type: {imagetype}")
            return f"\nNo {imagetype} frames found\n", 0.0

        if imagetype in {'LIGHT', 'LIGHTFRAME'}:
            required_columns = {'target', 'filter', 'gain', 'egain', 'meanFwhm', 'temperature', 'sensorCooling'}
            if not required_columns.issubset(image_group.columns):
                missing = required_columns - set(image_group.columns)
                logger.error(f"Missing required columns for LIGHT processing: {missing}")
                return f"\nError processing {imagetype}: Missing columns {missing}\n", 0.0

            format_string = " {:<8} {:<8} {:<8} {:<12} {:<12} {:<12} {:<12} {:<15} {:<15}"
            lines.append(f" {imagetype}S:")
            for target, target_group in image_group.groupby('target', observed=True):
                total_exposure_time_target = 0.0
                lines.append(f" Target: {target}\n")
                lines.append(format_string.format(
                    "Filter", "Frames", "Gain", "Egain", "Mean FWHM",
                    "Sensor Temp", "Mean Temp", "Exposure", "Total Exposure"
                ))
                for _, sub_group in target_group.groupby('filter', observed=True):
                    try:
                        frame_count = int(sub_group['number'].sum())
                        exposure = float(sub_group['duration'].iloc[0])
                        gain = f"{float(sub_group['gain'].iloc[0]) * 0.1:.2f} dB"
                        egain = f"{float(sub_group['egain'].iloc[0]):.2f} e/ADU"
                        mean_fwhm = f"{sub_group['meanFwhm'].mean():.2f} arcsec"
                        mean_temperature = f"{sub_group['temperature'].mean():.1f}\u00B0C"
                        sensor_temperature = f"{sub_group['sensorCooling'].mean():.1f}\u00B0C"

                        mean_temperature = f" {mean_temperature}" if mean_temperature[0] != '-' else mean_temperature
                        sensor_temperature = f" {sensor_temperature}" if sensor_temperature[0] != '-' else sensor_temperature

                        total_exposure = frame_count * exposure
                        total_exposure_time_target += total_exposure
                        lines.append(format_string.format(
                            str(sub_group['filter'].iloc[0]), frame_count, gain, egain,
                            mean_fwhm, sensor_temperature, mean_temperature,
                            f"{exposure:.2f} secs", seconds_to_hms(total_exposure, logger, aligned=True)
                        ))
                        logger.info(f"Processed {frame_count} frames for filter {sub_group['filter'].iloc[0]} "
                                   f"with total exposure time {seconds_to_hms(total_exposure, logger)}")
                    except (KeyError, IndexError, ValueError) as e:
                        logger.error(f"Error processing filter group for {imagetype} (target: {target}): {str(e)}")
                total_exposure_time += total_exposure_time_target
                lines.append(f"\n Exposure time for {target}: {seconds_to_hms(total_exposure_time_target, logger)}\n")

        elif imagetype in {'DARK', 'MASTERDARK', 'BIAS', 'MASTERBIAS'}:
            required_columns = {'gain', 'egain', 'duration', 'sensorCooling'} if imagetype in {'DARK', 'MASTERDARK'} else {'gain', 'egain', 'duration'}
            if not required_columns.issubset(image_group.columns):
                missing = required_columns - set(image_group.columns)
                logger.error(f"Missing required columns for {imagetype} processing: {missing}")
                return f"\nError processing {imagetype}: Missing columns {missing}\n", 0.0

            detail_format = (" {:<10} {:<8} {:<8} {:<12} {:<12} {:<15}"
                            if imagetype in {'DARK', 'MASTERDARK'}
                            else " {:<10} {:<8} {:<8} {:<12} {:<10} {:<15}")
            lines.append(f"\n {imagetype}S:\n")
            lines.append(detail_format.format(
                "Filter", "Frames", "Gain", "Egain", "Exposure",
                "Sensor Temp" if imagetype in {'DARK', 'MASTERDARK'} else "Total Exposure"
            ))
            for gain, sub_group in image_group.groupby('gain', observed=True):
                try:
                    frame_count = int(sub_group['number'].sum())
                    exposure = float(sub_group['duration'].iloc[0])
                    fgain = f"{float(gain) * 0.1:.2f} dB"
                    egain = f"{float(sub_group['egain'].iloc[0]):.2f} e/ADU"
                    filter_name = 'N/A' if 'filter' not in sub_group.columns else str(sub_group['filter'].iloc[0])
                    sensor_temperature = (f"{sub_group['sensorCooling'].mean():.1f}\u00B0C"
                                         if imagetype in {'DARK', 'MASTERDARK'} else "")
                    total_exposure = frame_count * exposure
                    total_exposure_time += total_exposure
                    lines.append(detail_format.format(
                        filter_name, frame_count, fgain, egain, f"{exposure:.2f} secs",
                        sensor_temperature if imagetype in {'DARK', 'MASTERDARK'} else seconds_to_hms(total_exposure, logger, aligned=True)
                    ))
                    logger.info(f"Processed {frame_count} frames for gain {fgain} "
                               f"with total exposure time {seconds_to_hms(total_exposure, logger)}")
                except (KeyError, IndexError, ValueError) as e:
                    logger.error(f"Error processing gain group for {imagetype} (gain: {gain}): {str(e)}")

        elif imagetype in {'FLAT', 'MASTERFLAT'}:
            required_columns = {'filter', 'number', 'duration', 'gain', 'egain'}
            if not required_columns.issubset(image_group.columns):
                missing = required_columns - set(image_group.columns)
                logger.error(f"Missing required columns for {imagetype} processing: {missing}")
                return f"\nError processing {imagetype}: Missing columns {missing}\n", 0.0

            detail_format = " {:<10} {:<8} {:<10} {:<15} {:<12} {:<15}"
            lines.append(f"\n {imagetype}S:\n")
            lines.append(detail_format.format("Filter", "Frames", "Gain", "Egain", "Exposure", "Total Exposure"))
            for _, sub_group in image_group.groupby('filter', observed=True):
                try:
                    frame_count = int(sub_group['number'].sum())
                    exposure = float(sub_group['duration'].iloc[0])
                    gain = f"{sub_group['gain'].mean() * 0.1:.2f} dB"
                    egain = f"{float(sub_group['egain'].iloc[0]):.2f} e/ADU"
                    total_exposure = frame_count * exposure
                    total_exposure_time += total_exposure
                    lines.append(detail_format.format(
                        str(sub_group['filter'].iloc[0]), frame_count, gain, egain,
                        f"{exposure:.2f} secs", seconds_to_hms(total_exposure, logger, aligned=True)
                    ))
                    logger.info(f"Processed {frame_count} frames for filter {sub_group['filter'].iloc[0]} "
                               f"with total exposure time {seconds_to_hms(total_exposure, logger)}")
                except (KeyError, IndexError, ValueError) as e:
                    logger.error(f"Error processing filter group for {imagetype} (filter: {sub_group['filter'].iloc[0]}): {str(e)}")

        logger.info(f"Completed processing {imagetype} with total exposure time: {seconds_to_hms(total_exposure_time, logger)}")
        return "\n".join(lines), total_exposure_time

    except Exception as e:
        logger.error(f"Failed to process image type {imagetype}: {str(e)}")
        return f"\nError processing {imagetype} frames\n", 0.0

def site_details(group: pd.DataFrame, site: str, logger: logging.Logger) -> str:
    """Generates a formatted summary of site details.

    Includes site name, latitude, longitude, Bortle scale, and SQM value.

    Args:
        group (pd.DataFrame): DataFrame containing site data.
        site (str): Name of the site.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        str: Formatted site details summary.

    Raises:
        ValueError: If group is not a DataFrame, site is invalid, or required columns are missing.
    """
    try:
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(group, pd.DataFrame):
            raise ValueError("group must be a pandas DataFrame")
        if not isinstance(site, str) or not site.strip():
            raise ValueError("site must be a non-empty string")

        logger.info(f"Generating site details for {site}")
        details = [f"\nSite: {site}"]

        required_columns = {'sitelat', 'sitelong', 'bortle', 'meanSqm'}
        if not required_columns.issubset(group.columns):
            missing = required_columns - set(group.columns)
            logger.error(f"Missing required columns for site {site}: {missing}")
            details.append("\tSite data unavailable")
            return "\n".join(details)

        try:
            sitelat = float(group['sitelat'].iloc[0])
            sitelong = float(group['sitelong'].iloc[0])
            bortle = float(group['bortle'].iloc[0])
            mean_sqm = float(group['meanSqm'].iloc[0])

            details.append(f"\tLatitude: {sitelat:.4f}\u00B0")
            details.append(f"\tLongitude: {sitelong:.4f}\u00B0")
            details.append(f"\tBortle scale: {bortle:.1f}")
            details.append(f"\tSQM: {mean_sqm:.2f} mag/arcsec²")
            logger.info(f"Site details for {site}: Latitude: {sitelat:.4f}\u00B0, "
                       f"Longitude: {sitelong:.4f}\u00B0, Bortle: {bortle:.1f}, SQM: {mean_sqm:.2f}")
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error accessing site data for {site}: {str(e)}")
            details.append("\tSite data unavailable")

        return "\n".join(details)

    except Exception as e:
        logger.error(f"Failed to generate site details for {site}: {str(e)}")
        return f"\nSite: {site}\n\tError retrieving site details"

def target_details(group: pd.DataFrame, logger: logging.Logger) -> str:
    """Determines and formats target details for an observation session.

    Normalizes target names and handles panel mosaics if present.

    Args:
        group (pd.DataFrame): DataFrame containing target data.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        str: Formatted target details.

    Raises:
        ValueError: If group is not a DataFrame or required columns are missing.
    """
    try:
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(group, pd.DataFrame):
            raise ValueError("group must be a pandas DataFrame")
        if 'target' not in group.columns:
            raise ValueError("group DataFrame missing required 'target' column")

        logger.info("Determining target details")
        target_format = " {:<6} {}"

        if group.empty:
            logger.warning("No target data available")
            return target_format.format("Target:", "No target data")

        # Normalize target names
        targets = pd.Series(group['target'].astype(str))
        normalized = targets.str.replace('_', ' ', regex=True).str.lower().str.split().str.join(' ')
        mapping = pd.Series(targets.values, index=normalized).to_dict()
        unique_values = pd.Series(list(mapping.keys())).unique()
        unique_targets = [mapping[val] for val in unique_values]

        logger.info(f"Unique targets: {unique_targets}")

        panels = [s for s in unique_targets if 'Panel' in s]
        if panels:
            target_name = panels[0].split('Panel')[0].strip()
            target = target_format.format("Target:", f"{target_name} {len(panels)} Panel Mosaic")
            logger.info(f"Target is a panel: {target}")
        else:
            target = target_format.format("Target:", unique_targets[0] if unique_targets else "Unknown")
            logger.info(f"Target: {target}")

        return target

    except Exception as e:
        logger.error(f"Failed to determine target details: {str(e)}")
        return "Target: Error retrieving target details"

def get_number_of_images(number_of_images: int, logger: logging.Logger) -> str:
    """Formats the total number of images processed.

    Args:
        number_of_images (int): Number of images processed.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        str: Formatted string with the number of images.

    Raises:
        ValueError: If number_of_images is not a non-negative integer or logger is invalid.
    """
    try:
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(number_of_images, int):
            raise ValueError(f"number_of_images must be an integer, got {type(number_of_images).__name__}")
        if number_of_images < 0:
            raise ValueError("number_of_images cannot be negative")

        logger.info(f"Total images processed: {number_of_images}")
        return f" {'Total number of images processed':<25}: {number_of_images}\n"

    except Exception as e:
        logger.error(f"Error formatting number of images: {str(e)}")
        return "Total number of images processed: Error\n"

def equipment_used(group: pd.DataFrame, df: pd.DataFrame, logger: logging.Logger) -> str:
    """Generates a summary of equipment used based on observation data.

    Includes telescope, camera, filter wheel, focuser, rotator, and capture software,
    with the first equipment item (Telescope) indented to align with subsequent items.

    Args:
        group (pd.DataFrame): DataFrame containing equipment data for LIGHT frames.
        df (pd.DataFrame): DataFrame containing all observation data.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        str: Formatted equipment summary with aligned indentation.

    Raises:
        ValueError: If inputs are not valid DataFrames or required columns are missing.
    """
    try:
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(group, pd.DataFrame) or not isinstance(df, pd.DataFrame):
            raise ValueError("group and df must be pandas DataFrames")

        required_columns = {'telescope', 'camera', 'filterWheel', 'focuser', 'rotator', 'swcreate'}
        if not required_columns.issubset(group.columns):
            missing = required_columns - set(group.columns)
            logger.error(f"Missing required columns in group DataFrame: {missing}")
            return "\nEquipment used:\n\tMissing required columns\n"

        logger.info("Generating equipment summary")
        equipment_format = "\t{:<20}: {}\n"
        equipment = ["\nEquipment used:\n"]

        equipment_items = {
            'Telescope': group['telescope'].iloc[0],
            'Camera': group['camera'].iloc[0],
            'Filterwheel': group['filterWheel'].iloc[0],
            'Focuser': group['focuser'].iloc[0],
            'Rotator': group['rotator'].iloc[0]
        }

        for item, value in equipment_items.items():
            if pd.notna(value) and value != 'None':
                equipment.append(equipment_format.format(item, value))
        logger.info(f"Equipment details: {', '.join(f'{k}: {v}' for k, v in equipment_items.items() if pd.notna(v) and v != 'None')}")

        software_set = set(group['swcreate'].dropna().unique())
        master_types = {'MASTERFLAT', 'MASTERDARKFLAT', 'MASTERBIAS', 'MASTERDARK'}
        for master_type in master_types:
            master_group = df[df['imageType'] == master_type]
            if not master_group.empty and 'swcreate' in master_group.columns:
                software_set.update(master_group['swcreate'].dropna().unique())

        if software_set:
            software_list = list(software_set)
            equipment.append(equipment_format.format("Capture software", software_list.pop(0)))
            for software in software_list:
                equipment.append(equipment_format.format("", software))
            logger.info(f"Capture software: {software_set}")
        else:
            logger.warning("No software information found")
            equipment.append(equipment_format.format("Capture software", "Unknown"))

        return "".join(equipment)

    except Exception as e:
        logger.error(f"Failed to generate equipment summary: {str(e)}")
        return "\nEquipment used:\n\tError retrieving equipment details\n"

def observation_period(group: pd.DataFrame, logger: logging.Logger) -> str:
    """Generates a summary of the observation period.

    Includes start date, end date, number of days, sessions, and temperature statistics,
    with the first item (Start date) indented to align with subsequent items.

    Args:
        group (pd.DataFrame): DataFrame containing observation data.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        str: Formatted observation period summary with aligned indentation.

    Raises:
        ValueError: If group is not a DataFrame or required columns are missing.
    """
    try:
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(group, pd.DataFrame):
            raise ValueError("group must be a pandas DataFrame")

        required_columns = {'imageType', 'start_date', 'end_date', 'num_days', 'sessions', 'temp_min', 'temp_max', 'temperature'}
        if not required_columns.issubset(group.columns):
            missing = required_columns - set(group.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Determining observation period")
        period = ["\nObservation period:\n"]
        period_format = "\t{:<25}: {}\n"
        error_date = pd.to_datetime('1900-01-01')

        light_group = group[group['imageType'] == 'LIGHT'].copy()
        if light_group.empty:
            logger.warning("No LIGHT frames available")
            return "\n".join([*period, "\tNo LIGHT frames available\n", "\tSession date information not available\n"])

        try:
            start_date = pd.to_datetime(light_group['start_date'].iloc[0])
            end_date = pd.to_datetime(light_group['end_date'].iloc[0])
            num_days = int(light_group['num_days'].iloc[0])
            sessions = int(light_group['sessions'].iloc[0])

            if start_date == error_date or end_date == error_date or num_days == 0 or sessions == 0:
                logger.error("Invalid date information for observation period")
                return "\n".join([*period, "\tSession date information not available\n"])

            period.extend([
                period_format.format("Start date", start_date.strftime('%Y-%m-%d')),
                period_format.format("End date", end_date.strftime('%Y-%m-%d')),
                period_format.format("Days", num_days),
                period_format.format("Observation sessions", sessions),
                period_format.format("Min temperature", f"{light_group['temp_min'].min():.1f}\u00B0C"),
                period_format.format("Max temperature", f"{light_group['temp_max'].max():.1f}\u00B0C"),
                period_format.format("Mean temperature", f"{light_group['temperature'].mean():.1f}\u00B0C"),
                "\n"  # Extra newline after Mean temperature
            ])
            logger.info(f"Observation period: Start {start_date.strftime('%Y-%m-%d')}, "
                       f"End {end_date.strftime('%Y-%m-%d')}, Days {num_days}, Sessions {sessions}")
            logger.info(f"Temperatures: Min {light_group['temp_min'].min():.1f}\u00B0C, "
                       f"Max {light_group['temp_max'].max():.1f}\u00B0C, "
                       f"Mean {light_group['temperature'].mean():.1f}\u00B0C")

        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error processing observation period data: {str(e)}")
            period.append("\tError retrieving observation period data\n")
            return "\n".join(period)

        return "".join(period)

    except Exception as e:
        logger.error(f"Failed to determine observation period: {str(e)}")
        return "\nObservation period:\n\tError retrieving observation period data\n"

def summarize_session(df: pd.DataFrame, logger: logging.Logger, number_of_images: int) -> str:
    """Generates a comprehensive summary of an observation session.

    Includes target, site, equipment, observation period, and image type details with
    corrected formatting: no indentation for target, aligned indentation for telescope
    and start date, new line after headers, empty line after mean temperature, and
    proper spacing between exposure lines.

    Args:
        df (pd.DataFrame): DataFrame containing observation data.
        logger (logging.Logger): Logger for logging messages.
        number_of_images (int): Total number of images processed.

    Returns:
        str: Formatted session summary.

    Raises:
        ValueError: If df is not a DataFrame, logger is invalid, or number_of_images is invalid.
    """
    try:
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger instance")
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if not isinstance(number_of_images, int) or number_of_images < 0:
            raise ValueError("number_of_images must be a non-negative integer")

        logger.info("Generating observation session summary")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        imagetype_order = ['LIGHT', 'FLAT', 'MASTERFLAT', 'DARKFLAT', 'BIAS', 'DARK', 'MASTERDARKFLAT', 'MASTERBIAS', 'MASTERDARK']
        summary_parts = [f"Observation session summary\nGenerated {current_time}\n"]

        if df.empty:
            logger.warning("Input DataFrame is empty")
            summary_parts.append("No observation data available\n")
            return "\n".join(summary_parts)

        required_columns = {'site', 'imageType'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            logger.error(f"Missing required columns in DataFrame: {missing}")
            summary_parts.append(f"Error: Missing required columns {missing}\n")
            return "\n".join(summary_parts)

        for site, group in df.groupby('site', observed=True):
            target_group = group[group['imageType'] == 'LIGHT'].reset_index(drop=True)
            if not target_group.empty:
                logger.info(f"Processing site: {site}")
                try:
                    # Remove leading newline and indentation from target details
                    target_summary = target_details(target_group, logger).lstrip('\n').strip()
                    summary_parts.append(f"{target_summary}\n")
                    logger.info("Added target details to summary")

                    # Add site details
                    summary_parts.append(site_details(target_group, site, logger))
                    logger.info("Added site details to summary")

                    # Add equipment used
                    summary_parts.append(equipment_used(target_group, df, logger))
                    logger.info("Added equipment details to summary")

                    # Add observation period
                    summary_parts.append(observation_period(target_group, logger))
                    logger.info("Added observation period to summary")
                except Exception as e:
                    logger.error(f"Error processing site {site}: {str(e)}")
                    summary_parts.append(f"\nError processing site {site}\n")

            for imagetype in imagetype_order:
                if imagetype in group['imageType'].unique():
                    try:
                        summary, total_exposure_time = process_image_type(group, imagetype, logger)
                        summary_parts.append(summary)
                        summary_parts.append(f"\nTotal {imagetype} Exposure Time: {seconds_to_hms(total_exposure_time, logger)}\n\n")
                        logger.info(f"Added {imagetype} details to summary, total exposure: {seconds_to_hms(total_exposure_time, logger)}")
                    except Exception as e:
                        logger.error(f"Error processing image type {imagetype} for site {site}: {str(e)}")
                        summary_parts.append(f"\nError processing {imagetype} frames\n")

            summary_parts.append(get_number_of_images(number_of_images, logger))
            logger.info("Added number of images to summary")

        logger.info("Observation session summary generated successfully")
        return "".join(summary_parts)

    except Exception as e:
        logger.error(f"Failed to generate session summary: {str(e)}")
        return "\nObservation session summary\n\tError generating summary\n"
    
