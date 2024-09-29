import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse
from typing import Tuple,Dict, Optional, Any, List, Union
from configobj import ConfigObj, Section
import os
from io import StringIO
import xml.etree.ElementTree as ET
import re
from astropy.io import fits
import struct
import requests
import math
import sys
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut,GeocoderUnavailable
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#utils_version = '1.1.4'
# Changes:
# Tuesday 9th January 2024
# Ensure all self.headers['IMAGETYP'] are upper case
# Corrected position print of 'imagetype' for DARK frames
# Wednesday 10th January 2024
# Ensure that if all geocoding fails, location_string is set to config['defaults']['SITE']
# Log all geocoding errors returned by geopy
# v1.1.3 11th January 2024
# Deal with WBPP
# Removed duplicate light frames from header, thes are typically found in WBPP directories
# If master calibration files are found drop the corresponding calibration frames an donly use the masters
# 12th January 2024
# Added grouping on object (target after name change) in aggregate_parameters.  Function was not counting light frames accurately
# added check for master dark, bias and flats. If found any dark, bias or flats frames are ignored
# v1.1.4 13th January 2024
# Redesign of how code parses data files
# Allow the use of symbolic links to calibration data directories when reading data into the program using the function process_directory
# Removed use of WBPP keyword
# 22nd January 2024
# Bug fix: check for master dark, bias and flatdarks. If found any dark, bias or flatdarks frames are ignored
# 24th January 2024
# Corrected FWHM calculation to fwhm = hfr * imscale*2 from fwhm = hfr * imscale
# Corrected observation sessions to be the number of unique dates in the dates set/2
# Corrected the mean session temparature to be the mean of the just ImageType = lights temperature column
# Corrected random selection of image directory when removing duplicates when processing WBPP directories. Force use of _c.xisf files as they contain the correct header information
# Tested 26 directory structures all passed
##############################################################################################################################################################################
#Saved as version 1.2.0 this is now a candidate release version
# 28th January 2024
# Corrected use of the wrong column name for sensorCooling. Used sensorTemperature when sensorCooling was required by Astrobin
# Corrected case when .xisf that have been processed by Pixinsight can have no header information.This seems to be occour when data is processed after calibration.
# and the WBPP directory is used which contains other images that are post processed. Some do not contain header information
# Corrected error where code was adding the same header to the aggregating data frame twice ( big miss!!)
# 29th January 2024
# Changed logic in site.process_new_location
#       1.0 Ensure that if all geocoding fails, location_string is set to config['defaults']['SITE'] in all cases
#       2.0 Dont geocode again if location_string is already set to config['defaults']['SITE']
#       3.0 If bortle and sqm returned are 0,0 then set them to config['defaults']['BORTLE'] and config['defaults']['SQM']
#       4.0 Log all geocoding errors returned by geopy
# Capture error when self.location_info = self.find_location_by_coords(est, lat_long_pair, 2) returns None
# This should not occur so code needs further investigation added logging to enable this
# 30th January 2024
# Version 1.2.3
# Corrected logic that causes coded to claim sites that had been seen were not seen.
# Improved logging to make it easier to read
# Added meassurement and reporting of session temperature statistics
# Added mesurement and reporting of total number of images processed
# Improved layout of summary report
# 31st January 2024
# Corrected observation session count
# 6th February 2024
# Version 1.2.4
# Corrected observation session count it was not corrected before
# had to leave date obs as date times to get the correct date,
# calculate session statistics in aggergate_parameters then convert to date after session calculations
# Minor formatting changes on summary report to align +ve and -ve temperature values
# 7th February 2024
# Version 1.2.5
# added check for session date calculations, any errors will be logged and code reporst session date information not available
# 7th February 2024
# Version 1.2.8
# dump data frames to .csv file for analysis
# 8th February 2024
# Corrected error in the function observation period where ['name'][0] was used instead of ['name'].iloc[0]
# 9th February 2024
# Version 1.2.10
# Ensured that only master flats and master darks that have the same filters and exposures as the light frames are used
# 12th February 2024
# Release version 1.3.0
# 27th February 2024
# Version 1.3.1
# Modified debugging file dumps, so they occour after the data has been processed not all at the the end
# 29th February 2024
# Version 1.3.2
# Code added to handle FOCUSER and SITENAME SGP-PRO keywords
# Code added to deal with situation where keyword pairs conflict with each other
# 'EXPTIME' and 'EXPOSURE' in the same data frame
# 'LAT-OBS' and 'SITELAT' in the same data frame
# Handles mutiple MasterFlat frames for same filter
# Correctly total MasterFlat frames in summary and astrobin output
# Ensure all calibration frame locations are set equal to the nearest light frame location
# Stops calibration frames generating their own site location
# Version 1.3.3
# 4th March 2024
# Corrected error in aggregate_parameters where if no MASTER frames are present the code would fail
# Version 1.3.4
# 5th March 2024
# Corrected utf-8 encoding errror with logging
# Reset index on group in summerize session such that group['target'].iloc[0] returns the correct value
# formatted time output seconds are now shown to 2dp
# Version 1.3.6
# 16 June 2024
# No changes, but matches the version number of the main package
# Version 1.3.7
# allows the scipt to be called from an image directory and process the images in that directory but allows for calibration directories to be passed as arguments
# eg AstroBinUploadv1_3_7.py "." /home/user/images/calibration
# All output files are written to the current image directory under a directory called "AstroBinUpload"
# version 1.3.8
# 28th September 2024
# Allows the processing of LIGHTFRAMES as well as LIGHT frames
# Modification in the process headers function to allow the processing of LIGHTFRAMES as well as LIGHT frames
# version 1.3.9
# 28th September 2024
# Allows the processing of LIGHTFRAMES and Light Frames as well as LIGHT frames
# Modification in the process headers function to allow the processing of LIGHTFRAMES Light Frames as well as LIGHT frames
# version 1.3.10
# 29th September 2024
# Deals with the case where a fractional part of a second is present in some date-obs keyword but not in others
# Deals with the case where the filter names in teh light frames have trailing white spaces.



utils_version = '1.3.10'

def initialise_logging(log_filename: str) -> logging.Logger:
        """
        Initializes logging for the application.

        Initializes logging for the application, by creating a logger object and adding a file
        handler to it. The file handler will write log messages to the specified log file.

        Parameters:
        - log_filename (str): The name of the log file.

        Returns:
        - logging.Logger: The logger object.
        """

        # Clear the log file
        with open(log_filename, 'w'):
            pass

        # Set up logging in the main part of your application
        logger = logging.getLogger(__name__)


        # Remove all handlers associated with the logger object.
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logger.setLevel(logging.INFO)

        #handler = logging.StreamHandler()
        handler = logging.FileHandler(log_filename, encoding='utf-8')  # Log to a file
        formatter = logging.Formatter('%(asctime)s - %(name)s - Line: %(lineno)d - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        return logger

def format_seconds_to_hms(seconds: int) -> str:
    """
    Converts seconds to a string formatted as hours, minutes, and seconds.

    This function takes an integer number of seconds and converts it to a string formatted as hours, minutes, and seconds.
    If the number of hours or minutes is zero, it is not included in the output string.

    Parameters:
    - seconds (int): The number of seconds to convert.

    Returns:
    - str: The formatted string representing the input time in hours, minutes, and seconds.
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = float(seconds % 60)

    return "{:>6} {:>6} {:>6}".format(
        f"{hours} hrs" if hours > 0 else "",
        f"{minutes} mins" if hours > 0 or minutes > 0 else "",
        f"{secs:.2f} secs"
    )

def seconds_to_hms(seconds: int) ->str:
    """
    Converts seconds to a string formatted as hours, minutes, and seconds.

    This function takes an integer number of seconds and converts it to a string formatted as hours, minutes, and seconds.
    If the number of hours or minutes is zero, it is not included in the output string.

    Parameters:
    - seconds (int): The number of seconds to convert.

    Returns:
    - str: The string representing the input time in hours, minutes, and seconds.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = float(seconds % 60)
    return f"{hours} hrs {minutes} mins {secs:.2f} secs"

def process_image_type(group: pd.DataFrame, imagetype: str, logger: logging.Logger) -> Tuple[str, float]:
    """
    Processes a specific image type and calculates total exposure time.

    This function processes a specific image type ('LIGHT', 'DARK', 'MASTERDARK', 'BIAS', 'MASTERBIAS', 'FLAT', 'MASTERFLAT')
    and calculates the total exposure time. It generates a detailed summary for each image type.

    Parameters:
    - group (pd.DataFrame): The DataFrame containing the grouped data.
    - imagetype (str): The type of the image to process.
    - logger (logging.Logger): The logger to use for logging messages.

    Returns:
    - Tuple[str, float]: A tuple containing the detailed summary as a string and the total exposure time as a float.
    """
    logger.info(f"Starting to process image type: {imagetype}")
    lines = []

    image_group = group[group['imageType'] == imagetype]
    #total_exposure_time_all_targets = 0  # Add this line
    total_exposure_time = 0  # Add this lin

    if imagetype in ['LIGHT', 'LIGHTFRAME']:
        format_string = " {:<8} {:<8}  {:<8} {:<12} {:<12} {:<12} {:<12} {:<15} {:<15}"
        lines.append(f" {imagetype}S:")

        for target, target_group in image_group.groupby('target'):
            total_exposure_time_target = 0
            lines.append(f" Target: {target}\n")
            lines.append(format_string.format("Filter","Frames", "Gain", "Egain", "Mean FWHM", "Sensor Temp","Mean Temp", "Exposure", "Total Exposure"))
            for _, sub_group in target_group.groupby('filter'):
                frame_count = sub_group['number'].sum()
                exposure = sub_group['duration'].iloc[0].astype(float).round(2)
                gain = f"{(sub_group['gain'].iloc[0]*.1).astype(float).round(2)} dB"
                egain =f"{sub_group['egain'].iloc[0].astype(float).round(2)} e/ADU"

                formatted_exposure = f"{exposure:.2f} secs"  # Format for two decimal places
                total_exposure = frame_count * exposure
                total_exposure_time_target += total_exposure
                mean_fwhm = sub_group['meanFwhm'].mean().round(2)
                mean_temperature = f"{sub_group['temperature'].mean().round(1)}\u00B0C"
                sensor_temperature = f"{sub_group['sensorCooling'].mean().round(1)}\u00B0C"

                # Add a space before the number for positive temperatures
                mean_temperature = f" {mean_temperature}" if mean_temperature[0] != '-' else mean_temperature
                sensor_temperature = f" {sensor_temperature}" if sensor_temperature[0] != '-' else sensor_temperature

                formatted_mean_fwhm = f"{mean_fwhm:.2f} arcsec"  # Format for two decimal places
                lines.append(format_string.format(sub_group['filter'].iloc[0],frame_count, gain,egain,formatted_mean_fwhm, sensor_temperature, mean_temperature, formatted_exposure , format_seconds_to_hms(total_exposure)))
                logger.info(f"Processed {frame_count} frames for filter {sub_group['filter'].iloc[0]} with total exposure time {seconds_to_hms(total_exposure)}")
            lines.append(f"\n Exposure time for {target}: {seconds_to_hms(total_exposure_time_target)}\n")
            total_exposure_time+= total_exposure_time_target  # Add this line


    elif imagetype in ['DARK', 'MASTERDARK', 'BIAS', 'MASTERBIAS']:

        if imagetype == 'DARK' or imagetype == 'MASTERDARK' :
            detail_format = " {:<10} {:<8} {:<8} {:<12} {:<12} {:<12} {:<15}"
            lines.append(f"\n {imagetype}S:\n")
            lines.append(detail_format.format("Filter", "Frames", "Gain", "Egain","Exposure","Sensor Temp", "Total Exposure"))
            for gain, sub_group in image_group.groupby('gain'):
                frame_count = sub_group['number'].sum()
                exposure = sub_group['duration'].iloc[0].astype(float).round(2)
                formatted_exposure = f"{exposure:.2f} secs"  # Format for two decimal places
                total_exposure = frame_count * exposure
                total_exposure_time += total_exposure
                fgain = f"{gain*.1} dB"
                egain =f"{sub_group['egain'].iloc[0].astype(float).round(2)} e/ADU"
                sensor_temperature = f"{sub_group['sensorCooling'].mean().round(1)}\u00B0C"
                filter_name = 'N/A' if 'filter' not in sub_group else sub_group['filter'].iloc[0]
                lines.append(detail_format.format(filter_name, frame_count, fgain, egain,formatted_exposure,sensor_temperature, format_seconds_to_hms(total_exposure)))
                logger.info(f"Processed {frame_count} frames for gain {fgain} with total exposure time {seconds_to_hms(total_exposure)}")
        else:
            lines.append(f"\n {imagetype}ES:\n")
            detail_format = " {:<10} {:<8} {:8} {:<12} {:<10} {:<15}"
            lines.append(detail_format.format("Filter", "Frames","Gain", "Egain", "Exposure", "Total Exposure"))
            for gain, sub_group in image_group.groupby('gain'):
                frame_count = sub_group['number'].sum()
                exposure = sub_group['duration'].iloc[0].astype(float).round(2)
                formatted_exposure = f"{exposure:.2f} secs"  # Format for two decimal places
                total_exposure = frame_count * exposure
                total_exposure_time += total_exposure
                fgain = f"{gain*.1} dB"
                egain =f"{sub_group['egain'].iloc[0].astype(float).round(2)} e/ADU"
                filter_name = 'N/A' if 'filter' not in sub_group else sub_group['filter'].iloc[0]
                lines.append(detail_format.format(filter_name, frame_count, fgain, egain, formatted_exposure, format_seconds_to_hms(total_exposure)))
                logger.info(f"Processed {frame_count} frames for gain {fgain} with total exposure time {seconds_to_hms(total_exposure)}")

    elif imagetype in ['FLAT', 'MASTERFLAT']:
        detail_format = " {:<10} {:<8} {:<10} {:<15} {:<12} {:<15}"
        lines.append(f"\n {imagetype}S:\n")
        #lines.append("Filter     Frames     Exposure        Total Exposure ")
        lines.append(detail_format.format("Filter", "Frames","Gain", "Egain", "Exposure", "Total Exposure"))
        for _, sub_group in image_group.groupby('filter'):
            frame_count = sub_group['number'].sum()
            exposure = sub_group['duration'].iloc[0].astype(float).round(2)
            gain = f"{(sub_group['gain'].mean()*.1).round(2)} dB"
            egain =f"{sub_group['egain'].iloc[0].astype(float).round(2)} e/ADU"
            formatted_exposure = f"{exposure:.2f} secs"  # Format for two decimal places
            total_exposure = frame_count * exposure
            total_exposure_time += total_exposure
            lines.append(detail_format.format(sub_group['filter'].iloc[0], frame_count, gain,egain,formatted_exposure, format_seconds_to_hms(total_exposure)))
            logger.info(f"Processed {frame_count} frames for filter {sub_group['filter'].iloc[0]} with total exposure time {seconds_to_hms(total_exposure)}")



    return "\n".join(lines), total_exposure_time

def site_details(group: pd.DataFrame, site: str, logger: logging.Logger) -> str:
    """
    Generates a summary of site details.

    This function generates a summary of the site details based on the DataFrame provided.
    It includes the site name, latitude, longitude, mean temperature, Bortle scale, and SQM.

    Parameters:
    - group (pd.DataFrame): The DataFrame containing the grouped data.
    - site (str): The name of the site.
    - logger (logging.Logger): The logger to use for logging messages.

    Returns:
    - str: The site details.
    """
    logger.info("Generating site details summary")

    details = f"\n Site:\t{site}\n"
    logger.info(f" Site: {site}")
    details += f"\tLatitude: {group['sitelat'].iloc[0]}\u00B0, Longitude: {group['sitelong'].iloc[0]}\u00B0\n"
    logger.info(f"Latitude: {group['sitelat'].iloc[0]}\u00B0, Longitude: {group['sitelong'].iloc[0]}\u00B0")
    details += f"\tBortle scale: {group['bortle'].iloc[0]}, SQM: {group['meanSqm'].iloc[0]} mag/arcsec\u00B2\n"
    logger.info(f"Bortle scale: {group['bortle'].iloc[0]}, SQM: {group['meanSqm'].iloc[0]} mag/arcsec\u00B2")

    return details

def equipment_used(group: pd.DataFrame, df: pd.DataFrame, logger: logging.Logger) -> str:
    """
    Determines the equipment used.

    This function generates a summary of the equipment used based on the 'LIGHT' frames in the DataFrame.
    It also captures the software used from both 'LIGHT' and 'MASTER' frames.

    Parameters:
    - group (pd.DataFrame): The DataFrame containing the grouped data.
    - df (pd.DataFrame): The DataFrame containing the observation data.
    - logger (logging.Logger): The logger to use for logging messages.

    Returns:
    - str: The equipment used.
    """
    logger.info("Determining equipment used")

    equipment_format = " {:<20}: {}\n"
    equipment = "\n Equipment used:\n"
    eq_group = group

    # Add equipment details
    if eq_group['telescope'].iloc[0] != 'None':
        equipment += equipment_format.format("Telescope", eq_group['telescope'].iloc[0])
    if eq_group['camera'].iloc[0] != 'None':
        equipment += equipment_format.format("Camera", eq_group['camera'].iloc[0])
    if eq_group['filterWheel'].iloc[0] != 'None':
        equipment += equipment_format.format("Filterwheel", eq_group['filterWheel'].iloc[0])
    if eq_group['focuser'].iloc[0] != 'None':
        equipment += equipment_format.format("Focuser" , eq_group['focuser'].iloc[0])
    if eq_group['rotator'].iloc[0] != 'None':
        equipment += equipment_format.format("Rotator" , eq_group['rotator'].iloc[0])

    logger.info(
    f"Equipment details: Telescope: {eq_group['telescope'].iloc[0]}, "
    f"Camera: {eq_group['camera'].iloc[0]}, Filterwheel: {eq_group['filterWheel'].iloc[0]}, "
    f"Focuser: {eq_group['focuser'].iloc[0]}, Rotator: {eq_group['rotator'].iloc[0]}"
    )
    # Capture software from LIGHT and MASTER frames
    software_set = set(eq_group['swcreate'].unique())
    master_types = ['MASTERFLAT', 'MASTERDARKFLAT', 'MASTERBIAS', 'MASTERDARK']
    for master_type in master_types:
        master_group = df[df['imageType'] == master_type]
        if not master_group.empty:
            software_set.update(master_group['swcreate'].unique())

    # Add software details
    software_list = list(software_set)
    equipment += equipment_format.format("Capture software", software_list.pop(0))
    for software in software_list:
        equipment += equipment_format.format("", software)

    logger.info(f"Capture software: {software_set}")

    return equipment

def target_details(group: pd.DataFrame, logger: logging.Logger) -> str:
    """
    Determines the target details.

    This function calculates the target details based on the 'LIGHT' frames in the DataFrame.
    It normalizes the target names, finds unique targets, and checks if any target is a panel.
    If a panel is found, it formats the target name accordingly.

    Parameters:
    - group (pd.DataFrame): The DataFrame containing the grouped data.
    - logger (logging.Logger): The logger to use for logging messages.

    Returns:
    - str: The target details.
    """
    logger.info("Determining target details")

    target_format = " {:<6} {}"
    #target_group = group[group['imageType'] == 'LIGHT'].reset_index(drop=True)

    #logger.info(f"Target group: {group}")


    # Get the 'target' column values
    arr = pd.Series(group['target'].astype(str))
    logger.info(f"Target value: {arr[0]}")

    # Normalize the strings: lowercase, remove underscores, and collapse multiple spaces to a single space
    normalized = arr.str.replace('_', ' ').str.lower().str.split().str.join(' ')
    #logger.info(f"Normalized targets: {normalized}")
    # Create a mapping from normalized strings to original strings
    mapping = pd.Series(arr.values, index=normalized).to_dict()
    logger.info(f"Mapping: {mapping}")
    # Get unique values
    unique_values = pd.Series(list(mapping.keys())).unique()
    logger.info(f"Unique values: {unique_values}")

    # Map unique values back to original strings
    targets = [mapping[val] for val in unique_values]

    logger.info(f"Unique targets: {targets}")

    # Check if 'Panel' is in each string
    panels = [s for s in targets if 'Panel' in s]

    if panels:
        # Split the first string at 'Panel' and take the first part
        target_name = panels[0].split('Panel')[0].strip()

        # Create the final string
        target = target_format.format("Target:", f"{target_name} {len(panels)} Panel Mosaic")
        logger.info(f"Target is a panel: {target}")
    else:
        target = target_format.format("Target:", targets[0])
        logger.info(f"Target: {target}")


    return target

def observation_period(group_in: pd.DataFrame, logger: logging.Logger) -> str:


    group = group_in.copy(deep=True)
    logger.info("Determining observation period")
    period_format = " {:<25}: {}\n"
    period = "\n Observation period: \n"
    error_date = pd.to_datetime('1900-01-01')

    if 'imageType' not in group.columns:
        logger.error(" imageType is not a column in the group DataFrame.")
        logger.info("Output:  Session date information not available")
        period += " Session date information not available\n"
        return period

    if (group['imageType'] != 'LIGHT').all():
        logger.info("No LIGHT frames available")
        period += " No LIGHT frames available\n"
        period += " Session date information not available\n"
        return period

    light_group = group[group['imageType'] == 'LIGHT']

    if not light_group.empty:

        start_date = light_group['start_date'].iloc[0]
        end_date = light_group['end_date'].iloc[0]
        num_days = light_group['num_days'].iloc[0]
        sessions= light_group['sessions'].iloc[0]


        if start_date == error_date or end_date == error_date or num_days == 0 or sessions == 0:
            period += " Session date information not available\n"
            logger.error("No valid date information available for observation period.")
            logger.info("Output:  Session date information not available")
        else:
            period += period_format.format("Start date", start_date)
            period += period_format.format("End date", end_date)
            period += period_format.format(f"Days",num_days)
            period += period_format.format("Observation sessions", sessions)

            logger.info(f"Start date: {start_date}")
            logger.info(f"End date: {end_date}")
            logger.info(f"Session length: {num_days}")
            logger.info(f"Number of sessions: {sessions}")


        period += period_format.format("Min temperature", f"{light_group['temp_min'].min().round(1)}\u00B0C")
        period += period_format.format("Max temperature", f"{light_group['temp_max'].max().round(1)}\u00B0C")
        period += period_format.format("Mean temperature", f"{light_group['temperature'].mean().round(1)}\u00B0C")



        logger.info(f"Min temperature: {light_group['temp_min'].min().round(1)}\u00B0C")
        logger.info(f"Max temperature: {light_group['temp_max'].max().round(1)}\u00B0C")
        logger.info(f"Mean temperature: {light_group['temperature'].mean().round(1)}\u00B0C")

        return period
    else:
        logger.info("No LIGHT frames available")

        return "\nObservation period: \n\tNo LIGHT frames available\n"

def get_number_of_images(number_of_images: int, logger: logging.Logger) -> str:
    logger.info("Determining number of images processed")
    images = number_of_images
    logger.info(f"Total number of images processed: {images}")
    number_of_images_format = " {:<25}: {}\n"

    return number_of_images_format.format("Total number of images processed",images)

def summarize_session(df: pd.DataFrame, logger: logging.Logger, number_of_images: int) -> str:
    """
    Summarizes an observation session.

    This function generates a summary of an observation session, including details about the target,
    site, equipment used, observation period, and exposure time for each image type.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the observation data.
    - logger (logging.Logger): The logger to use for logging messages.

    Returns:
    - str: The summary of the observation session.
    """
    logger.info("")
    logger.info("GENERATING OBSERVATION SESSION SUMMARY")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    imagetype_order = ['LIGHT', 'FLAT', 'MASTERFLAT', 'DARKFLAT', 'BIAS', 'DARK', 'MASTERDARKFLAT', 'MASTERBIAS', 'MASTERDARK']

    summary_parts = [f"\n Observation session summary\n Generated {current_time}\n"]

    logger.info("Added initial summary part")
    logger.info("")

    for site, group in df.groupby('site'):
        target_group = group[group['imageType'] == 'LIGHT'].reset_index(drop=True)
        if not target_group.empty:
            logger.info(f"Processing site: {site}")
            logger.info("")
            summary_parts.append(target_details(target_group ,logger))
            logger.info("Added target details to summary")
            logger.info("")
            target = site_details(target_group , site,logger)
            if target != 'No target':
                summary_parts.append(target)
                logger.info("Added site details to summary")
                logger.info("")

            summary_parts.append(equipment_used(target_group ,df,logger))
            logger.info("Added equipment used to summary")
            logger.info("")

            summary_parts.append(observation_period(target_group ,logger))
            logger.info("Added observation period to summary")
            logger.info("")



    #for site, group in df.groupby('site'):

        for imagetype in imagetype_order:

            if imagetype in group['imageType'].unique():
                logger.info(f"Processing image type: {imagetype}")
                summary, total_exposure_time = process_image_type(group, imagetype,logger)
                summary_parts.append(summary)
                if imagetype == 'LIGHT':
                    summary_parts.append(f" Total {imagetype} Exposure Time: {format_seconds_to_hms(total_exposure_time)}\n")
                else:
                    summary_parts.append(f"\n Total {imagetype} Exposure Time: {seconds_to_hms(total_exposure_time)}\n")
                logger.info(f"Finished processing image type: {imagetype}. Total exposure time: {seconds_to_hms(total_exposure_time)}")
                logger.info(f"Added image type {imagetype} to summary")
                logger.info("")
                logger.info("*"*200)
                logger.info("")


        summary_parts.append(get_number_of_images(number_of_images,logger))
        logger.info("Added number of images to summary")
        logger.info("")


    logger.info("Observation session summary generated")

    return "\n".join(summary_parts)

class Configuration:

    @staticmethod
    def cast_value(value: Any) -> Any:
        """
        Casts the input value to an appropriate type.

        This function attempts to cast the input value to a list, float, or int. If the input value is already a list,
        it recursively applies the casting to each item in the list. If the input value is a string that contains a comma,
        it splits the string at the comma and treats it as a list, recursively applying the casting to each item. If the
        input value is a string that contains a period, it attempts to cast it to a float. If the input value is a string
        that does not contain a period, it attempts to cast it to an int. If all casting attempts fail, it returns the
        input value as is.

        Parameters:
        value (Any): The input value to be cast. This can be of any type.

        Returns:
        Any: The cast value. This can be a list, float, int, or string, depending on the input.
        """

        if isinstance(value, list):  # Check if value is already a list
            return [Configuration.cast_value(item) for item in value]
        if ',' in str(value):  # Check if value is a list
            return [Configuration.cast_value(item) for item in str(value).split(',')]
        try:
            if '.' in str(value):  # Check if value can be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value  # return as is, if it's a string

    def __init__(self, config_filename,logger=None):

        self.filename = config_filename
        # ... ini_string definition ...
        self.ini_string= """
        [defaults]
        #FITS keyword    Default value
        IMAGETYP        = LIGHT
        EXPOSURE        = 0.0
        DATE-OBS        = 2023-01-01
        XBINNING        = 1
        GAIN            = -1
        EGAIN           = -1
        INSTRUME        = None
        TELESCOP        = None
        FOCNAME         = None
        FWHEEL          = None
        ROTATOR         = None
        XPIXSZ          = 3.76
        CCD-TEMP        = -10
        FOCALLEN        = 540
        FOCRATIO        = 5.4
        SITE            = Papworth Everard
        SITELAT         = 52.2484
        SITELONG        = -0.1231
        BORTLE          = 4
        SQM             = 20.5
        FILTER          = No Filter
        OBJECT          = No target
        FOCTEMP         = 20
        HFR             = 1.6
        FWHM            = 0
        SWCREATE        = Unknown package

        [filters]
        #Filter     code
        Ha          = 4663
        SII         = 4844
        OIII        = 4752
        Red         = 4649
        Green       = 4643
        Blue        = 4637
        Lum         = 2906
        [secret]
        #API key        API endpoint
        xxxxxxxxxxxxxxxx     = https://www.lightpollutionmap.info/QueryRaster/
        EMAIL_ADDRESS        = id@provider.com
        [sites]
        """
        self.logger = logger
        self.change = self.initialise_config()

    def cast_section(self, section: Dict[Any, Any]) -> None:
        """
        Recursively casts the values in a section dictionary to appropriate types.

        This function iterates over each key-value pair in the section dictionary. If a value is a dictionary, it
        recursively applies the casting to the nested dictionary. If a value is not a dictionary, it attempts to cast
        the value to an appropriate type using the `cast_value` method.

        Parameters:
        section (Dict[Any, Any]): The section dictionary where the casting should be applied. The dictionary can have
        keys and values of any type.

        Returns:
        None
        """

        for key, value in section.items():
            if isinstance(value, dict):
                self.cast_section(value)
            else:
                section[key] = self.cast_value(value)

    def initialise_config(self):
        '''
        Initialise the configuration file
        If the configuration file doesn't exist, create it with the  ini_string
        If it does exist, read it into the config object

        Parameters
        ----------
        filename : str
            The name of the configuration file

        Returns
        -------
        None

        '''
        self.logger.info("")
        self.logger.info("INITIALISING CONFIGURATION FILE")
        change = False
        if not os.path.exists(self.filename):
            self.logger.info('No configuration file found')
            self.logger.info('Creating configuration file: %s', self.filename)
            self.config = ConfigObj(StringIO(self.ini_string),encoding='utf-8')
            self.config.filename = self.filename
            self.config.write()
            change=True
        else:
            #read configuration file into config object
            self.config = ConfigObj(self.filename,encoding='utf-8')
            self.logger.info('Configuration file found')


        #correct any problems with the config file
        self.correct_config()
        # Cast all values to their appropriate types
        self.cast_section(self.config)
        self.logger.info('Read Config object : %s', self.get_config())
        return change

    def update_config(self, section: str, subsection: Optional[str], keys_values: Dict[str, Any]) -> bool:
        """
        Updates the configuration with a new section, subsection, and key-value pairs.

        This function checks if the section exists in the configuration. If not, it creates it. If the subsection is not
        provided, it creates a new one with the name 'sitex', where x is the number of existing subsections in the section
        plus one. It then checks if the subsection exists in the section. If not, it creates it and adds the provided
        key-value pairs. If any changes are made, it saves the updated configuration to the file.

        Parameters:
        section (str): The name of the section to be updated.
        subsection (Optional[str]): The name of the subsection to be updated. If not provided, a new subsection is created.
        keys_values (Dict[str, Any]): The key-value pairs to be added to the subsection.

        Returns:
        bool: True if any changes were made to the configuration, False otherwise.
        """
        self.logger.info("")
        self.logger.info("UPDATING CONFIGURATION FILE")

        def strip_leading_spaces(multiline_string):
            lines = multiline_string.split('\n')
            stripped_lines = [line.lstrip() for line in lines]
            return '\n'.join(stripped_lines)

        change = False

        # Check if the section exists, if not create it
        if section not in self.config:
            self.logger.info('Adding %s section', section)
            self.config[section] = {}
            change = True

         # If subsection is empty, set it to 'sitex'
        if not subsection:
            subsection = 'site' + str(len(self.config[section]) + 1)
            self.logger.info('No subsection passed adding %s sub-section', subsection)
            change = True

        # Add a blank line above the section and a comment below the section name
        self.config.comments[section] = ['']

        # Check if the subsection exists, if not create it and add keys and values
        if subsection not in self.config[section]:
            self.config[section][subsection] = keys_values
            change = True
            self.logger.info('%s not found in %s, adding it', subsection, section)
            self.logger.info('Adding key values pairs %s', keys_values)

        # Save the changes to the file
        if change:
            self.cast_section(self.config)
            self.config.write()
            self.logger.info('Updated %s', self.filename)
            self.logger.info('Updated Config object: %s', dict(self.config))
        else:
            self.logger.info('No changes made to %s', self.filename)

        return change

    def get_config(self) -> dict:
        """
        Returns a string representation of the configuration.

        This function iterates over each section in the configuration. If a value is a Section, it recursively prints
        the types of the nested section. If a value is not a Section, it adds the key, value, and type of the value to
        the string. The string is indented to reflect the structure of the configuration.

        Returns:
        str: A string representation of the configuration, with keys, values, and types of values.
        """
        self.logger.info("")
        self.logger.info("GETTING CONFIGURATION FILE")

        return dict(self.config)

    def read_config(self):

        return dict(ConfigObj(self.filename,encoding='utf-8'))

    def correct_config(self):
        """
        This method corrects the configuration by normalizing section names,
        reconstructing the configuration in a specific order, ensuring default keys order and existence,
        processing the 'sites' section, and finally saving the configuration to the config.ini file.
        """
        self.logger.info("Starting to correct the configuration.")

        # Parse self.ini_string into a ConfigObj object
        ini_string_config = ConfigObj(self.ini_string.splitlines())
        self.logger.info("Parsed ini_string into a ConfigObj object.")

        # Normalize section names in ini_string_config
        self._normalize_section_names(ini_string_config)
        self.logger.info("Normalized section names in ini_string_config.")

        # Step 1: Normalize section names in self.config
        self._normalize_section_names(self.config)
        self.logger.info("Normalized section names in self.config.")

        # Step 2: Reconstruct self.config in the order of ini_string_config
        self._reconstruct_config(ini_string_config)
        self.logger.info("Reconstructed self.config in the order of ini_string_config.")

        # Step 3: In the 'defaults' section, convert keys to upper case and remove whitespaces
        if 'defaults' in self.config:
            self.config['defaults'] = {key.upper().replace(' ', ''): value for key, value in self.config['defaults'].items()}
            self.logger.info("Converted keys in the 'defaults' section to upper case and removed whitespaces.")

        # Step 4, 5: Ensure default keys order and existence
        if 'defaults' in ini_string_config:
            self._ensure_default_keys_order(self.config['defaults'], ini_string_config['defaults'])
            self.logger.info("Ensured default keys order and existence.")

        # Additional processing for 'sites' section
        self._process_sites_section()
        self.logger.info("Processed 'sites' section.")

        # Step 6: Save self.config to the config.ini file
        self.config.write()
        self.logger.info("Saved self.config to the config.ini file.")

        self.logger.info("Finished correcting the configuration.")

    def _normalize_section_names(self, section):
        """
        This method normalizes the names of the sections in the provided section object.
        It iterates over the keys of the section, and if the key corresponds to a Section instance,
        it normalizes the key by converting it to lower case and removing spaces, and then updates the section with the normalized key.
        """
        self.logger.info("Starting to normalize section names.")

        for key in list(section.keys()):
            if isinstance(section[key], Section):
                normalized_key = key.lower().replace(' ', '')
                section[normalized_key] = section.pop(key)
                self.logger.info(f"Normalized section name: {normalized_key}")

        self.logger.info("Finished normalizing section names.")

    def _reconstruct_config(self, reference_config):
        """
        This method reconstructs the configuration based on a reference configuration.
        It creates a new ConfigObj, then iterates over the sections in the reference configuration.
        If a section exists in the current configuration, it is copied to the new configuration.
        If not, the section from the reference configuration is used.
        Finally, the current configuration is cleared and updated with the new configuration.
        """
        self.logger.info("Starting to reconstruct the configuration.")

        new_config = ConfigObj()
        for section in reference_config:
            if section in self.config:
                new_config[section] = self.config[section]
                self.logger.info(f"Copied section {section} from the current configuration.")
            else:
                new_config[section] = reference_config[section]
                self.logger.info(f"Copied section {section} from the reference configuration.")

        self.config.clear()
        self.logger.info("Cleared the current configuration.")

        self.config.update(new_config)
        self.logger.info("Updated the current configuration with the new configuration.")

        self.logger.info("Finished reconstructing the configuration.")

    def _process_sites_section(self):
        """
        This method processes the 'sites' section of the configuration.
        It strips leading and trailing spaces from subsection names, normalizes keys in subsections,
        and ensures the presence and order of specific keys.
        """
        self.logger.info("Starting to process 'sites' section.")

        ini_string_config = ConfigObj(self.ini_string.splitlines())
        if 'sites' in self.config:
            for subsection in list(self.config['sites'].keys()):
                # Only strip leading and trailing spaces from subsection names
                stripped_subsection = subsection.strip()
                if stripped_subsection != subsection:
                    self.config['sites'][stripped_subsection] = self.config['sites'].pop(subsection)
                    self.logger.info(f"Stripped spaces from subsection name: {stripped_subsection}")
                else:
                    stripped_subsection = subsection

                # Step 2: Normalize keys in subsections
                self.config['sites'][stripped_subsection] = {
                    key.lower().replace(' ', ''): value
                    for key, value in self.config['sites'][stripped_subsection].items()
                }
                self.logger.info(f"Normalized keys in subsection: {stripped_subsection}")

                # Step 3 and 4: Ensure the presence and order of specific keys
                required_keys = ['latitude', 'longitude', 'bortle', 'sqm']
                default_values = {
                    'latitude': ini_string_config['defaults'].get('SITELAT', ''),
                    'longitude': ini_string_config['defaults'].get('SITELONG', ''),
                    'bortle': ini_string_config['defaults'].get('BORTLE', ''),
                    'sqm': ini_string_config['defaults'].get('SQM', '')
                }

                # Create a new ordered dictionary
                new_dict = {}
                for key in required_keys:
                    if key in self.config['sites'][stripped_subsection]:
                        new_dict[key] = self.config['sites'][stripped_subsection][key]
                    else:
                        new_dict[key] = default_values[key]

                # Add any remaining keys from the original dictionary
                for key, value in self.config['sites'][stripped_subsection].items():
                    if key not in new_dict:
                        new_dict[key] = value

                # Replace the old dictionary with the new one
                self.config['sites'][stripped_subsection] = new_dict
                self.logger.info(f"Updated subsection: {stripped_subsection} with new dictionary.")

        self.logger.info("Finished processing 'sites' section.")

    def _ensure_default_keys_order(self, defaults_section, reference_defaults):
        """
        This method ensures the order and presence of keys in the 'defaults' section of the configuration.
        It creates an ordered dictionary, then iterates over the keys in the reference defaults.
        If a key exists in the defaults section, it is copied to the ordered dictionary.
        If not, the key from the reference defaults is used.
        Finally, the defaults section is cleared and updated with the ordered dictionary.
        """
        self.logger.info("Starting to ensure default keys order.")

        ordered_keys = {}
        for key in reference_defaults:
            normalized_key = key.upper().replace(' ', '')
            if normalized_key in defaults_section:
                ordered_keys[normalized_key] = defaults_section[normalized_key]
                self.logger.info(f"Copied key {normalized_key} from the defaults section.")
            else:
                ordered_keys[normalized_key] = reference_defaults[key]
                self.logger.info(f"Copied key {normalized_key} from the reference defaults.")

        for key in defaults_section:
            if key not in ordered_keys:
                ordered_keys[key] = defaults_section[key]
                self.logger.info(f"Added key {key} to the ordered keys.")

        defaults_section.clear()
        self.logger.info("Cleared the defaults section.")

        defaults_section.update(ordered_keys)
        self.logger.info("Updated the defaults section with the ordered keys.")

        self.logger.info("Finished ensuring default keys order.")

class Headers(Configuration):
    def __init__(self,config_filename,logger,dp):
        # Call the __init__ method of the Configuration class
        super().__init__(config_filename,logger = logger)
        self.number_of_images_processed = 0
        # Create a dictionary to store header data
        self.header = {}
        #create a list to store multiple header data
        self.headers = pd.DataFrame()
        self.headers_reduced = []
        #floating point precision
        self.dp = dp
        self.logger = logger
        self.logger.info("")
        self.logger.info("INITIALISING HEADERS CLASS")

    def read_xisf_header(self, file_path: str) -> Optional[None]:
        """
        Reads the header of an XISF file.

        Opens and reads the XISF file specified by 'file_path'. It checks for the XISF signature,
        reads the header, and stores it as an instance variable. If the file is not a valid XISF file or
        an error occurs, it returns None.

        Parameters:
        - file_path (str): The path to the XISF file.

        Returns:
        - None
        """
        xml_header = ""
        self.logger.info("")
        self.logger.info("READING XISF HEADER")
        try:
            with open(file_path, 'rb') as file:
                signature = file.read(8).decode('ascii')
                # Check for the XISF signature
                if signature != 'XISF0100':
                    self.logger.error("Invalid file format")
                    return None
                # Read and skip header length and reserved field
                header_length = struct.unpack('<I', file.read(4))[0]
                file.read(4)  # Skip reserved field
                xml_header = file.read(header_length).decode('utf-8')
                self.logger.info('Successfully read XISF header')

        except Exception as e:
            self.logger.error("Error reading XISF header from %s: %s", file_path, e)
            return None

        return xml_header

    def xml_to_data(self,xml_header) -> Optional[None]:
        """
        Parses the XML header data and stores it in the instance variable 'header'.

        This function registers the XISF namespace, parses the XML header data, and iterates through each 'FITSKeyword'
        tag in the XML. It adds the 'name' and 'value' pair to the 'header' dictionary. If the comment contains
        'Integration with PixInsight', it extracts the software creation name. If the comment contains
        'ImageIntegration.numberOfImages:', it extracts the number of images used to create the master and stores it in
        the instance variable 'number'.

        Returns:
        - None
        """
        header = {}
        self.logger.info("")
        self.logger.info("CONVERTING XML HEADER DATA TO DICTIONARY")
        # Register the namespace
        ns = {'xisf': 'http://www.pixinsight.com/xisf'}
        ET.register_namespace('', ns['xisf'])

        # Parse the XML data
        root = ET.fromstring(xml_header)
        self.logger.info('Successfully parsed XML header data')

        # Iterate through each 'FITSKeyword' tag in the XML
        for fits_keyword in root.findall('.//xisf:FITSKeyword', namespaces=ns):
            name = fits_keyword.get('name')
            value = fits_keyword.get('value')
            comment = fits_keyword.get('comment')

            # Add the 'name', 'value' pair to the dictionary
            header[name] = value
            #self.logger.info('Added %s: %s to header dictionary', name, value)

            # Deal with master image files and extract number of images used to create master
            # Search for the the creation software name
            if 'Integration with PixInsight' in comment:
                header['SWCREATE'] = ' '.join(comment.split()[2:])
                #self.logger.info('Found software creation name in comment: %s', header['SWCREATE'])

            # Halt search when number of images text is found in the comment
            if 'ImageIntegration.numberOfImages:' in comment:
                # Find the number in the string
                match = re.search(r'ImageIntegration.numberOfImages: (\d+)', comment)
                if match:
                    # Extract the number and convert it to an integer
                    self.number = int(match.group(1))
                    #self.logger.info('Found number of images in comment: %d', self.number)
                    return header
        return header

    def get_number_of_FIT_cal_frames(self,header) -> Optional[None]:
        """
        Extracts the number of calibration frames from the 'HISTORY' lines in the header.

        This function checks if 'HISTORY' is in the header. If it is, it iterates over each line in 'HISTORY'.
        If a line contains 'Integration with PixInsight', it extracts the software creation name. If a line contains
        'ImageIntegration.numberOfImages:', it extracts the number at the end of the line and stores it in the instance
        variable 'number'.

        Returns:
        - None
        """
        self.logger.info("")
        self.logger.info("GETTING NUMBER OF CALIBRATION FRAMES FROM MASTER HEADER FOR FITS FILE")
        # Check if 'HISTORY' is in the header
        if 'HISTORY' in header:
            # Get the 'HISTORY' lines
            history_lines = self.header['HISTORY']
            self.logger.info('Found HISTORY in header')

            # Iterate over each line
            for line in history_lines:
                # Search for the the creation software name
                if 'Integration with PixInsight' in line:
                    header['SWCREATE'] = ' '.join(line.split()[2:])
                    self.logger.info('Found software creation name in HISTORY: %s', header['SWCREATE'])

                # Search for the substring 'ImageIntegration.numberOfImages:'
                if 'ImageIntegration.numberOfImages:' in line:
                    # Extract the number at the end of the line
                    number = line.split()[-1]
                    # Set the variable self.number
                    self.number = int(number)
                    self.logger.info('Found number of calibration frames in HISTORY: %d', self.number)
                    return header
        return header

    def clean_object_column(self,df):
        for i in df.index:
            if df.loc[i, 'IMAGETYP'] == 'LIGHT':
                parts = re.split(r'\s|_', df.loc[i, 'OBJECT'])
                parts = [part.strip() for part in parts if part]  # remove empty strings
                if len(parts) >= 3 and parts[-1].isdigit():
                    parts[-2] = 'Panel'
                df.loc[i, 'OBJECT'] = ' '.join(parts)
        return df

    def process_headers(self, file_path: str) -> Optional[None]:
        """
        Processes the headers of the file at the given path.

        This function sets the instance variables 'header_filename', 'number', 'wanted_keys', and 'header'. It checks the
        file extension and processes the file accordingly. If the file is a FITS file, it opens the file and extracts the
        header. If the file is an XISF file, it reads and parses the XISF file header. It then converts the header to a
        dictionary and checks if 'IMAGETYP' is in the header. If 'IMAGETYP' is not in the header, it removes the frame
        from 'self.header'. If 'self.number' is greater than 1, it logs that the image is a master and the number of
        images used to create the master. It then reduces the headers and accumulates them.

        Args:
        - file_path: The path to the file to process.

        Returns:
        - None
        """
        self.number = 1

        # Create a keys dictionary to select the wanted keys in the header
        self.wanted_keys = list(self.config['defaults'].keys())
        self.wanted_keys.extend(['FILENAME', 'NUMBER', 'EGAIN', 'INSTRUME'])
        #self.wanted_keys.extend(['FILENAME', 'NUMBER', 'INSTRUME'])
        hdr= {}
        reduced_header = {}


        # Check file extension and process accordingly
        if file_path.lower().endswith(('.fits', '.fit', '.fts', '.xisf')):
            self.header_filename = os.path.basename(file_path)
            self.logger.info('Processing headers for file: %s', self.header_filename)
            fp_lower = file_path.lower()
            try:
                if fp_lower.endswith(('.fits', '.fit', '.fts')):
                    # Open FITS file and extract header
                    with fits.open(file_path) as hdul:
                        hdr= hdul[0].header
                        hdr= self.get_number_of_FIT_cal_frames(hdr)
                        self.logger.info('Opened FITS file and extracted header')
                elif fp_lower.endswith('.xisf'):
                    # Read and parse XISF file header
                    xml_header = self.read_xisf_header(file_path)
                    hdr= self.xml_to_data(xml_header)
                    self.logger.info('Read and parsed XISF file header')

                # Convert header to dictionary and process
                hdr= dict(hdr)
                self.logger.info('Converted header to dictionary')

                # Check 'IMAGETYP' key in header
                if 'IMAGETYP' not in hdr:
                    self.logger.info('IMAGETYP key not found in header %s', fp_lower)
                    self.logger.info('Dropping header from analysis')
                    # Remove frame from self.header
                    hdr= {}
                    return
                # Modified 28th September 2024
                # Modify 'IMAGETYP' if it's 'LIGHTFRAME' or "'LIGHTFRAME'"
                if hdr['IMAGETYP'] in ['LIGHTFRAME', "'LIGHTFRAME'","'Light Frame'",'Light Frame']:
                    hdr['IMAGETYP'] = 'LIGHT'
                    self.logger.info("IMAGETYP modified from 'LIGHTFRAME' to 'LIGHT'")

                if self.number > 1:
                    self.logger.info('Image is a master, number of images used to create master: %s', self.number)
                else:
                    self.logger.info('Image is not a master')
                    #hdr= self.clean_object_column(self.hdr)

                self.logger.info('Header = %s', hdr)
                reduced_header = self.reduce_headers(hdr)
                self.logger.info('Reduced header = %s', reduced_header)

                # Accumulate headers
                #hdrs.append(reduced_header)
                self.logger.info("")
                self.logger.info("*"*200)
                self.logger.info("")

            except Exception as e:
                self.logger.error('Error reading %s: %s', file_path, e)

        return reduced_header

    def process_directory(self, directory: str) -> pd.DataFrame:
        # Walk through the directory
        dir_headers = []
        for root, _, files in os.walk(directory,followlinks=True):
            self.logger.info("Extracting headers from directory: %s", root)
            print("Extracting headers from directory: ", root  )
            # Process each file in the directory
            self.logger.info("")
            self.logger.info("PROCESSING HEADERS")
            self.logger.info("*"*200)
            for file in files:
                #self.logger.info("Processing file: %s", file)
                file_path = os.path.join(root, file)  # Full path of the file
                header = self.process_headers(file_path)
                if header:
                    dir_headers.append(header)
                else:
                    self.logger.info("")
                    self.logger.info('No header found for file: %s', file_path)
                    self.logger.info("")


        return dir_headers

    def apply_light_values(self,df: pd.DataFrame) -> pd.DataFrame:
        # Find the first row where 'IMAGETYP' is 'LIGHT'
        #return df
        #print('Processing site:',df['SITE'].iloc[0])
        #print()
        #print(df['IMAGETYP'].unique())
        light_row = df[df['IMAGETYP'] == 'LIGHT'].iloc[0]
        #print('Light row:',light_row )
        #print()

        # Extract the 'SITE', 'SITELAT', and 'SITELONG' values
        site = light_row['SITE']
        sitelat = light_row['SITELAT']
        sitelong = light_row['SITELONG']

        # Apply these values to all rows that are not of 'IMAGETYP' 'LIGHT'
        df.loc[df['IMAGETYP'] != 'LIGHT', ['SITE', 'SITELAT', 'SITELONG']] = site, sitelat, sitelong

        return df

    def process_and_accumulate_headers(self, headers):
        # Convert headers to DataFrame and apply additional processing
        df = pd.DataFrame(headers)
        processed_df = self.apply_light_values(df)
        # Accumulate the processed DataFrame into self.headers
        self.headers = pd.concat([self.headers, processed_df])
        return

    def count_subdirectories(self,root):
        subdirectory_count = 0
        for root, dirs, files in os.walk(root):
            subdirectory_count += len(dirs)
        return subdirectory_count

    def process_directories(self, directories: List[str]) -> pd.DataFrame:
        """
        Processes the directories provided and extracts headers from the files in them.

        This function iterates over each directory provided, walks through the directory, and processes each file in the
        directory. It then converts 'self.headers' to a DataFrame and checks if it contains LIGHT frames. If it does not,
        it ends the program. It then checks the camera gains, sets 'self.headers['GAIN']' to an absolute value, and sets
        HFR, IMSCALE, and FWHM values.

        Args:
        - directories: A list of directories to process.

        Returns:
        - A DataFrame of the processed headers.
        """
        self.logger.info("")
        self.logger.info("PROCESSING DIRECTORIES")
        headers = []
        for directory in directories:
            headers+= self.process_directory(os.path.realpath(directory))

        conditioned_headers = self.condition_headers(headers)

        return conditioned_headers,pd.DataFrame(headers)

    def condition_headers(self, headers):

        # Convert self.headers to a dataframe
        headers = pd.DataFrame(headers)
        self.logger.info('Read Headers to DataFrame')
        print("\nRead Headers to DataFrame")

        # Check self.headers contains LIGHT frames
        if headers.empty or 'LIGHT' not in headers['IMAGETYP'].values:
            self.logger.info("No LIGHT 2 frames found in headers")
            print("\nNo LIGHT frames found in headers")
            self.logger.info("Exiting program")
            print("\nExiting program")
            sys.exit(0)


        # check if master dark or bias frames are present if so drop bias and dark frames
        self.logger.info('Check if master dark or bias frames are present and drop bias and dark frames if they are.')
        print("\nCheck if master dark or bias frames are present and drop bias and dark frames if they are.")
        headers = self.deal_with_calibration_frames(headers)


        #check if master flat frames are present if so drop flat frames
        self.logger.info("")
        self.logger.info('Check for master flat frames, if they exist drop flat frames')
        print("\nCheck for master flat frames, if they exist drop flat frames\n")
        headers = self.deal_with_flat_frames(headers)

        # remove duplicate LIGHT frames having WBPP post fixes
        self.logger.info("")
        self.logger.info('Check for duplicate LIGHT frames having WBPP post fixes')
        print("\nCheck for duplicate LIGHT frames having WBPP post fixes\n")
        headers = self.filter_and_remove_duplicates(headers)

        #convert all headers['IMAGETYPE'] to upper case
        headers['IMAGETYP'] = headers['IMAGETYP'].str.upper()
        self.logger.info('Converted all IMAGETYP to upper case')

        # Check camera gains
        headers = self.check_camera_gains(headers)
        self.logger.info('Checked camera gains')

        # Update 'GAIN' and 'EGAIN' where 'GAIN' is -1, for any residual values
        headers.loc[headers['GAIN'] == -1, 'EGAIN'] = self.config['defaults']['EGAIN']
        headers.loc[headers['GAIN'] == -1, 'GAIN'] = self.config['defaults']['GAIN']

        # Set headers['GAIN'] to an absolute value
        headers['GAIN'] = headers['GAIN'].abs()
        self.logger.info("")
        self.logger.info('Set GAIN to absolute value')

        # Set HFR, IMSCALE and FWHM values
        headers = self.get_HFR(headers)
        self.logger.info("")
        self.logger.info(f"Number of images processed: {self.number_of_images_processed}")
        self.logger.info('Set HFR, IMSCALE and FWHM values')

        # Set All calibration frame lat/long to the same as the first light frame nearest to the calibration frame Lat/Long
        # this stops calibration frames being associated with the wrong site
        headers = self.modify_lat_long(headers)
        self.logger.info("")
        self.logger.info('Set All calibration frame lat/long to the same as the first light frame nearest to the calibration frame Lat/Long')


        #clean up object column
        headers = self.clean_object_column(headers)

        return headers

    def modify_lat_long_very_old(self,df):
        # Create a DataFrame of 'LIGHT' rows
        light_df = df[df['IMAGETYP'] == 'LIGHT']

        # For each row in df, find the 'LIGHT' row that is closest in terms of 'SITELAT' and 'SITELONG'
        for i, row in df[df['IMAGETYP'] != 'LIGHT'].iterrows():
            # Calculate the Euclidean distance between the 'SITELAT' and 'SITELONG' of this row and each 'LIGHT' row
            distances = np.sqrt((light_df['SITELAT'] - row['SITELAT'])**2 + (light_df['SITELONG'] - row['SITELONG'])**2)
            # Find the index of the 'LIGHT' row with the smallest distance
            closest_light_index = distances.idxmin()
            # Set the 'SITELAT' and 'SITELONG' of this row to the 'SITELAT' and 'SITELONG' of the closest 'LIGHT' row
            df.at[i, 'SITELAT'] = light_df.at[closest_light_index, 'SITELAT']
            df.at[i, 'SITELONG'] = light_df.at[closest_light_index, 'SITELONG']

        return df

    def modify_lat_long_old(self, df):
        # Create a DataFrame of 'LIGHT' rows
        light_df = df[df['IMAGETYP'] == 'LIGHT']
        self.logger.info("")

        # For each row in df, find the 'LIGHT' row that is closest in terms of 'SITELAT' and 'SITELONG'
        for i, row in df[df['IMAGETYP'] != 'LIGHT'].iterrows():
            # Print the current row's 'SITELAT' and 'SITELONG' values
            print(f"Current row SITELAT: {row['SITELAT']}, SITELONG: {row['SITELONG']}")

            # Calculate the Euclidean distance between the 'SITELAT' and 'SITELONG' of this row and each 'LIGHT' row
            try:
                distances = np.sqrt((light_df['SITELAT'] - row['SITELAT'])**2 + (light_df['SITELONG'] - row['SITELONG'])**2)
            except TypeError as e:
                # Print the light_df 'SITELAT' and 'SITELONG' values to identify the problematic values
                print(f"Error calculating distances. light_df SITELAT: {light_df['SITELAT'].tolist()}, SITELONG: {light_df['SITELONG'].tolist()}")
                raise e

            # Find the index of the 'LIGHT' row with the smallest distance
            closest_light_index = distances.idxmin()
            # Set the 'SITELAT' and 'SITELONG' of this row to the 'SITELAT' and 'SITELONG' of the closest 'LIGHT' row
            df.at[i, 'SITELAT'] = light_df.at[closest_light_index, 'SITELAT']
            df.at[i, 'SITELONG'] = light_df.at[closest_light_index, 'SITELONG']

        return df
    
    def modify_lat_long(self, df):
        # Ensure SITELAT and SITELONG in light_df are floats
        light_df = df[df['IMAGETYP'] == 'LIGHT'].copy()
        light_df['SITELAT'] = light_df['SITELAT'].astype(float)
        light_df['SITELONG'] = light_df['SITELONG'].astype(float)

        for i, row in df[df['IMAGETYP'] != 'LIGHT'].iterrows():
            # Ensure current row's SITELAT and SITELONG are floats
            sitelat = float(row['SITELAT'])
            sitelong = float(row['SITELONG'])

            # Calculate Euclidean distance
            distances = np.sqrt(np.square(light_df['SITELAT'] - sitelat) + np.square(light_df['SITELONG'] - sitelong))
            
            # Find the index of the 'LIGHT' row with the smallest distance
            closest_light_index = distances.idxmin()
            
            # Update the 'SITELAT' and 'SITELONG' of the current row
            df.at[i, 'SITELAT'] = light_df.at[closest_light_index, 'SITELAT']
            df.at[i, 'SITELONG'] = light_df.at[closest_light_index, 'SITELONG']

        return df

    def reduce_headers(self, hdr: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reduces the headers by dealing with specific keys and creating a sub dictionary of wanted keys.

        This function deals with the 'CREATOR' key instead of 'SWCREATE', the 'EXPTIME' key instead of 'EXPOSURE', removes
        space and capitalises 'IMAGETYP' and 'INSTRUME', sets the file name to 'self.header_filename', and adds a 'number'
        key-value pair to 'self.updated_header'. It then removes double quotes and single quotes from values in 'hdr' that
        are strings, and creates a sub dictionary of wanted keys from 'hdr'. If a wanted key is not in 'hdr' or the value
        is None or '', it uses the default value from the config file.

        Args:
        - hdr: The headers to reduce.

        Returns:
        - The reduced headers.
        """
        self.logger.info("")
        self.logger.info("REDUCING HEADERS")

        # Deal with SGP CREATOR key instead of SWCREATE
        if 'CREATOR' in hdr:
            hdr['SWCREATE'] = hdr['CREATOR']
            self.logger.info('Dealt with CREATOR key')

        # Deal with EXPTIME key instead of EXPOSURE
        if 'EXPTIME' in hdr and hdr['EXPTIME'] != 'No Data':
            hdr['EXPOSURE'] = hdr['EXPTIME']
            self.logger.info('Dealt with EXPTIME key')
        #masters use LAT-OBS and LONG-OBS forSITELAT and SITELONG'
        # so convert here if they exist
        if 'LAT-OBS' in hdr and hdr['LAT-OBS'] != 'No Data':
            hdr['SITELAT'] = hdr['LAT-OBS']
            hdr['SITELONG'] = hdr['LONG-OBS']
            hdr['FWHM'] = 0
            self.logger.info('Dealt with Masters LAT-OBS and LONG-OBS')
        
        #Deal with fractional seconds in DATE-OBS
        #added 29th September 2024
        #if 'DATE-OBS' in hdr:
        #    hdr['DATE-OBS'] = hdr['DATE-OBS'].split('.')[0]

        # Deal with SITENAME key
        if 'SITENAME' in hdr:
            hdr['SITE'] = hdr['SITENAME']
            self.logger.info('Dealt with SITENAME key')

        # Deal with FOCUSER key
        if 'FOCUSER' in hdr:
            hdr['FOCNAME'] = hdr['FOCUSER']
            self.logger.info('Dealt with FOCUSER key')

        # Remove space and capitalise hdr['IMAGETYP']

        hdr['IMAGETYP'] = hdr['IMAGETYP'].replace(' ', '').upper()
        self.logger.info('Removed space and capitalised IMAGETYP')

        # Set file name to self.updated_header
        if 'FILENAME' not in hdr:
            hdr['FILENAME'] = self.header_filename
            self.logger.info('Set file name to self.header_filename')

        # Add 'number' key-value pair to self.updated_header
        hdr['NUMBER'] = self.number
        self.logger.info('Added number key-value pair to self.updated_header')

        # Remove space and capitalise hdr['INSTRUME']
        hdr['INSTRUME'] = hdr['INSTRUME'].upper()
        self.logger.info('Removed space and capitalised INSTRUME')

        #if hdr['GAIN'] does not exist create it and set it to -1
        hdr.setdefault('GAIN', -1)

        #if hdr['EGAIN'] does not exist create it and set it to 1
        hdr.setdefault('EGAIN', 1)

        # If a value in hdr is a string, remove double quotes and single quotes
        hdr = {k: v.strip('"').strip("'") if isinstance(v, str) else v for k, v in hdr.items()}
        self.logger.info('Removed double quotes and single quotes from string values in hdr')

        # Create a sub dictionary of wanted keys from hdr
        hdr = {k: hdr[k] if k in hdr and hdr[k] != '' else self.config['defaults'][k] for k in self.wanted_keys}
        self.logger.info('Created sub dictionary of wanted keys from hdr')

        return self.check_and_convert_data_types(hdr)
    def filter_and_remove_duplicates(self, headers_df: pd.DataFrame) -> pd.DataFrame:

        # Filter for IMAGETYP = 'LIGHT'
        #light_df = headers_df[headers_df['IMAGETYP'] == 'LIGHT']
        #light_df = headers_df[headers_df['IMAGETYP'] == 'LIGHT']

        # Initialize an empty DataFrame to store the final results
        final_df = pd.DataFrame()

        # Remove rows where IMAGETYP is 'masterlight'
        headers_df= headers_df[headers_df['IMAGETYP'] != 'masterlight']

        # Extract the base filenames
        headers_df['base_filename'] = headers_df['FILENAME'].str.extract(r'(.+?)(?:_c.*)?(\.xisf|\.fits|\.fit|\.fts)')[0]

        # Check for duplicates
        if headers_df['base_filename'].duplicated().any():
            self.logger.info('Duplicates found and removed.')
            print('Duplicates found and removed.')
        else:
            self.logger.info('No duplicates found.')
            print('No duplicates found.')


        # Iterate over unique base filenames
        for base_filename in headers_df['base_filename'].unique():
            # Check if the base filename with primary extensions exists
            for ext in ['.xisf', '.fits', '.fit', '.fts']:
                match = headers_df[headers_df['FILENAME'] == f'{base_filename}{ext}']
                if not match.empty:
                    final_df = pd.concat([final_df, match])
                    break
            else:
                # Try to find a match for the filename with '_c.xisf'
                match_c = headers_df[headers_df['FILENAME'] == f'{base_filename}_c.xisf']
                if not match_c.empty:
                    # Use the header data from the '_c.xisf' file if it exists
                    final_df = pd.concat([final_df, match_c])
                else:
                    # Default to '.xisf' if no primary extension is found
                    default_filename = f'{base_filename}.xisf'
                    default_row = headers_df[headers_df['base_filename'] == base_filename].iloc[0].copy()
                    default_row['FILENAME'] = default_filename
                    final_df = pd.concat([final_df, pd.DataFrame(default_row).transpose()])

        final_df.drop(columns=['base_filename'], inplace=True)


        return final_df.reset_index(drop=True)

    def deal_with_calibration_frames(self,df):
        # Convert 'IMAGETYP' to lower case for case insensitive comparison
        df['IMAGETYP'] = df['IMAGETYP'].str.lower()

        # Remove any masterdark frames that dont have the same duration as the light frames
        #identify the duration times in the light frames
        #identify the filters used in the light frames
        light_exposures = df[df['IMAGETYP']=='light']['EXPOSURE'].unique()
        #identify the filters used in the flat frames and master flat frames
        dark_exposures = df[df['IMAGETYP'].str.contains('dark')]['EXPOSURE'].unique()

        #remove from df any dark frames or master dark frames that do not have a corresponding exposure in light_filters
        for exposure_value in dark_exposures:
            if exposure_value not in light_exposures :
                self.logger.info(f'No light frame with exposure  {exposure_value} found: dropping {exposure_value} dark frames.')
                df = df[~(df['IMAGETYP'].str.contains('dark') & (df['EXPOSURE'] == exposure_value))]



        master_types = ['masterbias', 'masterdark', 'masterflat', 'masterflatdark']

        for master_type in master_types:
            if any(master_type in s for s in df['IMAGETYP']):
                removeimgtype =master_type.replace("master","")
                print(f'{master_type} found removing {master_type.replace("master","")} frames')
                self.logger.info(f'{master_type} found removing {removeimgtype} frames')
                self.logger.info('Removing bias and dark frames')
                df = df[df['IMAGETYP'] != removeimgtype ]

        return df

    def deal_with_flat_frames(self,df):
        df['IMAGETYP'] = df['IMAGETYP'].str.lower()
        #identify the filters used in the light frames
        light_filters = df[df['IMAGETYP']=='light']['FILTER'].unique()
        #identify the filters used in the flat frames and master flat frames
        flat_filters = df[df['IMAGETYP'].str.contains('flat')]['FILTER'].unique()

        #remove from df any flat frames or master flat frames that do not have a corresponding filter in light_filters
        for filter_value in flat_filters:
            if filter_value not in light_filters:
                self.logger.info(f'No light frame with filter {filter_value} found: dropping {filter_value} flat frames.')
                df = df[~(df['IMAGETYP'].str.contains('flat') & (df['FILTER'] == filter_value))]

        # Find the filter values for 'master flat' frames
        master_flat_filters = df[(df['IMAGETYP'].str.contains('master')) & (df['IMAGETYP'].str.contains('flat'))]['FILTER'].unique()
        detail_format = "{:<30} {:<5} {:<10} {:<5} {:<10}"
        # Log that a master flat frame with a certain filter type was found
        for filter_value in master_flat_filters:
            self.logger.info(f'Master flat frame with filter {filter_value} found: dropped {filter_value} flat frames.')
            print(detail_format.format('Master flat frame with filter', filter_value, 'found: dropped', filter_value, 'flat frames.'))

        # Remove 'flat' frames with the same filter values
        for filter_value in master_flat_filters:
            df = df[~((df['IMAGETYP'] == 'flat') & (df['FILTER'] == filter_value))]

        # If there is more than one master flat frame with the same filter then:
        # 1. Add the ['NUMBER'] values and make ['NUMBER'].
        # 2. Drop all entries except one, it does not matter which one is kept
        master_flat_df = df[df['IMAGETYP'].str.contains('master') & df['IMAGETYP'].str.contains('flat')]
        master_flat_df['NUMBER'] = master_flat_df.groupby('FILTER')['NUMBER'].transform('sum')
        master_flat_df = master_flat_df.drop_duplicates('FILTER')

        # Replace the old master flat frames with the new ones
        df = df[~(df['IMAGETYP'].str.contains('master') & df['IMAGETYP'].str.contains('flat'))]
        df = pd.concat([df, master_flat_df])

        return df

    def check_and_convert_data_types(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks and converts the data types of the values in the dictionary provided.

        This function defines the expected data types for each key in the dictionary, and then iterates over each key-value
        pair in the dictionary. If the value is a string, it removes double quotes and strips outer single quotes, and if
        the key is 'DATE-OBS', it converts the value to a datetime string in the format 'YYYY-MM-DD'. If the key is
        'SITELAT' or 'SITELONG' and the value contains a space, it converts the value from DMS format to decimal format.
        It then tries to convert the value to the expected data type. If it cannot, it prints an error message. If the key
        is not found in the dictionary, it prints a message and the dictionary.

        Args:
        - d: The dictionary to check and convert the data types of.

        Returns:
        - The dictionary with the data types of the values checked and converted.
        """
        self.logger.info("")
        self.logger.info('CHECKING AND CONVERTING DATA TYPES')

        def dms_to_decimal_old(dms_str: str) -> float:
            """
            Converts a string in DMS format to decimal format.

            Args:
            - dms_str: The string in DMS format to convert.

            Returns:
            - The decimal representation of the DMS string.
            """
            try:
                degrees, minutes, seconds = map(float, dms_str.split())
                return degrees + minutes / 60 + seconds / 3600
            except Exception:
                return dms_str
            

        def dms_to_decimal(dms_str: str) -> float:
            """
            Converts a string in DMS format to decimal format. Supports both '0d7m0.000s W' format
            and the original space-separated format.

            Args:
            - dms_str: The string in DMS format to convert.

            Returns:
            - The decimal representation of the DMS string.
            """
            try:
                # Check if the string is in '0d7m0.000s W' format
                if 'd' in dms_str and 'm' in dms_str and 's' in dms_str:
                    parts = dms_str.replace('d', ' ').replace('m', ' ').replace('s', ' ').split()
                    degrees, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
                    direction = parts[3] if len(parts) == 4 else ''
                else:
                    # Original functionality for space-separated format
                    degrees, minutes, seconds = map(float, dms_str.split())
                    direction = ''

                decimal = degrees + minutes / 60 + seconds / 3600

                # Adjusting for West (W) and South (S) coordinates to be negative
                if direction in ['W', 'S']:
                    decimal = -decimal

                return decimal
            except Exception as e:
                print(f"Error converting DMS to decimal: {e}")
                return 0.0  # Consider returning a default value or re-raise the exception


        # Define data types for each key
        data_types = {
            'IMAGETYP': str,
            'EXPOSURE': float,
            'DATE-OBS': str,  # datetime string format YYY-MM-DD
            'XBINNING': int,
            'GAIN':  int,
            'EGAIN': float,
            'XPIXSZ': float,
            'CCD-TEMP': float,
            'FOCALLEN': int,
            'FOCRATIO': float,
            'FILTER': str,
            'FOCTEMP': float,
            'SWCREATE': str,
            'HFR':   float,
            'FWHM': float,
            'SITELAT': float,
            'SITELONG': float,
            'OBJECT': str,
            'FILENAME': str,
            'NUMBER': int,
            'INSTRUME': str
        }

        for key, expected_type in data_types.items():
            if key in d:
                v = d[key]
                #self.logger.info(f"KEY: {key} VALUE: {v} EXPECTED TYPE: {expected_type}")

                if isinstance(v, str):
                    v = v.replace('"', '').strip("'")  # Remove double quotes and strip outer single quotes
                    if key == 'DATE-OBS':
                        # Check if fractional seconds exist (i.e., there's a '.' in the time part)
                        if '.' in v:
                            # Remove everything after the decimal point (i.e., fractional seconds)
                            v = v.split('.')[0]                        
                        # Convert the string to the correct datetime format (without microseconds)
                        v = datetime.strptime(v, "%Y-%m-%dT%H:%M:%S").strftime('%Y-%m-%dT%H:%M:%S')


                    if key in ['SITELAT', 'SITELONG'] and ' ' in v:  # DMS format
                        v = dms_to_decimal(v)

                try:
                    if expected_type == float:
                        d[key] = round(float(v), self.dp)
                    elif expected_type == int:
                        d[key] = int(float(v))
                    elif expected_type == str:
                        d[key] = str(v)
                except ValueError:
                    self.logger.error(f"Error converting {key}: could not convert {v} to {expected_type}")
            else:
                self.logger.error(f"Key '{key}' not found in dictionary")
                self.logger.error(d)

        return d

    def check_camera_gains(self,hdrs) -> pd.DataFrame:
        """
        Checks the gains of the cameras found in the headers.

        This function logs the start of the operation, then creates a DataFrame of unique rows from the headers where the
        gain is greater than or equal to 0. It then applies a function to the 'INSTRUME', 'EGAIN', and 'GAIN' columns of
        the DataFrame that replaces empty or None values with the default values from the configuration. It then applies a
        function to the 'GAIN' column of the headers that replaces values less than or equal to 0 with the corresponding
        value from the DataFrame if it exists, otherwise it leaves the value as is. It then returns the DataFrame.

        Returns:
        - The DataFrame of unique rows from the headers where the gain is greater than or equal to 0, with empty or None
        values replaced with the default values from the configuration.
        """
        self.logger.info("")
        self.logger.info("CHECKING CAMERA GAINS")

        # Replace 'No Data' with 0 in 'Gain' column
        hdrs['GAIN'] = hdrs['GAIN'].replace('No Data', 0)

        unique_df = hdrs[hdrs['GAIN'] >= 0][['INSTRUME', 'GAIN', 'EGAIN']].drop_duplicates()
        self.logger.info('Created DataFrame of unique rows from headers')

        # If any unique_df['INSTRUME'] or df['EGAIN'] or df['GAIN'] is empty or None
        # Set the value to those taken from self.config['defaults'] for INSTRUM, EGAIN and GAIN
        unique_df['INSTRUME'] = unique_df['INSTRUME'].apply(lambda x: self.config['defaults']['INSTRUME'] if x == '' or x is None else x)
        unique_df['EGAIN'] = unique_df['EGAIN'].apply(lambda x: self.config['defaults']['EGAIN'] if x == '' or x is None or x < 0 else x)
        unique_df['GAIN'] = unique_df['GAIN'].apply(lambda x: self.config['defaults']['GAIN'] if x == '' or x is None or x < 0 else x)
        self.logger.info('Cameras found: %s', unique_df['INSTRUME'].unique())
        self.logger.info('Camera gains: %s', unique_df['GAIN'].unique())
        self.logger.info('Camera EGains: %s', unique_df['EGAIN'].unique())
        self.logger.info('Replaced empty or None values with default values from configuration')

        hdrs['GAIN'] = hdrs.apply(
            lambda row: unique_df[(unique_df['EGAIN'] == row['EGAIN'])]['GAIN'].values[0]
            if row['GAIN'] <= 0 and len(unique_df[(unique_df['EGAIN'] == row['EGAIN'])]['GAIN'].values) > 0
            else row['GAIN'],
            axis=1
        )

        return hdrs


        #self.headers['GAIN'] = self.headers.apply(
        #    lambda row: unique_df[(unique_df['INSTRUME'] == row['INSTRUME']) & (unique_df['EGAIN'] == row['EGAIN'])]['GAIN'].values[0]
        #    if row['GAIN'] <= 0 and len(unique_df[(unique_df['INSTRUME'] == row['INSTRUME']) & (unique_df['EGAIN'] == row['EGAIN'])]['GAIN'].values) > 0
        #    else row['GAIN'],
        #    axis=1
        #)


        self.logger.info('Replaced values in headers with corresponding values from DataFrame')
        self.logger.info('Gain in headers: %s', self.headers['GAIN'].unique())

        return unique_df

    def get_HFR(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts the Half Flux Radius (HFR), Image Scale (IMSCALE), and Full Width at Half Maximum (FWHM) from the given DataFrame.

        This function iterates over each row in the DataFrame, and for each row, it calculates and updates the HFR, IMSCALE,
        and FWHM values. It does this by extracting the HFR from the filename, calculating the IMSCALE from the pixel size
        and focal length, and calculating the FWHM from the HFR and IMSCALE. If the HFR is not found in the filename or is
        less than or equal to 0, it uses a default value from the configuration. It then logs and prints the last calculated
        HFR, IMSCALE, and FWHM values, and returns the DataFrame.

        Args:
        - df: The DataFrame to extract the HFR, IMSCALE, and FWHM from.

        Returns:
        - The DataFrame with the HFR, IMSCALE, and FWHM extracted and updated.
        """
        self.logger.info("")
        self.logger.info("STARTING HFR EXTRACTION")
        print("\nStarting HFR Extraction")

        hfr_set = self.config['defaults']['HFR']

        for index, row in df.iterrows():
            if row['IMAGETYP'] == 'LIGHT':

                # Calculate and update HFR, IMSCALE, and FWHM values
                file_path = row['FILENAME']
                hfr_match = re.search(r'HFR_([0-9.]+)', file_path)
                hfr = float(hfr_match.group(1)) if hfr_match and float(hfr_match.group(1)) > 0 else float(hfr_set)
                focal_length = float(row['FOCALLEN'])
                pixel_size = float(row['XPIXSZ'])
                imscale = float(row['XPIXSZ']) / float(row['FOCALLEN']) * 206.265
                fwhm = hfr * imscale*2 if hfr >= 0.0 else 0.0

                df.at[index, 'HFR'] = round(hfr,2)
                df.at[index, 'IMSCALE'] = round(imscale,2)
                df.at[index, 'FWHM'] = round(fwhm,2)

                #self.logger.info(f"Updated row {index}: Focal length: {focal_length} Pixel size : {pixel_size} um HFR: {hfr:.2f}, IMSCALE: {imscale:.2f}, FWHM: {fwhm:.2f}")

        self.logger.info(f"Completed HFR extraction: mean HFR: {df['HFR'].mean():.2f}, Image Scale: {imscale:.2f}, mean FWHM: {df['FWHM'].mean():.2f}")
        print(f"\nCompleted HFR extraction. Processed {index+1} images")
        self.number_of_images_processed = index+1

        return df

class Processing():
    def __init__(self,headers,logger):
        self.headers = headers
        self.logger = logger
        self.dp = headers.dp
        self.logger.info("Processing initialised")

    def aggregate_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates parameters from the given DataFrame.

        This function standardizes column names to lower case, splits the DataFrame into two based on the 'imagetyp' column,
        aggregates the parameters of interest from each split DataFrame, concatenates the results, renames some columns for
        clarity, rounds and casts values to the correct type, and returns the aggregated DataFrame. If the input DataFrame is
        empty, it returns an empty DataFrame.

        Args:
        - df: The DataFrame to aggregate parameters from.

        Returns:
        - The DataFrame with the aggregated parameters.
        """

        def replace_numeric_with_none(value):
            if value == 'None':
                return value
            try:
                float(value)
                return 'None'
            except ValueError:
                return value


        self.logger.info("")
        self.logger.info("STARTING PARAMETER AGGREGATION")

        if not df.empty:
            # Standardizing column names to lower case
            df.columns = df.columns.str.lower()
            self.logger.info("Standardized column names to lower case")
            #convert df['date-obs'] to date

            #calculate observation statistics
            threshold=timedelta(hours=5)
            # Convert strings to datetime objects
            date_strings = df[df['imagetyp']=='LIGHT']['date-obs']
            self.logger.info("Extracted date-obs from DataFrame")
            #dates = [datetime.fromisoformat(s) for s in date_strings]

            #date_strings=[] #set empty dates for testing
            #sessions=min # set

            try:
                dates = [parse(s) for s in date_strings]
                self.logger.info(f"Parsed {len(dates)} dates.")
            except ValueError as e:
                self.logger.error(f"Error parsing dates: {e}")
                self.logger.error(f"Date strings: {date_strings}")
                return

            try:
                dates.sort()  # Ensure dates are in ascending order
                self.logger.info("Sorted dates.")
            except TypeError as e:
                self.logger.error(f"Error sorting dates: {e}")
                return
            self.logger.info("Sorted dates")

            sessions = 1  # Start with one group
            for i in range(1, len(dates)):
                if dates[i] - dates[i-1] > threshold:
                    sessions += 1  #Its a new session if the gap is more than the threshold
            self.logger.info("Calculated number of sessions")
            self.logger.info(f"Number of sessions: {sessions}")

            # Find the max and min dates and calculate the number of days between them
            default_date = pd.to_datetime('1900-01-01')



            try:
                df['sessions'] = sessions
            except Exception as e:
                self.logger.error(f"Error setting sessions: {e}")
                df['sessions'] = 0

            try:
                df['start_date'] = pd.to_datetime(min(dates).date())
            except Exception as e:
                self.logger.error(f"Error setting start date: {e}")
                df['start_date'] = default_date

            try:
                df['end_date'] = pd.to_datetime(max(dates).date())
            except Exception as e:
                self.logger.error(f"Error setting end date: {e}")
                df['end_date'] = default_date

            try:
                df['num_days'] = (max(dates).date() - min(dates).date()).days + 1
            except Exception as e:
                self.logger.error(f"Error setting number of days: {e}")
                df['num_days'] = 0

            self.logger.info("Calculated start and end dates and number of days")
            self.logger.info(f"Start date: {df['start_date'].iloc[0]}, End date: {df['end_date'].iloc[0]}, Number of days: {df['num_days'].iloc[0]}")

             # First, try parsing with the most common format (without milliseconds)
            try:
                df['date-obs'] = pd.to_datetime(df['date-obs'], format='%Y-%m-%dT%H:%M:%S')
            except ValueError:
                # If the first attempt fails, try parsing with milliseconds
                try:
                    df['date-obs'] = pd.to_datetime(df['date-obs'], format='%Y-%m-%dT%H:%M:%S.%f')
                except ValueError:
                    # As a last resort, use infer_datetime_format to handle varied formats
                    df['date-obs'] = pd.to_datetime(df['date-obs'], infer_datetime_format=True, errors='coerce')
   
            #df['date-obs'] = pd.to_datetime(df['date-obs'], format='%Y-%m-%d')
            df['date-obs'] = pd.to_datetime(df['date-obs']).dt.date


            self.logger.info("Created a date column to date")
            # Split the DataFrame into two
            #df_master = df[df['imagetyp'].str.contains('MASTER')]
            # Split the DataFrame into two
            df_master = df[df['imagetyp'].str.contains('MASTER') & ~df['imagetyp'].str.contains('MASTERLIGHT')]

            df_not_master = df[~df['imagetyp'].str.contains('MASTER')]
            self.logger.info("Split DataFrame into 'MASTER' and not 'MASTER'")

            # Define the common aggregation dictionary
            # Define the common aggregation dictionary
            agg_dict = {
                'ccd-temp': ('ccd-temp', 'mean'),
                'temp_min': ('foctemp', 'min'),
                'temp_max': ('foctemp', 'max'),
                'fwhm': ('fwhm', 'mean'),
                'sitelat': ('sitelat','mean'),
                'sitelong': ('sitelong','mean'),
                'focratio': ('focratio','mean'),
                'foctemp': ('foctemp','mean'),
                'focallen': ('focallen','first'),
                'bortle': ('bortle', 'mean'),
                'sqm': ('sqm', 'mean'),
                'hfr': ('hfr','mean'),
                'xpixsz': ('xpixsz','first'),
                'egain': ('egain','mean'),
                'instrume': ('instrume','first'),
                'imscale': ('imscale','mean'),
                'telescop': ('telescop','first'),
                'focname': ('focname','first'),
                'fwheel': ('fwheel','first'),
                'rotator': ('rotator','first'),
                'target': ('object','first'),
                'swcreate': ('swcreate','first'),
                'filename': ('filename','first'),
                'sessions': ('sessions','first'),
                'start_date': ('start_date','first'),
                'end_date': ('end_date','first'),
                'num_days': ('num_days','first')
            }

            # Aggregate the 'MASTER' DataFrame
            #agg_master_df = df_master.agg(number=('number', 'first'),**agg_dict).reset_index()
            #self.logger.info("Aggregated 'MASTER' DataFrame")

            # Aggregate the not 'MASTER' DataFrame
            agg_not_master_df = df_not_master.groupby(['site','date-obs','imagetyp', 'filter', 'gain', 'xbinning', 'exposure','object']).agg(
                number=('date-obs', 'count'),
                **agg_dict
            ).reset_index()
            self.logger.info("Aggregated none 'MASTER' DataFrame")

            if df_master.empty:
                self.logger.info("No MASTER data found")
                aggregated_df = agg_not_master_df
            else:
            # Set 'imscale' in df_master to 'imscale' in df_not_master_df
                df_master['imscale'] = df_not_master['imscale']
            # Concatenate the results
                aggregated_df = pd.concat([df_master, agg_not_master_df])
                self.logger.info("Concatenated MASTER and non-MASTER results")

            # Renaming columns for clarity
            aggregated_df.rename(columns={
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
                'exposure': 'duration',
                'focallen': 'focalLength',
                'xpixsz': 'pixelSize',
                'focratio': 'fNumber'}, inplace=True)

            self.logger.info("Renamed columns for clarity")

            #force numeric data types
            # Change data type of 'sensorCooling' column to float

            aggregated_df['sensorCooling'] = aggregated_df['sensorCooling'].astype(float).round().astype(int)
            aggregated_df['temp_min'] = aggregated_df['temp_min'].astype(float)
            aggregated_df['temp_max'] = aggregated_df['temp_max'].astype(float)
            aggregated_df['temperature'] = aggregated_df['temperature'].astype(float)
            aggregated_df['meanFwhm'] = aggregated_df['meanFwhm'].astype(float)
            aggregated_df['sitelat'] = aggregated_df['sitelat'].astype(float)
            aggregated_df['sitelong'] = aggregated_df['sitelong'].astype(float)
            aggregated_df['fNumber'] = aggregated_df['fNumber'].astype(float)
            aggregated_df['bortle'] = aggregated_df['bortle'].astype(float)
            aggregated_df['meanSqm'] = aggregated_df['meanSqm'].astype(float)
            aggregated_df['egain'] = aggregated_df['egain'].astype(float)
            aggregated_df['camera'] = aggregated_df['camera'].astype(str)
            aggregated_df['telescope'] = aggregated_df['telescope'].astype(str)
            aggregated_df['focuser'] = aggregated_df['focuser'].astype(str)
            aggregated_df['filterWheel'] = aggregated_df['filterWheel'].astype(str)
            aggregated_df['rotator'] = aggregated_df['rotator'].astype(str)
            aggregated_df['target'] = aggregated_df['target'].astype(str)
            aggregated_df['swcreate'] = aggregated_df['swcreate'].astype(str)
            aggregated_df['duration'] = aggregated_df['duration'].astype(float)
            aggregated_df['focalLength'] = aggregated_df['focalLength'].astype(float)
            aggregated_df['pixelSize'] = aggregated_df['pixelSize'].astype(float)
            aggregated_df['imageType'] = aggregated_df['imageType'].astype(str)
            try:
                aggregated_df['sessions'] = aggregated_df['sessions'].astype(int)
            except TypeError as e:
                self.logger.error(f"Error converting sessions to int: {e}")
                aggregated_df['sessions'] = 0

            try:
                aggregated_df['start_date'] = pd.to_datetime(aggregated_df['start_date']).dt.date
            except Exception as e:
                self.logger.error(f"Error converting start_date to datetime: {e}")
                aggregated_df['start_date'] = pd.to_datetime('1900-01-01')

            try:
                aggregated_df['end_date'] = pd.to_datetime(aggregated_df['end_date']).dt.date
            except Exception as e:
                self.logger.error(f"Error converting end_date to datetime: {e}")
                aggregated_df['end_date'] = pd.to_datetime('1900-01-01')

            try:
                aggregated_df['num_days'] = aggregated_df['num_days'].astype(int)
            except TypeError as e:
                self.logger.error(f"Error converting num_days to int: {e}")
                aggregated_df['num_days'] = 0


            # Round and cast values to correct type
            dp = 2  # number of decimal places
            aggregated_df = aggregated_df.round({
                'temp_min': dp,
                'temp_max': dp,
                'temperature': dp,
                'meanFwhm': dp,
                'sitelat': self.headers.dp,
                'sitelong': self.headers.dp,
                'fNumber': dp,
                'bortle': dp,
                'meanSqm': dp,
                'duration': dp
            })


            #aggregated_df['sensorCooling'] = aggregated_df['sensorCooling'].round().astype(int)
            self.logger.info("Rounded and cast values to correct type")

        else:
            aggregated_df = pd.DataFrame()
            self.logger.info("No data found")

        aggregated_df['rotator'] = aggregated_df['rotator'].apply(replace_numeric_with_none)
        self.logger.info(f"aggregated_df keys")
        for key in aggregated_df.keys():
            self.logger.info(f"aggregated_df key: {key}")
        self.logger.info("")
        self.logger.info(f"Number of observation session: {aggregated_df['sessions'].iloc[0]}")
        self.logger.info(f"start_date: {aggregated_df['start_date'].iloc[0]}")
        self.logger.info(f"end_date: {aggregated_df['end_date'].iloc[0]}")
        self.logger.info(f"num_days: {aggregated_df['num_days'].iloc[0]}")
        self.logger.info("")


        return pd.DataFrame(aggregated_df)

    def update_filter(self, filter_value: str, filter_to_code: Dict[str, str]) -> Union[int, str]:
        """
        Maps a filter name to its corresponding code based on the 'filter_to_code' dictionary.

        If the filter name exists in 'filter_to_code', this function returns the associated code.
        If the code is not a four-digit integer, a warning is logged, and a placeholder
        indicating no code found is returned.

        Args:
        - filter_value: The name of the filter to be mapped to a code.
        - filter_to_code: A dictionary mapping filter names to their corresponding codes.

        Returns:
        - The code corresponding to the filter name, or a string indicating no valid code is found.
        """
        self.logger.info("")
        self.logger.info("STARTING FILTER UPDATE")
        self.logger.info("Updating filter: %s", filter_value)

        # Check code is a five digit integer, if it is assume valid code
        code = filter_to_code.get(filter_value)
        # Try to convert the code to an integer
        try:
            code = int(code)
        except (ValueError, TypeError):
            code = None
            self.logger.warning("Failed to convert code to integer for filter: %s", filter_value)

        if isinstance(code, int) and 1000 <= code <= 99999:
            self.logger.info("Valid code found for filter: %s", filter_value)
            return code
        else:
            # Log a warning if the code is not found for a filter
            self.logger.warning("No code found for filter: %s. Enter a valid code in filters.csv file or input the code in the astrobin upload file.", filter_value)
            return f"{filter_value}: no code found"

    def create_astrobin_output(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Creates an output DataFrame for AstroBin based on the input DataFrame.

            This function groups the input DataFrame by 'site', processes each 'imagetyp' and updates the corresponding column in
            the 'light_group' DataFrame, processes 'MASTER' frames, maps 'filter' names to their codes, orders the columns, and
            returns the processed DataFrame. If the input DataFrame is empty, it returns an empty DataFrame.

            Args:
            - df: The input DataFrame to process.

            Returns:
            - The processed DataFrame for AstroBin.
            """
            self.logger.info("")
            self.logger.info("STARTING ASTROBIN OUTPUT CREATION")


            filters_dict = self.headers.config['filters']
            light_group_agg= pd.DataFrame()
            if not df.empty:
                self.logger.info("Data found, processing")

                light_group = df[df['imageType'] == 'LIGHT'].copy()  # Create a copy of the DataFrame
                self.logger.info("Created 'LIGHT' DataFrame copy")

                # Map 'imagetyp' to column name in 'light_group'
                imagetyp_to_column = {
                    'BIAS': 'bias',
                    'DARK': 'darks',
                    'DARKFLAT': 'flatDarks',
                    'FLAT': 'flats'
                }

                # Process each 'imagetyp' and update the corresponding column in 'light_group'
                for imagetyp, column in imagetyp_to_column.items():
                    self.logger.info("")
                    self.logger.info("Processing 'image type': %s", imagetyp)

                    sub_group = df[df['imageType'] == imagetyp]
                    #self.logger.info("Sub-group size: %d", len(sub_group))

                    light_group[column] = light_group.apply(lambda x: sub_group[sub_group['gain' if imagetyp != 'FLAT' else 'filter'] == x['gain' if imagetyp != 'FLAT' else 'filter']]['number'].sum(), axis=1)
                    self.logger.info("Updated 'light_group' column: %s", column)

                master_imagetyp_to_column = {
                    'MASTERBIAS': 'bias',
                    'MASTERDARK': 'darks',
                    'MASTERDARKFLAT': 'flatDarks',
                    'MASTERFLAT': 'flats'
                }

                for imagetyp, column in master_imagetyp_to_column.items():
                    self.logger.info("")
                    self.logger.info("Processing 'MASTER' frames for 'image type': %s", imagetyp)

                    sub_group = df[df['imageType'] == imagetyp]
                    if not sub_group.empty:
                        #light_group[column] += sub_group['number'].iloc[0]
                        light_group[column] = light_group.apply(lambda x: sub_group[sub_group['gain' if imagetyp != 'MASTERFLAT' else 'filter'] == x['gain' if imagetyp != 'MASTERFLAT' else 'filter']]['number'].sum(), axis=1)
                        self.logger.info("Updated 'light_group' column with 'MASTER' frame number: %s", column)


                if light_group['filter'].dtype == 'object':

                    # Get unique filters for logging and remove leading/trailing spaces from each
                    original_filters = [f.strip() for f in light_group['filter'].unique()]  # Remove leading/trailing spaces        
                    light_group['filter'] = light_group['filter'].apply(lambda x: filters_dict.get(x.strip(), x.strip()))

                    # Prepare log message
                    self.logger.info("")
                    self.logger.info("Mapped 'filter' names to codes:")
                    for original_filter in original_filters:
                        mapped_code = filters_dict.get(original_filter, original_filter)
                        self.logger.info(f"{original_filter} -> {mapped_code}")

                    #self.logger.info(log_message)

                    column_order = ['date', 'filter', 'number', 'duration', 'binning', 'gain',
                                    'sensorCooling', 'fNumber', 'darks', 'flats', 'flatDarks', 'bias', 'bortle',
                                    'meanSqm', 'meanFwhm', 'temperature']
                    self.logger.info("")
                    self.logger.info("Ordered columns to meet AstroBin requirements")
                    light_group.rename(columns={'date-obs': 'date'}, inplace=True)
                    self.logger.info("")
                    self.logger.info("COMPLETED ASTROBIN OUTPUT CREATION")

                    light_group_agg =pd.concat([light_group_agg,light_group[column_order]])



                return light_group_agg

            else:
                self.logger.info("No data found")
                return pd.DataFrame()

class Sites():
    def __init__(self, headers: Dict[str, str], logger: logging.Logger):
        """
        Initializes the Sites object.

        Parameters:
        - headers (Dict[str, str]): The headers for the API requests.
        - logger (logging.Logger): The logger to use for logging messages.

        Attributes:
        - headers (Dict[str, str]): The headers for the API requests.
        - logger (logging.Logger): The logger to use for logging messages.
        - existing_sites_df (pd.DataFrame): A DataFrame containing existing site data.
        - geolocator (geopy.geocoders.Nominatim): The geolocator to use for geolocation queries.
        """
        self.logger = logger
        self.logger.info("")
        self.logger.info("INITIALIZING SITES OBJECT")
        self.headers = headers
        email_address = self.headers.config['secret']['EMAIL_ADDRESS']

        try:
            self.existing_sites_df = pd.DataFrame(self.headers.config['sites'])
            self.logger.info("Existing sites DataFrame created")
        except:
            self.existing_sites_df = pd.DataFrame()
            self.logger.info("Failed to create existing sites DataFrame, initialized as empty DataFrame")

        # Initialize the geolocator
        self.geolocator = Nominatim(user_agent="AstroBinUpload.py_"+email_address.strip())
        self.logger.info("Geolocator initialized")

    def find_location_by_coords(self, sites_df: pd.DataFrame, lat_long_pair: Tuple[float, float]) -> Optional[pd.DataFrame]:
        """
        Finds the location in the sites DataFrame that matches the given latitude and longitude pair.

        Parameters:
        - sites_df (pd.DataFrame): The DataFrame containing site data.
        - lat_long_pair (Tuple[float, float]): The latitude and longitude pair to match.
        - n (int): The number of decimal places to consider when matching latitude and longitude.

        Returns:
        - pd.DataFrame or None: A DataFrame containing the matching rows, or None if no match is found.
        """


        target_lat, target_long = lat_long_pair
        #matching_rows = sites_df[(round(sites_df['latitude'], n) == round(target_lat, n))
        #                        & (round(sites_df['longitude'], n) == round(target_long, n))]

        matching_rows = sites_df[(np.ceil(sites_df['latitude'] * 10) / 10  == np.ceil(target_lat * 10) / 10 )
                                & (np.ceil(sites_df['longitude'] * 10) / 10 == np.ceil(target_long * 10) / 10)]
        if not matching_rows.empty:
            #self.logger.info("Matching location in sites found")
            #self.logger.info(f"Site name: {matching_rows.index[0]} Site latitude: {matching_rows['latitude'][0]} Site longitude: {matching_rows['longitude'][0]}")
            return matching_rows
        else:

            return None

    def get_bortle_sqm(self, lat: float, lon: float, api_key: str, api_endpoint: str) -> Tuple[int, float, Optional[str], bool, bool]:
        """
        Retrieves the Bortle scale classification and SQM (Sky Quality Meter) value for a given latitude and longitude.

        Parameters:
        - lat (float): The latitude coordinate.
        - lon (float): The longitude coordinate.
        - api_key (str): The API key.
        - api_endpoint (str): The API endpoint.

        Returns:
        - tuple: A tuple containing the Bortle scale classification, SQM value, error message (if any),
                and flags indicating the validity of the API key and endpoint.
        """
        self.logger.info("")
        self.logger.info("GETTING BORTLE SCALE AND SQM VALUE")
        self.logger.info(f"Retrieving Bortle scale and SQM value for coordinates ({lat}, {lon})")

        def is_valid_api_key(api_key: str) -> bool:
            """ Check if the API key is valid. """
            return api_key is not None and len(api_key) == 16 and api_key.isalnum()

        def is_valid_api_endpoint(api_endpoint: str) -> bool:
            """ Check if the API endpoint is valid. """
            return bool(api_endpoint and api_endpoint.strip())

        api_valid = is_valid_api_key(api_key)
        api_endpoint_valid = is_valid_api_endpoint(api_endpoint)

        # Check the validity of the API key and endpoint
        if not api_valid and not api_endpoint_valid:
            self.logger.error("Both API key and API endpoint are invalid.")
            return 0, 0, "Both API key and API endpoint are invalid.", api_valid, api_endpoint_valid
        elif not api_valid:
            self.logger.error("API key is malformed.")
            return 0, 0, "API key is malformed.", api_valid, api_endpoint_valid
        elif not api_endpoint_valid:
            self.logger.error("API endpoint is empty.")
            return 0, 0, "API endpoint is empty.", api_valid, api_endpoint_valid

        # Define the parameters for the API request
        params = {
            'ql': 'wa_2015',
            'qt': 'point',
            'qd': f'{lon},{lat}',
            'key': api_key
        }
        try:
            self.logger.info("Sending request to API endpoint")
            response = requests.get(api_endpoint, params=params)
            response.raise_for_status()

            if response.text.strip() == 'Invalid authentication.':
                self.logger.error("Authentication error: Missing or invalid API key.")
                return 0, 0, "Authentication error: Missing or invalid API key.", False, api_endpoint_valid

            artificial_brightness = float(response.text)
            sqm = round((math.log10((artificial_brightness + 0.171168465)/108000000)/-0.4),2)
            bortle_class = round(self.sqm_to_bortle(sqm),2)
            self.logger.info(f"Retrieved Bortle scale: {bortle_class}, SQM value: {sqm}")
            return bortle_class, sqm, None, api_valid, api_endpoint_valid

        except requests.exceptions.HTTPError as err:
            self.logger.error(f"HTTP Error: {err}")
            return 0, 0, f"HTTP Error: {err}", api_valid, False
        except ValueError:
            self.logger.error("Could not convert response to float.")
            return 0, 0, "Could not convert response to float.", api_valid, api_endpoint_valid
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request Error: {str(e)}")
            return 0, 0, f"Request Error: {str(e)}", api_valid, api_endpoint_valid
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return 0, 0, f"An error occurred: {e}", api_valid, api_endpoint_valid

    def sqm_to_bortle(self, sqm: float) -> int:
        """
        Converts an SQM (Sky Quality Meter) value to the corresponding Bortle scale classification.

        The Bortle scale is a nine-level numeric scale used to quantify the astronomical observability of celestial objects,
        affected by light pollution. The scale ranges from 1, indicating the darkest skies, to 9, the brightest.

        Args:
        - sqm (float): The SQM value indicating the level of light pollution.

        Returns:
        - int: The Bortle scale classification (ranging from 1 to 9).
        """
        self.logger.info("")
        self.logger.info("CONVERTING SQM VALUE TO BORTLE SCALE")

        self.logger.info(f"Converting SQM value {sqm} to Bortle scale")

        # Bortle scale classification based on SQM values
        if sqm > 21.99:
            self.logger.debug("SQM value corresponds to Bortle scale 1")
            return 1  # Class 1: Excellent dark-sky site
        elif 21.50 <= sqm <= 21.99:
            self.logger.debug("SQM value corresponds to Bortle scale 2")
            return 2  # Class 2: Typical truly dark site
        elif 21.25 <= sqm <= 21.49:
            self.logger.debug("SQM value corresponds to Bortle scale 3")
            return 3  # Class 3: Rural sky
        elif 20.50 <= sqm <= 21.24:
            self.logger.debug("SQM value corresponds to Bortle scale 4")
            return 4  # Class 4: Rural/suburban transition
        elif 19.50 <= sqm <= 20.49:
            self.logger.debug("SQM value corresponds to Bortle scale 5")
            return 5  # Class 5: Suburban sky
        elif 18.50 <= sqm <= 19.49:
            self.logger.debug("SQM value corresponds to Bortle scale 6")
            return 6  # Class 6: Bright suburban sky
        elif 17.50 <= sqm <= 18.49:
            self.logger.debug("SQM value corresponds to Bortle scale 7")
            return 7  # Class 7: Suburban/urban transition
        elif 17.00 <= sqm <= 17.49:
            self.logger.debug("SQM value corresponds to Bortle scale 8")
            return 8  # Class 8: City sky
        else:
            self.logger.debug("SQM value corresponds to Bortle scale 9")
            return 9  # Class 9: Inner-city sky

    def is_new_location(self, existing_sites: pd.DataFrame, lat_long_pair: Tuple[float, float]) -> bool:
        """
        Checks if the given location is new.

        Args:
        - existing_sites: DataFrame of existing sites.
        - lat_long_pair: Tuple of latitude and longitude.

        Returns:
        - True if the location is new, False otherwise.
        """

        #is_new = existing_sites.empty or existing_sites[(round(existing_sites['latitude'], 1) == round(lat_long_pair[0],1)) & (round(existing_sites['longitude'], 1) == round(lat_long_pair[1],1))].empty

        is_new = existing_sites.empty or \
            existing_sites[(np.ceil(existing_sites['latitude'] * 10) / 10 == np.ceil(lat_long_pair[0] * 10) / 10) &
                           (np.ceil(existing_sites['longitude'] * 10) / 10 == np.ceil(lat_long_pair[1] * 10) / 10)].empty

        return is_new

    def process_new_location(self, lat_long_pair: Tuple[float, float], api_key: str, api_endpoint: str) -> Tuple[str, int, float]:
        """
        Processes a new location.

        Args:
        - lat_long_pair: Tuple of latitude and longitude.
        - api_key: API key for the geolocation service.
        - api_endpoint: API endpoint for the geolocation service.

        Returns:
        - Tuple of location string, Bortle scale, and SQM value.
        """
        self.logger.info("")
        self.logger.info("PROCESSING NEW LOCATION")
        self.logger.info("Site location does not exist in existing sites. Reverse geocoding location address.")
        #initialise
        location_str = self.headers.config['defaults']['SITE']
        try:
            # Temporarily raise a GeocoderUnavailable exception for testing
            #raise GeocoderUnavailable("Test exception: GeocoderUnavailable")
            location = self.geolocator.reverse(lat_long_pair, exactly_one=True)
            if location is None:
                raise Exception("No location found for the given lat_long_pair.")
            self.logger.info(f"Using location string of {location.address}")
            self.logger.info(f"Using SITELAT of {lat_long_pair[0]} and SITELONG of {lat_long_pair[1]}")
            location_str = location.address
            bortle, sqm, _, _, _ = self.get_bortle_sqm(lat_long_pair[0], lat_long_pair[1], api_key, api_endpoint)
            self.logger.info(f"Using Bortle of {bortle} and SQM of {sqm}")
            #check if bortle and sqm are 0 if so use default values
            if bortle == 0 and sqm == 0:
                bortle = self.headers.config['defaults']['BORTLE']
                sqm = self.headers.config['defaults']['SQM']
                self.logger.warning(f"Bortle and SQM values returned from API are 0. Using default values of: Bortle {bortle} SQM {sqm} from config.")
            self.logger.info(f"Processed new location: {location_str}, Bortle: {bortle}, SQM: {sqm}")
        except Exception as e:
            #for any geocoding error use default values
            self.logger.warning(f"Exception: {str(e)}. Using default site information from config['defaults'].")
            self.logger.info(f"Using location string of {self.headers.config['defaults']['SITE']}, Bortle of {self.headers.config['defaults']['BORTLE']}, and SQM of {self.headers.config['defaults']['SQM']}")
            self.logger.info(f"Using SITELAT of {self.headers.config['defaults']['SITELAT']} and SITELONG of {self.headers.config['defaults']['SITELONG']}")
            location_str = self.headers.config['defaults']['SITE']
            lat_long_pair = (round(self.headers.config['defaults']['SITELAT'], self.headers.dp), round(self.headers.config['defaults']['SITELONG'], self.headers.dp))
            bortle = self.headers.config['defaults']['BORTLE']
            sqm = self.headers.config['defaults']['SQM']

        return lat_long_pair,location_str, bortle, sqm

    def save_new_location(self, location_str: str, bortle: int, sqm: float, lat_long_pair: Tuple[float, float]) -> None:
        """
        Saves a new location.

        Args:
        - location_str: Location string.
        - bortle: Bortle scale.
        - sqm: SQM value.
        - lat_long_pair: Tuple of latitude and longitude.
        """
        #self.logger.info("")
        #self.logger.info("SAVING NEW LOCATION")

        keys_values = {'latitude': lat_long_pair[0], 'longitude': lat_long_pair[1], 'bortle': bortle, 'sqm': sqm}
        self.headers.update_config('sites', location_str, keys_values)
        self.logger.info(f"Saved new site: {location_str} with Latitude {lat_long_pair[0]}, Longitude {lat_long_pair[1]}, Bortle: {bortle}, SQM: {sqm}")

    def get_site_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieves site data for each row in the DataFrame. If the location is new, it processes and saves the new location.
        If the location has been seen before, it retrieves the location information. It then updates the DataFrame with the
        site, Bortle scale, and mean SQM values.

        Args:
        - df: The input DataFrame to process.

        Returns:
        - The processed DataFrame with updated site, Bortle scale, and mean SQM values.
        """
        self.logger.info("")
        self.logger.info("GETTING SITE DATA")
        self.logger.info("Starting site data retrieval")
        new = False

        seen_locations = {}
        api_key = self.headers.config['secret'].keys()[0].strip()
        api_endpoint = self.headers.config['secret'].values()[0].strip()
        self.logger.info(f"Retrieved API key and endpoint from config.ini: {api_key}, {api_endpoint}")

        if df.empty:
            self.logger.error("Error: No images found in session. Terminating program.")
            sys.exit("Error: No images found in session. Terminating program.")

        est = self.existing_sites_df.transpose()


        self.logger.info("")
        self.logger.info("CHECKING IF LOCATIONS ARE NEW")
        for index, row in df.iterrows():

            lat_long_pair = (round(row['SITELAT'], self.headers.dp), round(row['SITELONG'], self.headers.dp))
            r_lat_long_pair = (np.ceil(row['SITELAT'] * 10) / 10, np.ceil(row['SITELONG'] * 10) / 10)


            if self.is_new_location(est, lat_long_pair):

                self.logger.info(f"New location not in default[sites] found: {lat_long_pair}")

                lat_long_pair_return,location_str, bortle, sqm = self.process_new_location(lat_long_pair, api_key, api_endpoint)

                if lat_long_pair_return != lat_long_pair:
                    self.logger.warning("Default site used instead of given lat_long_pair.")
                    row['SITELAT'] = lat_long_pair_return[0]
                    row['SITELONG'] = lat_long_pair_return[1]
                    lat_long_pair = lat_long_pair_return

                self.logger.info("Location not seen in config.ini : %s, Bortle: %s, SQM: %s", location_str, bortle, sqm)

                self.save_new_location(location_str, bortle, sqm, lat_long_pair)

                self.existing_sites_df = pd.DataFrame(self.headers.config['sites'])
                est = self.existing_sites_df.transpose()  # Refresh after saving new location
                self.logger.info("Refresh existing locations in code after saving: %s", est.to_dict())
                self.logger.info("")
                site = location_str
            else:

                if r_lat_long_pair not in seen_locations:

                    seen_locations[r_lat_long_pair] = r_lat_long_pair
                    self.logger.info("")
                    self.logger.info("New location seen during run but exists in config.ini : %s", lat_long_pair)
                    self.logger.info(f"Seen locations: {seen_locations}")

            self.location_info = self.find_location_by_coords(est, lat_long_pair)

            if self.location_info is None:
                self.logger.error(f"No location found in ['sites'] for lat_long_pair: {lat_long_pair}")
                self.logger.info("Using default site information from config['defaults'].")
                site = self.headers.config['defaults']['SITE']
                bortle = self.headers.config['defaults']['BORTLE']
                sqm = self.headers.config['defaults']['SQM']
            else:
                site = self.location_info.index[0]
                bortle = self.location_info['bortle'].iloc[0]
                sqm = self.location_info['sqm'].iloc[0]
                #self.logger.info("Existing location found: Site: %s, Bortle: %s, SQM: %s", site, bortle, sqm)

            df.at[index, 'BORTLE'] = bortle
            df.at[index, 'SQM'] = sqm
            df.at[index, 'SITE'] = site

        self.logger.info("")
        self.logger.info("Completed site data retrieval")
        return df
