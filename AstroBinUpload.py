#!/usr/bin/env python3

import os
from utils import initialise_logging,summarize_session, Headers,Processing,Sites,utils_version
import sys

version = '1.3.11'
'''
# Changes:
# v1.1.2 9th January 2024
# Allows initialisation of the script without a config.ini file and no directory paths
# Explicitly prevents single files being passed as arguments
# Added utils_version check
# v1.1.3 10th January 2024
# Deal with WBPP
# v1.1.3 13th January 2024
# Redesign of how code parses data files
# v1.1.4 14th January 2024
# New utils.py script v1.1.4
# 22nd January 2024
# Saved as version 1.2.0 this is now a candidate release version
# New utils.py script v1.2.0
# 28th January 2024
# Saved as version 1.2.1
# New utils.py script v1.2.1
# 29th January 2024
# Saved as version 1.2.2
# New utils.py script v1.2.2
# 30th January 2024
# Saved as version 1.2.3
# New utils.py script v1.2.3
# 6th February 2024
# Saved as version 1.2.4
# New utils.py script v1.2.4
# 7th February 2024
# Saved as version 1.2.5
# New utils.py script v1.2.5
# 7th February 2024
# Saved as version 1.2.6
# New utils.py script v1.2.6
# Saved as version 1.2.7
#added logging of data frames
# New utils.py script v1.2.8
# Saved as version 1.2.8
# added logging of data frames
# added logging of basic header data frames
# New utils.py script v1.2.8
# 8th February 2024
# New utils.py script v1.2.9
# Saved as version 1.2.9
# added DEBUG switch to save dataframes to csv files
# use --debug in command line to save dataframes to csv files
# 9th February 2024
# New utils.py script v1.2.10
# Saved as version 1.2.10
# 12th February 2024
# Release version 1.3.0
# 26th February 2024
# Release version 1.3.1
# Modified debugging file dumps, so they occour after the data has been processed not all at the the end
# New utils.py script v1.3.1
# Release version 1.3.2
# New utils.py script v1.3.2
# 4th March 2024
# Release version 1.3.3
# New utils.py script v1.3.3
# 5th March 2024
# Release version 1.3.4
# New utils.py script v1.3.4
# 5th May 2024
# Release version 1.3.5
# New utils.py script v1.3.5
# Added the ability to call the script with no arguments and it will process the current directory
# All debug, txt and csv output files are saved to a subdirectory of the directory being processed
# 16 June 2024
# Release version 1.3.6
# New utils.py script v1.3.6
# Added the ability to call the script with --debug flag to save dataframes to csv files
# Corrected error caused by site coordinates and date format of some images
# modified script to save all output files to a subdirectory of the directory being processed
# Release version 1.3.7
# allows the scipt to be called from an image directory and process the images in that directory but allows for calibration directories to be passed as arguments
# eg AstroBinUploadv1_3_7.py "." /home/user/images/calibration
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
# Deals with the case where the filter names in the light frames have trailing white spaces.
# version 1.3.11
# 16th October 2024
# Fixes bug where running the script for the first time from the installation directory would fail
# Deals with the case where the filter names in the light frames have trailing white spaces.
# Modifed script to take a new default parameter for the config file
# The parameter is USEOBSDATE  and is set to True if the actual date of the observation session is to be used when aggregating data
# for the astrobin upload .csv output. 
# If this prameter is set to False then the date the observation session was started is used. 
# Deals with the case where the filter names in the light frames have trailing white spaces.

'''

#see utils.py for change details

# Determine the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# CONFIGFILENAME should only exist in the directory where the script is located
CONFIGFILENAME = os.path.join(script_dir, 'config.ini')

PRECISION = 4
DEBUG = False

def main() -> None:
    """
    Main function to process directories containing FITS files, aggregate parameters,
    get site data, summarize the session, and export the summary and AstroBin data.
    """
    #Check for debug flag in arguments and set debug flag if found
    DEBUG = '--debug' in sys.argv
    #remove the debug flag from the arguments
    if DEBUG:
        sys.argv.remove('--debug')

    # Validate directory paths
    # bug fix for running the script for the first time from the installation directory here
    if len(sys.argv) < 2: # and os.path.isfile(CONFIGFILENAME):
            err = "No directory path provided. Please provide a directory path as an argument."
            #logger.error(err)
            print(err)
            sys.exit(1)
    elif len(sys.argv) >= 2 and sys.argv[1] == ".":
        directory_paths = [os.getcwd()]+ sys.argv[2:]
    else:
        directory_paths = sys.argv[1:]

    #Output directory is the first argument
    output_dir = sys.argv[1]

    # Ensure the output directory path is absolute
    output_dir = os.path.abspath(output_dir)

    # Construct the path for the new directory
    output_dir_path = os.path.join(output_dir, 'AstroBinUploadInfo')

    # Create the output directory called 'AstroBinUploadInfo' if it does not exist in output_dir
    os.makedirs(output_dir_path, exist_ok=True)
    print(f"Output directory: {output_dir_path}")

    # Set LOGFILENAME based on the output directory
    LOGFILENAME = os.path.join(output_dir_path, 'AstroBinUploader.log')

    # Create the directory for the log file if it does not exist
    os.makedirs(os.path.dirname(LOGFILENAME), exist_ok=True)

    # Initialise logging
    logger = initialise_logging(LOGFILENAME)
    logger.info("Logging initialised.")
    print("Logging initialised.")
    #log the version of the script
    logger.info(f"main version : {version}")
    #log the version of utils
    logger.info(f"utils version: {utils_version}")

    print(f"main version : {version}")
    print(f"utils version: {utils_version}")

    #log the provided arguments
    logger.info(f"Calling function and arguments provided: {sys.argv}")
    logger.info("")

    # Validate directory paths
    for directory in directory_paths:
        if not os.path.exists(directory):
            err = f"The path '{directory}' does not exist. Exiting the program."
            logger.error(err)
            print(err)
            sys.exit(1)
        elif not os.path.isdir(directory):
            err = f"The path '{directory}' is not a directory. Exiting the program."
            logger.error(err)
            print(err)
            sys.exit(1)
        logger.info(f"Processing directory: {directory}")


    #check version of utils script is same version as this script
    if utils_version != version:
        print(f"Error: utils version is {utils_version} and must be {version}")
        print("Exiting the program.")
        logger.error(f"utils version is {utils_version} and must be {version}")
        logger.error("Exiting the program.")
        sys.exit(1)
    

    # Create headers object
    headers = Headers(CONFIGFILENAME, logger, PRECISION)
    print("Headers object created.")
    logger.info("Headers object created.")

    # Check if a configuration file was created
    if headers.change:
        #  inform user that a new config.ini file was created and exit
        print ("A new config.ini file was created. Please edit this before re-running the script: ")
        sys.exit(0)

    # Create processing object
    processing = Processing(headers, logger)
    logger.info("Processing object created.")

    # Create sites object
    sites = Sites(headers, logger)
    logger.info("Sites object created.")

    # Extract FITS headers
    logger.info("Reading FITS headers...")
    print('\nReading FITS headers...\n')
    headers_df,basic_headers = headers.process_directories(directory_paths)

    if DEBUG:
        #if debug is True save dataframes
        #save basic headers to a csv file
        basic_headers_csv = os.path.join(output_dir_path, os.path.basename(directory_paths[0]).replace(" ", "_") + "_basic_headers.csv")
        basic_headers.to_csv(basic_headers_csv, index=False)
        #save modied headers to a csv file
        headers_csv = os.path.join(output_dir_path, os.path.basename(directory_paths[0]).replace(" ", "_") + "_headers.csv")
        headers_df.to_csv(headers_csv, index=False)
        logger.info("Debugging is True.")
        logger.info(f"basic headers exported to {basic_headers_csv}")
        logger.info(f"headers exported to {headers_csv}")

    # Get location information
    logger.info("Getting site data...")
    site_modified_headers_df = sites.get_site_data(headers_df)

    if DEBUG:
        modified_csv = os.path.join(output_dir_path, os.path.basename(directory_paths[0]).replace(" ", "_") + "_modified.csv")
        site_modified_headers_df.to_csv(modified_csv, index=False)
        logger.info(f"site modified headers exported to {modified_csv}")

    # Aggregate parameters
    logger.info("Aggregating parameters...")
    aggregated_df = processing.aggregate_parameters(site_modified_headers_df)

    if DEBUG:
        # Save the aggregated data to a csv file
        aggregated_csv = os.path.join(output_dir_path, os.path.basename(directory_paths[0]).replace(" ", "_") + "_aggregated.csv")
        aggregated_df.to_csv(aggregated_csv, index=False)
        logger.info(f"aggregated headers exported to {aggregated_csv}")

    # Create and print summary of observation session
    logger.info("Summarizing session...")
    summary_txt = summarize_session(aggregated_df, logger,headers.number_of_images_processed)

    # Transform data for AstroBin output
    logger.info("Creating AstroBin output...")
    output_csv = os.path.basename(directory_paths[0]).replace(" ", "_") + "_acquisition.csv"
    astrobin_df = processing.create_astrobin_output(aggregated_df)
    summary_txt += "\n "+output_csv
    summary_txt+= "\n\n "+ astrobin_df.to_string(index=False).replace('\n', '\n ') + "\n "

    # Export summary to a text file
    logger.info("Exporting summary to text file...")
    summary_file = os.path.join(output_dir_path, os.path.basename(directory_paths[0]).replace(" ", "_") + "_session_summary.txt")
    with open(summary_file, 'w') as file:
        file.write(summary_txt)
    summary_txt+=f"\n Processing summary exported to {summary_file}\n"

    # Export final data to CSV
    # Export final data to CSV
    if not astrobin_df.empty:
        logger.info("\nExporting AstroBin data to CSV...")
        output_csv = os.path.join(output_dir_path, os.path.basename(directory_paths[0]).replace(" ", "_") + "_acquisition.csv")
        astrobin_df.to_csv(output_csv, index=False)
        summary_txt+=f"\n AstroBin data exported to {output_csv}\n"
    else:
        logger.info("No AstroBin data to export.")
        print("\n No AstroBin data to export.\n")

    logger.info("Processing completed.")
    summary_txt+="\n Processing complete.\n"

    print(summary_txt)
    return

if __name__ == "__main__":
    main()