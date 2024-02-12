import os
from utils import initialise_logging,summarize_session, Headers,Processing,Sites,utils_version
import sys

version = '1.3.0'
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

#see utils.py for change details



LOGFILENAME = 'AstroBinUploader.log'
CONFIGFILENAME = 'config.ini'
PRECISION = 4
DEBUG = False
def main() -> None:
    """
    Main function to process directories containing FITS files, aggregate parameters,
    get site data, summarize the session, and export the summary and AstroBin data.
    """
    # Initialise logging
    logger = initialise_logging(LOGFILENAME)
    logger.info("Logging initialised.")
    #log the version of the script
    logger.info(f"main version: {version}")
    #log the version of utils
    logger.info(f"utils version: {utils_version}")

    #check version of utils script is same version as this script
    if utils_version != version:
        print(f"Error: utils version is {utils_version} and must be {version}")
        print("Exiting the program.")
        logger.error(f"utils version is {utils_version} and must be {version}")
        logger.error("Exiting the program.")
        sys.exit(1)

    # Create headers object
    headers = Headers(CONFIGFILENAME, logger, PRECISION)
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

    #log the provided arguments
    logger.info(f"Calling function and arguments provided: {sys.argv}")

    #check for debug flag in arguments
    DEBUG = '--debug' in sys.argv
    #remove the debug flag from the arguments
    if DEBUG:
        sys.argv.remove('--debug')

    # Validate directory paths
    if len(sys.argv) < 2 and os.path.isfile(CONFIGFILENAME):
        logger.error("No directory path provided. Exiting the program.")
        print("Error: Please provide at least one directory path.")
        print("Exiting the program.")
        sys.exit(1)

    directory_paths = sys.argv[1:]

    # Validate directory paths
    for directory in directory_paths:
        if not os.path.exists(directory):
            logger.error(f"The path '{directory}' does not exist. Exiting the program.")
            print(f"Error: The path '{directory}' does not exist.")
            sys.exit(1)
        elif not os.path.isdir(directory):
            logger.error(f"The path '{directory}' is not a directory. Exiting the program.")
            print(f"Error: The path '{directory}' is not a directory.Exiting the program.")
            sys.exit(1)
        logger.info(f"Processing directory: {directory}")

    # Extract FITS headers
    logger.info("Reading FITS headers...")
    print('\nReading FITS headers...\n')
    headers_df,basic_headers = headers.process_directories(directory_paths)
    
    # Get location information
    logger.info("Getting site data...")
    site_modified_headers_df = sites.get_site_data(headers_df)
  
    # Aggregate parameters
    logger.info("Aggregating parameters...")
    aggregated_df = processing.aggregate_parameters(site_modified_headers_df)

    # Create and print summary of observation session
    logger.info("Summarizing session...")
    summary_txt = summarize_session(aggregated_df, logger,headers.number_of_images_processed)

    if DEBUG:
        #if debug is True save dataframes
        #save basic headers to a csv file
        basic_headers_csv = os.path.basename(directory_paths[0]).replace(" ", "_") + "_basic_headers.csv"
        basic_headers.to_csv(basic_headers_csv, index=False)
        #save modied headers to a csv file
        headers_csv = os.path.basename(directory_paths[0]).replace(" ", "_") + "_headers.csv"
        headers_df.to_csv(headers_csv, index=False)
        #save site modified headers to a csv file
        modified_csv = os.path.basename(directory_paths[0]).replace(" ", "_") + "_modified.csv"
        site_modified_headers_df.to_csv(modified_csv, index=False)
        #save the aggregated data to a csv file
        aggregated_csv = os.path.basename(directory_paths[0]).replace(" ", "_") + "_aggregated.csv"
        aggregated_df.to_csv(aggregated_csv, index=False)
        logger.info("Debugging is True.")
        logger.info(f"basic headers exported to {basic_headers_csv}")
        logger.info(f"headers exported to {headers_csv}")
        logger.info(f"site modified headers exported to {modified_csv}")
        logger.info(f"aggregated headers exported to {aggregated_csv}")


    # Transform data for AstroBin output
    logger.info("Creating AstroBin output...")
    output_csv = os.path.basename(directory_paths[0]).replace(" ", "_") + "_acquisition.csv"
    astrobin_df = processing.create_astrobin_output(aggregated_df)
    summary_txt += "\n "+output_csv
    summary_txt+= "\n\n "+ astrobin_df.to_string(index=False).replace('\n', '\n ') + "\n "
    
    # Export summary to a text file
    logger.info("Exporting summary to text file...")
    summary_file = os.path.basename(directory_paths[0]).replace(" ", "_") + "_session_summary.txt"
    with open(summary_file, 'w') as file:
        file.write(summary_txt)
    summary_txt+=f"\n Processing summary exported to {summary_file}\n"

    # Export final data to CSV
    if not astrobin_df.empty:
        logger.info("\nExporting AstroBin data to CSV...")
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
