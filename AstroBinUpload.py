#!/usr/bin/env python3

import os
import sys
import argparse
from config_functions import initialise_config,config_version
from headers_functions import initialize_headers, process_directories
from processing_functions import initialize_processing, aggregate_parameters, create_astrobin_output
from sites_functions import initialize_sites, get_site_data
from utils import initialise_logging, summarize_session, utils_version

version = '1.4.1'

# Determine the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# CONFIGFILENAME should only exist in the directory where the script is located
CONFIGFILENAME = os.path.join(script_dir, 'config.ini')

PRECISION = 4
DEBUG = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process FITS files and create AstroBin acquisition CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s "/path/to/images"                    # Process single directory
  %(prog)s "/path/to/images" "/path/to/cals"   # Process multiple directories
  %(prog)s . --debug                           # Process current directory with debug output"""
    )
    
    parser.add_argument('directories', nargs='*', 
                       help='Directory paths to process (use "." for current directory)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output and save intermediate CSV files')
    parser.add_argument('--version', action='version', version=f'%(prog)s {version}')
    
    args = parser.parse_args()
    
    # Handle no arguments case for config initialization
    if not args.directories:
        return args, []
    
    # Handle current directory shortcut
    if args.directories == ['.']:
        args.directories = [os.getcwd()]
    
    return args, args.directories

def main() -> None:
    """
    Main function to process directories containing FITS files, aggregate parameters,
    get site data, summarize the session, and export the summary and AstroBin data.
    """
    # Parse command line arguments
    args, directory_paths = parse_arguments()
    DEBUG = args.debug
    
    # Handle config initialization when no directories provided
    if not directory_paths:
        # Initialize config and exit
        try:
            config, change = initialise_config(CONFIGFILENAME, None)
            if change:
                print("A new config.ini file was created. Please edit this before re-running the script.")
                sys.exit(0)
            else:
                print("Error: No directory paths provided. Use --help for usage information.")
                sys.exit(1)
        except Exception as e:
            print(f"Error initializing configuration: {str(e)}")
            sys.exit(1)
    
    # Output directory is the first argument
    output_dir = directory_paths[0]

    # Ensure the output directory path is absolute
    output_dir = os.path.abspath(output_dir)

    # Construct the path for the new directory
    output_dir_path = os.path.join(output_dir, 'AstroBinUploadInfo')

    # Create the output directory if it does not exist
    try:
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"Output directory: {output_dir_path}")
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        sys.exit(1)

    # Set LOGFILENAME based on the output directory
    LOGFILENAME = os.path.join(output_dir_path, 'AstroBinUploader.log')

    # Create the directory for the log file
    try:
        os.makedirs(os.path.dirname(LOGFILENAME), exist_ok=True)
    except Exception as e:
        print(f"Error creating log directory: {str(e)}")
        sys.exit(1)

    # Initialize logging
    try:
        logger = initialise_logging(LOGFILENAME)
        logger.info("Logging initialized.")
        print("Logging initialized.")
        logger.info(f"main version: {version}")
        logger.info(f"utils version: {utils_version}")
        print(f"main version: {version}")
        print(f"utils version: {utils_version}")
        logger.info(f"Calling function and arguments provided: {' '.join([sys.argv[0]] + directory_paths + (['--debug'] if DEBUG else []))}")
        logger.info("")
    except Exception as e:
        print(f"Error initializing logging: {str(e)}")
        sys.exit(1)

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

    # Check version compatibility
    if utils_version != version:
        err = f"Error: utils version is {utils_version} and must be {version}"
        logger.error(err)
        print(err)
        logger.error("Exiting the program.")
        sys.exit(1)

    # Initialize config
    try:
        config, change = initialise_config(CONFIGFILENAME, logger)
        if change:
            print("A new config.ini file was created. Please edit this before re-running the script.")
            logger.info("New config.ini created. Exiting.")
            sys.exit(0)
    except Exception as e:
        logger.error(f"Error initializing configuration: {str(e)}")
        print(f"Error initializing configuration: {str(e)}")
        sys.exit(1)

    # Initialize headers state
    try:
        headers_state = initialize_headers(config, logger, PRECISION)
        logger.info("Headers state initialized")
        print("Headers state initialized")
    except Exception as e:
        logger.error(f"Error initializing headers state: {str(e)}")
        print(f"Error initializing headers state: {str(e)}")
        sys.exit(1)

    # Initialize processing state
    try:
        processing_state = initialize_processing(headers_state, logger)
        logger.info("Processing state initialized")
        print("Processing state initialized")
    except Exception as e:
        logger.error(f"Error initializing processing state: {str(e)}")
        print(f"Error initializing processing state: {str(e)}")
        sys.exit(1)

    # Initialize sites state
    try:
        sites_state = initialize_sites(headers_state, logger)
        logger.info("Sites state initialized")
        print("Sites state initialized")
    except Exception as e:
        logger.error(f"Error initializing sites state: {str(e)}")
        print(f"Error initializing sites state: {str(e)}")
        sys.exit(1)

    # Extract FITS headers
    logger.info("Reading FITS headers...")
    print('\nReading FITS headers...\n')
    try:
        headers_df, basic_headers = process_directories(directory_paths, headers_state)
    except Exception as e:
        logger.error(f"Error processing directories: {str(e)}")
        print(f"Error processing directories: {str(e)}")
        sys.exit(1)

    if DEBUG:
        try:
            basic_headers_csv = os.path.join(output_dir_path, os.path.basename(directory_paths[0]).replace(" ", "_") + "_basic_headers.csv")
            basic_headers.to_csv(basic_headers_csv, index=False)
            headers_csv = os.path.join(output_dir_path, os.path.basename(directory_paths[0]).replace(" ", "_") + "_headers.csv")
            headers_df.to_csv(headers_csv, index=False)
            logger.info("Debugging is True.")
            logger.info(f"Basic headers exported to {basic_headers_csv}")
            logger.info(f"Headers exported to {headers_csv}")
        except Exception as e:
            logger.error(f"Error saving debug CSV files: {str(e)}")
            print(f"Error saving debug CSV files: {str(e)}")

    # Get location information
    logger.info("Getting site data...")
    try:
        site_modified_headers_df = get_site_data(headers_df, sites_state)
    except Exception as e:
        logger.error(f"Error getting site data: {str(e)}")
        print(f"Error getting site data: {str(e)}")
        sys.exit(1)

    if DEBUG:
        try:
            modified_csv = os.path.join(output_dir_path, os.path.basename(directory_paths[0]).replace(" ", "_") + "_modified.csv")
            site_modified_headers_df.to_csv(modified_csv, index=False)
            logger.info(f"Site modified headers exported to {modified_csv}")
        except Exception as e:
            logger.error(f"Error saving modified headers CSV: {str(e)}")
            print(f"Error saving modified headers CSV: {str(e)}")

    # Aggregate parameters
    logger.info("Aggregating parameters...")
    try:
        aggregated_df = aggregate_parameters(site_modified_headers_df, processing_state)
    except Exception as e:
        logger.error(f"Error aggregating parameters: {str(e)}")
        print(f"Error aggregating parameters: {str(e)}")
        sys.exit(1)

    if DEBUG:
        try:
            aggregated_csv = os.path.join(output_dir_path, os.path.basename(directory_paths[0]).replace(" ", "_") + "_aggregated.csv")
            aggregated_df.to_csv(aggregated_csv, index=False)
            logger.info(f"Aggregated headers exported to {aggregated_csv}")
        except Exception as e:
            logger.error(f"Error saving aggregated CSV: {str(e)}")
            print(f"Error saving aggregated CSV: {str(e)}")

    # Create and print summary of observation session
    logger.info("Summarizing session...")
    try:
        summary_txt = summarize_session(aggregated_df, logger, headers_state['number_of_images_processed'])
    except Exception as e:
        logger.error(f"Error summarizing session: {str(e)}")
        print(f"Error summarizing session: {str(e)}")
        sys.exit(1)

    # Transform data for AstroBin output
    logger.info("Creating AstroBin output...")
    try:
        output_csv = os.path.basename(directory_paths[0]).replace(" ", "_") + "_acquisition.csv"
        astrobin_df = create_astrobin_output(aggregated_df, processing_state)
        summary_txt += f"\n {output_csv}"
        summary_txt += f"\n\n {astrobin_df.to_string(index=False).replace('\n', '\n ')}\n "
    except Exception as e:
        logger.error(f"Error creating AstroBin output: {str(e)}")
        print(f"Error creating AstroBin output: {str(e)}")
        sys.exit(1)

    # Export summary to a text file
    logger.info("Exporting summary to text file...")
    summary_file = os.path.join(output_dir_path, os.path.basename(directory_paths[0]).replace(" ", "_") + "_session_summary.txt")
    try:
        with open(summary_file, 'w', encoding='utf-8') as file:
            file.write(summary_txt)
        summary_txt += f"\n Processing summary exported to {summary_file}\n"
    except Exception as e:
        logger.error(f"Error exporting summary to text file: {str(e)}")
        print(f"Error exporting summary to text file: {str(e)}")

    # Export final data to CSV
    if not astrobin_df.empty:
        logger.info("\nExporting AstroBin data to CSV...")
        try:
            output_csv = os.path.join(output_dir_path, os.path.basename(directory_paths[0]).replace(" ", "_") + "_acquisition.csv")
            astrobin_df.to_csv(output_csv, index=False)
            summary_txt += f"\n AstroBin data exported to {output_csv}\n"
        except Exception as e:
            logger.error(f"Error exporting AstroBin data to CSV: {str(e)}")
            print(f"Error exporting AstroBin data to CSV: {str(e)}")
    else:
        logger.info("No AstroBin data to export.")
        print("\n No AstroBin data to export.\n")

    logger.info("Processing completed.")
    summary_txt += "\n Processing complete.\n"
    print(summary_txt)

if __name__ == "__main__":
    main()