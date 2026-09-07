#!/usr/bin/env python3
"""
AstroBin Upload Utility v2.1.0

This is the primary entry point for the application. It orchestrates the 
entire ETL (Extract, Transform, Load) workflow using a modern Pipeline 
Architecture.

The utility scans directories for FITS and XISF files, extracts their 
metadata, normalizes hardware naming variations, matches calibration frames, 
and generates a finalized acquisition report compatible with AstroBin's 
bulk upload system.

Architecture:
- **Modular Steps**: Each logical operation is isolated in the 'engine/steps' directory.
- **Typed State**: The 'SessionState' object flows through the pipeline, carrying 
  the data between steps.
- **Vectorized Logic**: Leveraging Pandas for high-performance data manipulation 
  suitable for large datasets (1000+ files).

Usage:
    python3 AstroBinUpload.py [directories] [--test csv_file] [--debug]
"""

import argparse
import logging
import os
import sys
from engine.loader import ConfigLoader
from engine.extractor import HeaderExtractor
from engine.processor import PipelineProcessor
from engine.steps.base import NormalizeHeadersStep
from engine.steps.optical import OpticalParameterStep
from engine.steps.deduplicate import DeduplicateStep
from engine.steps.calibration import CalibrationMatcherStep
from engine.steps.geocode import GeocodeStep
from engine.steps.aggregate import AggregationStep
from engine.exporter import Exporter
from models import SessionState
from _version import __version__ as APP_VERSION

def initialise_logging(log_filename: str) -> logging.Logger:
    """
    Initializes a professional logging system with automatic context resolution.
    """
    try:
        log_dir = os.path.dirname(log_filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write('')

        new_logger = logging.getLogger("AstroBinV2")
        new_logger.handlers.clear()
        new_logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_filename, encoding='utf-8')
        # %(funcName)s is stdlib logging's own resolution of the immediate
        # caller (via the stack frame active at record creation), which is
        # exactly what the previous FunctionNameFilter walked inspect.stack()
        # by hand to reproduce on every single log record -- every
        # logger.X() call in this codebase is made directly inside the
        # function of interest (never through an intermediate wrapper), so
        # the two report the same name, without the per-record stack walk
        # (B4 in REMEDIATION_PLAN.md). %(lineno)d was already correct
        # before this change -- it's captured by Logger.makeRecord() at
        # record-creation time, before any Filter runs, so it was never
        # actually affected by the custom filter.
        formatter = logging.Formatter(
            '%(asctime)s - %(funcName)s - Line: %(lineno)d - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        new_logger.addHandler(handler)

        new_logger.info("Logging system initialized successfully.")
        return new_logger

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize logging: {e}")
        return logging.getLogger()

def main():
    """
    Main execution loop.
    
    Orchestrates the environment setup, data discovery, pipeline 
    configuration, and the final export of reports.
    """
    # Define and parse CLI arguments
    parser = argparse.ArgumentParser(
        description="AstroBin Upload Utility v2.1.0 - A high-performance ETL pipeline for astronomical metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example Usage:
        python3 AstroBinUpload.py /path/to/my/images
        python3 AstroBinUpload.py /path/to/my/images /path/to/my/calibrationfiles
        python3 AstroBinUpload.py /images /calibration_dir --debug
        python3 AstroBinUpload.py . --test my_headers.csv
        """
    )
    parser.add_argument(
        'directory_paths', 
        nargs='+', 
        help='One or more directory paths to recursively scan for FITS (.fits, .fit, .fts) or XISF (.xisf) files.'
    )
    parser.add_argument(
        '--test', 
        type=str, 
        metavar='CSV_FILE',
        help='Diagnostic Mode: Instead of scanning disk, inject metadata from a pre-processed CSV file. The CSV must reside in the first directory path provided.'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable verbose debug logging and preserve intermediate dataframes for troubleshooting.'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.ini',
        metavar='CONFIG_FILE',
        help='Specify a custom configuration file (default: config.ini).'
    )
    args = parser.parse_args()

    # --- Step 1: Environment Setup ---

    # Resolve absolute paths to ensure reliable file access
    directory_paths = [os.path.abspath(os.path.expanduser(p)) for p in args.directory_paths]

    # B10 in REMEDIATION_PLAN.md: previously unvalidated. A typo'd path
    # would reach os.makedirs(output_dir, exist_ok=True) below, which
    # creates every missing intermediate directory including the typo'd
    # one itself -- silently manufacturing a new, empty directory tree
    # rather than failing, and the pipeline would then proceed to scan it,
    # producing a "0 images processed" result with no indication of the
    # actual mistake. Fail clearly here instead.
    invalid_paths = [p for p in directory_paths if not os.path.isdir(p)]
    if invalid_paths:
        for p in invalid_paths:
            print(f"[ERROR] Not a directory: {p}")
        print(
            "\nOne or more input paths do not exist or are not directories. "
            "Check for typos before re-running."
        )
        sys.exit(1)

    # Establish the primary output directory inside the first target path
    output_dir = os.path.join(directory_paths[0], 'AstroBinUploadInfo')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the centralized logging system
    log_file = os.path.join(output_dir, 'AstroBinUploader.log')
    logger = initialise_logging(log_file)

    logger.info("Logging initialized.")
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Legacy-compliant console boot sequence for user feedback
    logger.info(f"main version: {APP_VERSION}")
    logger.info(f"utils version: {APP_VERSION}")
    logger.info(f"Calling function and arguments provided: {sys.argv}")
    logger.info("")

    print(f"Output directory: {output_dir}")
    print("Logging initialized.")
    print(f"main version: {APP_VERSION}")
    print(f"utils version: {APP_VERSION}")

    try:
        # --- Step 2: Configuration & Data Loading ---
        
        # Load and normalize config.ini into a strongly-typed AppConfig object
        loader = ConfigLoader(logger)
        config = loader.load(args.config)

        # Metadata Discovery: Scan file system or inject diagnostic CSV
        print('\nReading FITS headers...\n')
        extractor = HeaderExtractor(logger, config)
        if args.test:
            # Load from CSV for reproducibility and rapid testing
            raw_df = extractor.extract_from_csv(args.test)
        else:
            # Parallelized scan of all provided directories
            raw_df = extractor.extract_from_directories(directory_paths)
            
            # NEW: Export raw headers if debug is enabled. 
            # This file is perfectly matched for the --test injection point.
            if args.debug and not raw_df.empty:
                raw_csv_path = os.path.join(output_dir, "debug_step_00_RawHeaders.csv")
                raw_df.to_csv(raw_csv_path, index=False)
                logger.info(f"Raw scanned headers exported to {raw_csv_path}")

        # --- Step 3: Pipeline Configuration ---
        
        # Build the transformation sequence using logical Steps.
        # The order of these steps is critical as they have data dependencies.
        processor = PipelineProcessor(logger)
        processor.add_step(NormalizeHeadersStep())    # Stage 1: Sanitation & Overrides
        processor.add_step(OpticalParameterStep())    # Stage 2: Resolution & Star Metrics
        processor.add_step(DeduplicateStep())         # Stage 3: WBPP Filtering
        processor.add_step(CalibrationMatcherStep())  # Stage 4: Gain Handshake & CAL matching
        processor.add_step(GeocodeStep())             # Stage 5: Site identification
        processor.add_step(AggregationStep())         # Stage 6: Vectorized Session Summary

        # --- Stage 4: Execution & Export ---
        
        # Initialize the shared SessionState container
        state = SessionState(config=config, raw_df=raw_df)
        
        # Execute the transformation pipeline
        state = processor.run(state, debug=args.debug, output_dir=output_dir)
        
        # Export the final artifacts (Acquisition CSV and Text Summary)
        output_basename = os.path.basename(args.directory_paths[0]).replace(" ", "_")
        exporter = Exporter(logger)
        exporter.export(state, output_basename, output_dir)

        print("\nProcessing complete.")

    except Exception as e:
        # Final safety net: Ensure any unhandled exception is logged before the program dies
        logger.error("The application encountered a fatal error and must exit.")
        logger.exception(e)
        
        # If we have any data at all, dump it for emergency diagnostics.
        # Kept broad and still non-fatal on purpose -- this runs while the
        # program is already dying, and must never itself raise -- but now
        # at least logs what went wrong with the dump attempt, rather than
        # silently discarding it (B3 in REMEDIATION_PLAN.md).
        try:
            if 'raw_df' in locals() and not raw_df.empty:
                emergency_csv = os.path.join(output_dir, "emergency_raw_dump.csv")
                raw_df.to_csv(emergency_csv, index=False)
                print(f"Emergency data dump saved to: {emergency_csv}")
        except Exception as dump_error:
            logger.debug(f"Emergency data dump also failed: {dump_error}")

        print(f"\n[CRITICAL ERROR]: {str(e)}")
        print(f"Detailed diagnostics have been saved to: {log_file}")
        sys.exit(1)

if __name__ == "__main__":
    main()