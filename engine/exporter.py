"""
Data Exporter Module - AstroBin Upload Utility v2.0.2

Responsible for generating the final user-facing artifacts from the 
processed session state. This module handles the conversion of internal 
vectorized data into formatted reports that are either compatible with 
the AstroBin web interface or designed for human review.

Key Outputs:
1.  **Acquisition CSV**: A focused table containing only LIGHT frames, 
    formatted to match the AstroBin "Bulk Upload" requirements.
2.  **Session Summary**: A comprehensive text-based report detailing 
    equipment used, environmental conditions, and all frame types 
    (Light and Calibration).
"""

import os
import logging
from models import SessionState
from constants import ImageType, InternalColumns

class Exporter:
    """
    Handles the formatting and persistence of processed session data.
    
    The Exporter acts as the final stage of the pipeline, turning the 
    internal state into persistent files on disk.
    """
    def __init__(self, logger: logging.Logger):
        """
        Initializes the Exporter with a system logger.

        Args:
            logger (logging.Logger): The active application logger.
        """
        self.logger = logger

    def export(self, state: SessionState, output_basename: str, output_dir: str):
        """
        Generates and saves the final output reports.
        
        This method performs the final mapping, rounding, and file I/O 
        required to complete the session processing.

        Args:
            state (SessionState): The finalized state containing aggregated data.
            output_basename (str): The prefix used for generated filenames.
            output_dir (str): The destination directory for the output files.
        """
        # Create a local copy of aggregated data to avoid side-effects
        df = state.aggregated_df.copy()
        if df.empty:
            self.logger.warning("Export requested but aggregated data is empty.")
            return

        # --- Stage 1: Acquisition CSV Generation (AstroBin Specific) ---
        
        # 1. Filter for Acquisition CSV (LIGHTS ONLY)
        # The AstroBin bulk upload interface strictly processes light frames. 
        # Calibration frames are summarized elsewhere.
        acq_source = df[df[InternalColumns.IMAGE_TYPE] == ImageType.LIGHT.value].copy()

        # 2. Map Filter Names to IDs
        # AstroBin uses specific numeric IDs for many common filters. These 
        # mappings are loaded from the [filters] section of config.ini and 
        # were pre-processed in the AggregationStep.
        if 'filter_code' not in acq_source.columns:
            filter_dict = state.config.filters
            acq_source['filter_code'] = acq_source['filter'].apply(lambda x: filter_dict.get(str(x).strip(), x))

        # 3. Map Internal Column Names to AstroBin CSV Standards
        # This mapping bridges the gap between our internal lowercase keys 
        # and the specific header names expected by the AstroBin CSV parser.
        mapping = {
            'session_date': 'date',
            'filter_code': 'filter',
            'number': 'number',
            'exposure': 'duration',
            'xbinning': 'binning',
            'gain': 'gain',           # Uses pure linear integer gain
            'ccd-temp': 'sensorCooling',
            'focratio': 'fNumber',
            'darks': 'darks',
            'flats': 'flats',
            'flatDarks': 'flatDarks',
            'bias': 'bias',
            'bortle': 'bortle',
            'sqm': 'meanSqm',
            'fwhm': 'meanFwhm',
            'foctemp': 'temperature'
        }
        
        # Perform selection and renaming in one step.
        # B12 in REMEDIATION_PLAN.md: a plain acq_source[list(mapping.keys())]
        # raises a bare KeyError naming only the missing column, with no
        # indication of why an expected core column would be absent from
        # aggregated data at this late a stage. Check explicitly and raise
        # something a user (or a future maintainer) can actually act on.
        missing_cols = [c for c in mapping if c not in acq_source.columns]
        if missing_cols:
            raise KeyError(
                f"Aggregated data is missing expected column(s) {missing_cols} "
                f"required for the acquisition CSV. This usually means a core "
                f"column was renamed or an AggregationStep reduction rule was "
                f"dropped -- check engine/steps/aggregate.py's `rules` dict "
                f"against this module's `mapping`."
            )
        acq_df = acq_source[list(mapping.keys())].rename(columns=mapping)
        
        # 4. Final Data Hardening & Rounding
        # Apply standard rounding to ensure clean output in the CSV.
        acq_df = acq_df.round({
            'duration': 2, 
            'sensorCooling': 0, 
            'fNumber': 2, 
            'meanSqm': 2, 
            'meanFwhm': 2, 
            'temperature': 2
        })

        # Persist the Acquisition CSV to disk
        output_csv_path = os.path.join(output_dir, f"{output_basename}_acquisition.csv")
        acq_df.to_csv(output_csv_path, index=False)
        self.logger.info(f"Acquisition CSV saved: {output_csv_path}")

        # --- Stage 2: Human-Readable Summary Generation ---

        # 1. Generate the text-based summary using the dedicated reports module
        # This includes calibration statistics and equipment summaries.
        from engine.reports import generate_full_summary
        summary = generate_full_summary(df, self.logger, len(state.raw_df))
        
        # 2. Append the CSV preview
        # We append a string representation of the acquisition table to the 
        # bottom of the text summary for quick visual verification by the user.
        # We use max_rows=None to ensure the entire table is visible in the text report.
        df_string = acq_df.to_string(index=False, max_rows=None, max_cols=None).replace('\n', '\n ')
        summary += f"\n{output_basename}_acquisition.csv\n\n {df_string}\n"
        
        # 3. Save the Text Summary to disk
        summary_file = os.path.join(output_dir, f"{output_basename}_session_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # 4. Final Console Output
        # Print the summary to standard out for immediate feedback
        print(summary)
        self.logger.info(f"Session summary saved: {summary_file}")
