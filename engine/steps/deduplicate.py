"""
File Deduplication Module - AstroBin Upload Utility v2.0.2

This module addresses the problem of 'metadata duplication' caused by image 
preprocessing software (e.g., PixInsight WeightedBatchPreprocessing - WBPP). 

When a raw file is calibrated or registered, WBPP creates new files with 
postfixes like '_c', '_cc', or '_r'. If these files coexist in the same 
directory as the raw captures, the utility would normally count the same 
exposure multiple times. 

The DeduplicateStep identifies these related files and selects only the 
highest-priority version (usually the raw capture or the XISF master) 
to ensure accurate total exposure calculations.
"""

import pandas as pd
import re
import logging
from models import SessionState
from constants import InternalColumns

class DeduplicateStep:
    """
    Identifies and removes duplicate metadata entries resulting from preprocessing.
    """
    def execute(self, state: SessionState) -> SessionState:
        """
        Executes the deduplication logic using base filename extraction and ranking.
        
        Args:
            state (SessionState): The current pipeline state.
            
        Returns:
            SessionState: The state with a unique set of captures.
        """
        logger = logging.getLogger("AstroBinV2")
        logger.info("Executing WBPP deduplication filter")
        
        df = state.processed_df
        if df.empty: return state

        # Track count for logging
        orig_count = len(df)

        # --- Stage 1: Base Filename Extraction ---
        # We use a non-greedy regex to strip off WBPP postfixes (e.g., _c, _cc, _r, _rn) 
        # to identify the original capture name. 
        # Example: 'M31_Light_001_c.xisf' and 'M31_Light_001.fits' both map to 'M31_Light_001'
        df['base_filename'] = df[InternalColumns.FILENAME].str.extract(
            r'(.+?)(?:_c.*)?(\.xisf|\.fits|\.fit|\.fts)', 
            flags=re.IGNORECASE
        )[0]
        
        # --- Stage 2: Priority Selection ---
        # When multiple files share the same base, we apply a ranking system 
        # to decide which one to keep.
        final_rows = []
        
        # Preference: PixInsight XISF > Standard FITS > Aliases
        ext_priority = {'.xisf': 0, '.fits': 1, '.fit': 2, '.fts': 3}

        for base, group in df.groupby('base_filename'):
            if pd.isna(base): continue
            
            # Create a rank for each file based on its extension
            group = group.copy()
            group['ext_rank'] = group[InternalColumns.FILENAME].apply(
                lambda x: next((v for k, v in ext_priority.items() if str(x).lower().endswith(k)), 9)
            )
            
            # Sorting logic: 
            # 1. Prefer higher extension priority (ext_rank).
            # 2. Prefer shorter filenames (raw files are usually shorter than post-processed ones).
            match = group.sort_values(['ext_rank', InternalColumns.FILENAME], key=lambda x: x.str.len() if x.name == InternalColumns.FILENAME else x).iloc[0]
            final_rows.append(match)
            
        # Reconstruct the dataframe from the unique selection
        if final_rows:
            new_df = pd.DataFrame(final_rows).drop(columns=['base_filename', 'ext_rank'])
            new_count = len(new_df)
            if orig_count != new_count:
                logger.info(f"Deduplication: Removed {orig_count - new_count} duplicate/intermediate frames")
            state.processed_df = new_df
        
        return state