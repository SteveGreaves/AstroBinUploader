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

import os
import pandas as pd
import re
import logging
from models import SessionState
from constants import InternalColumns, RegexPatterns

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
        # Strip off a WBPP postfix chain (e.g. _c, _c_cc, _c_lps_r) to
        # identify the original capture name.
        # Example: 'M31_Light_001_c.xisf' and 'M31_Light_001.fits' both map to 'M31_Light_001'
        #
        # A1 in REMEDIATION_PLAN.md: the previous pattern here was
        # unanchored and matched a bare '_c' anywhere in the filename
        # (e.g. inside '_calibrated_'), then swallowed everything up to
        # the extension -- silently merging unrelated captures. See
        # constants.RegexPatterns.WBPP_FILENAME for the anchored
        # replacement and its rationale.
        df['base_filename'] = df[InternalColumns.FILENAME].str.extract(
            RegexPatterns.WBPP_FILENAME,
            flags=re.IGNORECASE
        )[0]

        # --- Stage 1b: Directory Key ---
        # Deduplication must be scoped to a directory: two different capture
        # sessions frequently reuse the same filename (e.g. many capture
        # tools default to 'Light_0001.fits'), and without this, frames from
        # unrelated sessions would collapse into one (A2 in
        # REMEDIATION_PLAN.md). SOURCE_PATH is only absent when replaying a
        # --test CSV captured by a pre-A2 version; degrade gracefully to the
        # old filename-only behaviour in that case rather than erroring, so
        # existing debug_step_00_RawHeaders.csv / emergency_raw_dump.csv
        # exports stay replayable.
        if InternalColumns.SOURCE_PATH in df.columns:
            df['_dedup_dir'] = df[InternalColumns.SOURCE_PATH].apply(
                lambda p: os.path.dirname(str(p)) if pd.notna(p) else ''
            )
        else:
            logger.warning(
                "SOURCE_PATH column absent (--test CSV predates A2) -- "
                "deduplicating on filename alone, which can merge "
                "identically-named captures from different directories. "
                "Re-run with a live directory scan, or a fixture captured "
                "by the current version, to get directory-aware dedup."
            )
            df['_dedup_dir'] = ''

        # --- Stage 2: Priority Selection ---
        # When multiple files share the same (directory, base filename), we
        # apply a ranking system to decide which one to keep.
        final_rows = []

        # Preference: PixInsight XISF > Standard FITS > Aliases
        ext_priority = {'.xisf': 0, '.fits': 1, '.fit': 2, '.fts': 3}

        for (dedup_dir, base), group in df.groupby(['_dedup_dir', 'base_filename']):
            if pd.isna(base): continue
            
            # Create a rank for each file based on its extension
            group = group.copy()
            group['ext_rank'] = group[InternalColumns.FILENAME].apply(
                lambda x: next((v for k, v in ext_priority.items() if str(x).lower().endswith(k)), 9)
            )
            
            # Sorting logic:
            # 1. Prefer higher extension priority (ext_rank).
            # 2. Prefer shorter filenames (raw files are usually shorter than post-processed ones).
            # kind='mergesort' (stable): when both keys tie -- two genuinely
            # distinct captures of equal filename length -- the survivor is
            # then decided by input order, which is deterministic given the
            # sorted dispatch order from HeaderExtractor (A9 in
            # REMEDIATION_PLAN.md), rather than by the default quicksort's
            # unspecified tie placement.
            match = group.sort_values(
                ['ext_rank', InternalColumns.FILENAME],
                key=lambda x: x.str.len() if x.name == InternalColumns.FILENAME else x,
                kind='mergesort'
            ).iloc[0]
            final_rows.append(match)

            if len(group) > 1:
                dropped = [f for f in group[InternalColumns.FILENAME] if f != match[InternalColumns.FILENAME]]
                logger.debug(
                    f"Deduplication: kept '{match[InternalColumns.FILENAME]}', "
                    f"dropped {dropped} (same base '{base}' in '{dedup_dir}')"
                )

        # Reconstruct the dataframe from the unique selection
        if final_rows:
            new_df = pd.DataFrame(final_rows).drop(columns=['base_filename', 'ext_rank', '_dedup_dir'])
            new_count = len(new_df)
            if orig_count != new_count:
                logger.info(f"Deduplication: Removed {orig_count - new_count} duplicate/intermediate frames")
            state.processed_df = new_df
        
        return state