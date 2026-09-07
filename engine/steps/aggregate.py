"""
Vectorized Aggregation Engine - AstroBin Upload Utility v2.1.0

This module implements the final transformation stage of the pipeline: 
Summarizing hundreds or thousands of individual frame headers into a 
concise set of session-level statistics. 

It utilizes high-speed Pandas vectorized operations for:
1.  **Temporal Analysis**: Identifying discrete observation sessions based 
    on time gaps.
2.  **Date Shifting**: Normalizing overnight sessions into single logical 
    observation dates.
3.  **Statistical Aggregation**: Calculating mean temperatures, FWHM, and 
    summing total exposure counts.
"""

import pandas as pd
import logging
from datetime import timedelta
from models import SessionState
from constants import ImageType, InternalColumns

# A gap longer than this between consecutive light frames is treated as the
# boundary between two separate observation sessions (e.g. two different
# nights). 5 hours comfortably exceeds a typical meridian flip or brief
# cloud-cover pause within one night's imaging run, while still being well
# short of the daylight gap between one night and the next.
SESSION_GAP_HOURS = 5

class AggregationStep:
    """
    Groups and summarizes the processed metadata into export-ready session stats.
    """
    def execute(self, state: SessionState) -> SessionState:
        """
        Executes the aggregation logic on the processed dataframe.
        
        Args:
            state (SessionState): The current pipeline state.
            
        Returns:
            SessionState: The state with populated aggregated_df.
        """
        logger = logging.getLogger("AstroBinV2")
        logger.info("Aggregating parameters...")
        df = state.processed_df
        if df.empty: return state

        # --- Stage 1: Temporal Normalization ---
        logger.debug("Performing temporal normalization")

        # Ensure observation dates are proper datetime objects for vectorized math
        df[InternalColumns.DATE_OBS] = pd.to_datetime(df[InternalColumns.DATE_OBS], errors='coerce')
        # kind='mergesort' (stable): the default quicksort does not preserve
        # input order among frames sharing a timestamp, which made ties break
        # arbitrarily -- and every downstream first()/iloc[0] read depends on
        # a deterministic order (A9 in REMEDIATION_PLAN.md).
        df = df.sort_values(InternalColumns.DATE_OBS, kind='mergesort').reset_index(drop=True)
        
        # Identify GLOBAL session statistics based exclusively on Light frames
        lights_mask = df[InternalColumns.IMAGE_TYPE] == ImageType.LIGHT.value
        lights = df[lights_mask].copy()
        
        if not lights.empty:
            # Session Detection: Any gap larger than 5 hours indicates a new session
            time_diff = lights[InternalColumns.DATE_OBS].diff()
            session_count = (time_diff > pd.Timedelta(hours=SESSION_GAP_HOURS)).cumsum().max() + 1
            
            # Temporal Bounds
            start_date = lights[InternalColumns.DATE_OBS].min().strftime('%Y-%m-%d')
            end_date = lights[InternalColumns.DATE_OBS].max().strftime('%Y-%m-%d')
            num_days = (lights[InternalColumns.DATE_OBS].max().date() - lights[InternalColumns.DATE_OBS].min().date()).days + 1
        else:
            session_count = 0
            start_date = "N/A"
            end_date = "N/A"
            num_days = 0

        # --- Stage 2: Overnight Date Shifting ---
        
        # If use_obs_date is False, we shift frames taken after midnight to the 
        # previous day's date so they appear as part of a single continuous night.
        if not state.config.use_obs_date:
            logger.debug("Applying overnight date shifting")
            time_diff = df[InternalColumns.DATE_OBS].diff()
            session_ids = (time_diff > pd.Timedelta(hours=SESSION_GAP_HOURS)).cumsum()
            
            def calculate_ref_date(ts):
                # If taken before noon, it belongs to the previous calendar day
                if ts.time() < pd.Timestamp("12:00:00").time():
                    return (ts - timedelta(days=1)).date()
                return ts.date()

            session_starts = df.groupby(session_ids)[InternalColumns.DATE_OBS].transform('first')
            df['session_date'] = session_starts.apply(calculate_ref_date)
        else:
            # Otherwise, use the actual calendar date of capture
            df['session_date'] = df[InternalColumns.DATE_OBS].dt.date

        # Broadcast global session values to all rows for inclusion in final aggregation
        df[InternalColumns.SESSIONS] = session_count
        df[InternalColumns.START_DATE] = str(start_date)
        df[InternalColumns.END_DATE] = str(end_date)
        df[InternalColumns.NUM_DAYS] = num_days

        # Ensure dates are strings to prevent formatting issues in reports
        df[InternalColumns.START_DATE] = df[InternalColumns.START_DATE].astype(str)
        df[InternalColumns.END_DATE] = df[InternalColumns.END_DATE].astype(str)

        # B1 in REMEDIATION_PLAN.md: this used to be a `for i in
        # range(1, total_lights + 1): print(...)` loop -- a fake progress
        # bar. It did no actual per-frame work (the real aggregation below
        # is one vectorized groupby().agg() call) and, for a large dataset,
        # wasted real time on nothing but repeated print()+flush() calls
        # while implying slow per-frame processing that wasn't happening.
        if not lights.empty:
            logger.debug(f"Aggregating {len(lights)} light frame(s).")

        # --- Stage 3: Aggregation ---
        logger.debug("Grouping and summarizing metadata")

        # Define the primary keys for grouping data
        agg_cols = [
            InternalColumns.SITE_NAME, 
            'session_date', 
            InternalColumns.IMAGE_TYPE, 
            InternalColumns.FILTER_NAME, 
            InternalColumns.GAIN, # Grouping by linear integer gain for CSV/Report consistency
            InternalColumns.BINNING, 
            InternalColumns.DURATION, 
            InternalColumns.TARGET
        ]
        
        # Prevent "Lossy Aggregation" by filling missing grouping keys.
        # pandas groupby drops a row outright if any of its key columns is
        # null, so every grouping key needs a non-null fallback.
        #
        # A5 in REMEDIATION_PLAN.md: filling a *numeric* key (gain,
        # xbinning, exposure are all in agg_cols) with the string "None"
        # promotes the whole column to object dtype -- as soon as a single
        # null appears anywhere in it, every value in the column becomes a
        # Python float/str mix, and the acquisition CSV then writes e.g.
        # "100.0" instead of the integer "100" AstroBin's importer expects.
        # In the current pipeline this branch is unreachable for those
        # three specifically: NormalizeHeadersStep's Stage 7 hardening
        # (engine/steps/base.py) unconditionally fills and casts all three
        # to a non-null numeric dtype before this step ever runs. This is
        # nonetheless fixed to fail safe rather than silently corrupting
        # output if that hardening guarantee is ever changed or bypassed.
        for col in agg_cols:
            if col not in df.columns:
                df[col] = "None"
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("None")

        # Ensure all numeric columns are strictly typed before mathematical aggregation
        numeric_agg_cols = [
            InternalColumns.SENSOR_COOLING, InternalColumns.MEAN_FWHM, 
            InternalColumns.SITE_LAT, InternalColumns.SITE_LONG, 
            InternalColumns.F_NUMBER, InternalColumns.TEMPERATURE, 
            InternalColumns.BORTLE, InternalColumns.MEAN_SQM, 
            InternalColumns.EGAIN, 'darks', 'flats', 'flatDarks', 'bias'
        ]
        for col in numeric_agg_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Define the reduction rules for each column
        rules = {
            InternalColumns.NUMBER: (InternalColumns.NUMBER, 'sum'),
            InternalColumns.SENSOR_COOLING: (InternalColumns.SENSOR_COOLING, 'mean'),
            InternalColumns.TEMP_MIN: (InternalColumns.TEMPERATURE, 'min'),
            InternalColumns.TEMP_MAX: (InternalColumns.TEMPERATURE, 'max'),
            InternalColumns.MEAN_FWHM: (InternalColumns.MEAN_FWHM, 'mean'),
            InternalColumns.SITE_LAT: (InternalColumns.SITE_LAT, 'mean'),
            InternalColumns.SITE_LONG: (InternalColumns.SITE_LONG, 'mean'),
            InternalColumns.F_NUMBER: (InternalColumns.F_NUMBER, 'mean'),
            InternalColumns.TEMPERATURE: (InternalColumns.TEMPERATURE, 'mean'),
            InternalColumns.FOCAL_LENGTH: (InternalColumns.FOCAL_LENGTH, 'first'),
            InternalColumns.BORTLE: (InternalColumns.BORTLE, 'mean'),
            InternalColumns.MEAN_SQM: (InternalColumns.MEAN_SQM, 'mean'),
            InternalColumns.PIXEL_SIZE: (InternalColumns.PIXEL_SIZE, 'first'),
            InternalColumns.GAIN_MATCH: (InternalColumns.GAIN_MATCH, 'first'),
            InternalColumns.EGAIN: (InternalColumns.EGAIN, 'mean'),
            InternalColumns.CAMERA: (InternalColumns.CAMERA, 'first'),
            InternalColumns.TELESCOPE: (InternalColumns.TELESCOPE, 'first'),
            InternalColumns.FOCUSER: (InternalColumns.FOCUSER, 'first'),
            InternalColumns.FILTER_WHEEL: (InternalColumns.FILTER_WHEEL, 'first'),
            InternalColumns.ROTATOR_NAME: (InternalColumns.ROTATOR_NAME, 'first'),
            InternalColumns.SWCREATE: (InternalColumns.SWCREATE, 'first'),
            InternalColumns.FILENAME: (InternalColumns.FILENAME, 'first'),
            InternalColumns.SESSIONS: (InternalColumns.SESSIONS, 'first'),
            InternalColumns.START_DATE: (InternalColumns.START_DATE, 'first'),
            InternalColumns.END_DATE: (InternalColumns.END_DATE, 'first'),
            InternalColumns.NUM_DAYS: (InternalColumns.NUM_DAYS, 'first'),
            'darks': ('darks', 'max'),
            'flats': ('flats', 'max'),
            'flatDarks': ('flatDarks', 'max'),
            'bias': ('bias', 'max')
        }

        # Perform the actual vectorized aggregation
        agg_df = df.groupby(agg_cols, observed=True).agg(**rules).reset_index()

        # --- Stage 4: Filter Mapping & Logging ---
        logger = logging.getLogger("AstroBinV2")
        filter_dict = state.config.filters
        
        def map_filter(name):
            name_str = str(name).strip()
            if name_str in filter_dict:
                code = filter_dict[name_str]
                logger.debug(f"Filter Mapping: SUCCESS - Mapped '{name_str}' to code '{code}'")
                return code
            else:
                logger.debug(f"Filter Mapping: FAILURE - No code found for '{name_str}', using original name.")
                return name_str

        if InternalColumns.FILTER_NAME in agg_df.columns:
            agg_df['filter_code'] = agg_df[InternalColumns.FILTER_NAME].apply(map_filter)

        state.aggregated_df = agg_df
        return state
                        