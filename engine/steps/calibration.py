"""
Calibration Matching Module - AstroBin Upload Utility v2.1.0

This module implements the 'Calibration Matcher' logic, identifying which 
Dark, Flat, and Bias frames belong to each Light frame. 

Improvements:
- Strict Linear Gain Handshake.
- Strict Light-Frame Authority (Drop mismatched linear gains).
- Master Preference (If Master exists, ignore Raws).
"""

import pandas as pd
import logging
from models import SessionState
from constants import InternalColumns, ImageType

# EGAIN is treated as "not meaningfully set" when it sits within this
# tolerance of the NormalizeHeadersStep default (1.0) -- a value that close
# to the placeholder is far more likely to be the unset default than a
# genuine electronic gain reading of exactly 1.0 e/ADU, so the hybrid key
# falls back to the linear integer GAIN instead.
EGAIN_UNSET_TOLERANCE = 0.0001

class CalibrationMatcherStep:
    """
    Associates calibration frames with their corresponding LIGHT frames.
    """
    def execute(self, state: SessionState) -> SessionState:
        """
        Executes matching logic with strict 'Light-Frame Authority' pruning.
        """
        logger = logging.getLogger("AstroBinV2")
        logger.info("Starting calibration matching process")
        df = state.processed_df
        if df.empty: return state

        # --- Stage 1: Hybrid Handshake (EGAIN with GAIN Fallback) ---
        # We prioritize EGAIN if it exists (not 1.0), else fall back to linear GAIN.
        def create_hybrid_key(row):
            try:
                egain_val = float(row[InternalColumns.EGAIN])
                # Use a rounded signature (2 decimals) to bridge precision differences (0.2467 -> 0.25)
                if abs(egain_val - 1.0) > EGAIN_UNSET_TOLERANCE:
                    return f"E_{egain_val:.2f}"
            except (ValueError, TypeError) as e:
                logger.debug(
                    f"Hybrid gain key: unparseable EGAIN for "
                    f"{row.get(InternalColumns.FILENAME, '<unknown file>')}, falling back to GAIN ({e})"
                )

            # Fallback to Linear Gain anchor
            try:
                gain_val = int(round(float(row[InternalColumns.GAIN])))
                return f"G_{gain_val}"
            except (ValueError, TypeError) as e:
                logger.debug(
                    f"Hybrid gain key: unparseable GAIN for "
                    f"{row.get(InternalColumns.FILENAME, '<unknown file>')}, using G_0 ({e})"
                )
                return "G_0"

        df[InternalColumns.GAIN_MATCH] = df.apply(create_hybrid_key, axis=1)
        
        unique_keys = df[InternalColumns.GAIN_MATCH].unique()
        logger.debug(f"Calibration Handshake: Identified hybrid matching keys: {list(unique_keys)}")
        
        # --- Stage 2: Light-Frame Authority (Strict Pruning) ---
        # Extract valid anchors from Light frames. 
        logger.debug("Identifying calibration frame anchors from light frames")
        is_light = df[InternalColumns.IMAGE_TYPE] == ImageType.LIGHT.value
        lights_auth = df[is_light].copy()
        
        if not lights_auth.empty:
            # Anchor set for Darks/Bias (EGAIN + Binning)
            dark_bias_anchors = set(zip(
                lights_auth[InternalColumns.GAIN_MATCH],
                lights_auth[InternalColumns.BINNING]
            ))
            
            # Anchor set for Flats (Filter + EGAIN + Binning)
            # We strictly enforce that Flats must match the EGAIN of the Light.
            flat_anchors = set(zip(
                lights_auth[InternalColumns.FILTER_NAME].astype(str).str.lower(),
                lights_auth[InternalColumns.GAIN_MATCH],
                lights_auth[InternalColumns.BINNING]
            ))

            def is_orphaned(row):
                itype = str(row[InternalColumns.IMAGE_TYPE]).upper()
                if itype == ImageType.LIGHT.value: return False
                
                gain = row[InternalColumns.GAIN_MATCH]
                binning = row[InternalColumns.BINNING]
                filt = str(row[InternalColumns.FILTER_NAME]).lower()
                
                # Rule 1: Flats must match Filter AND Gain AND Binning
                if "FLAT" in itype and "DARK" not in itype:
                    return (filt, gain, binning) not in flat_anchors
                
                # Rule 2: Darks/Bias must match Gain AND Binning
                # (Ignore filter for these)
                if any(t in itype for t in ["DARK", "BIAS"]):
                    return (gain, binning) not in dark_bias_anchors
                
                return False

            # Completely discard orphaned calibration data
            # This drops the Gain 1 flats if Lights are Gain 100
            mask_orphaned = df.apply(is_orphaned, axis=1)
            df = df[~mask_orphaned].copy()

        # --- Stage 3: Segmentation ---
        logger.debug("Segmenting lights and calibration frames")
        lights_mask = df[InternalColumns.IMAGE_TYPE] == ImageType.LIGHT.value
        lights = df[lights_mask].copy()
        cals = df[~lights_mask].copy()
        
        logger.debug(f"Split DataFrame: {len(lights)} LIGHTS and {len(cals)} calibration frames")

        if lights.empty: 
            state.processed_df = df
            return state

        # Initialize counts
        for col in ['darks', 'flats', 'flatDarks', 'bias']:
            lights[col] = 0

        # --- Stage 4: Matching Loop with Master Preference ---
        logger.debug("Matching calibration frames to light frames")
        for idx, row in lights.iterrows():
            
            # 4a. Identify Candidates
            # Darks: Match Gain, Bin, Duration
            dark_candidates = cals[
                cals[InternalColumns.IMAGE_TYPE].str.upper().str.contains('DARK', na=False) & \
                ~cals[InternalColumns.IMAGE_TYPE].str.upper().str.contains('FLAT', na=False) & \
                (cals[InternalColumns.GAIN_MATCH] == row[InternalColumns.GAIN_MATCH]) & \
                (cals[InternalColumns.BINNING] == row[InternalColumns.BINNING]) & \
                (cals[InternalColumns.DURATION] == row[InternalColumns.DURATION])
            ]

            # Bias: Match Gain, Bin
            bias_candidates = cals[
                cals[InternalColumns.IMAGE_TYPE].str.upper().str.contains('BIAS', na=False) & \
                (cals[InternalColumns.GAIN_MATCH] == row[InternalColumns.GAIN_MATCH]) & \
                (cals[InternalColumns.BINNING] == row[InternalColumns.BINNING])
            ]

            # Flats: Match Filter, Gain, Bin
            flat_candidates = cals[
                cals[InternalColumns.IMAGE_TYPE].str.upper().str.contains('FLAT', na=False) & \
                ~cals[InternalColumns.IMAGE_TYPE].str.upper().str.contains('DARK', na=False) & \
                (cals[InternalColumns.FILTER_NAME].str.lower() == str(row[InternalColumns.FILTER_NAME]).lower()) & \
                (cals[InternalColumns.GAIN_MATCH] == row[InternalColumns.GAIN_MATCH]) & \
                (cals[InternalColumns.BINNING] == row[InternalColumns.BINNING])
            ]

            # 4b. Apply Master Preference Rule
            # If a Master exists in the candidates, use ONLY the Master(s).
            # Otherwise, use all Raw candidates.
            
            def resolve_count(candidates):
                if candidates.empty: return 0

                # Check for Master presence
                is_master_mask = candidates[InternalColumns.IMAGE_TYPE].str.upper().str.contains('MASTER', na=False)

                if is_master_mask.any():
                    # If Masters exist, keep ONLY one to prevent double-counting
                    # between multiple identified master versions of the same data.
                    #
                    # A10 in REMEDIATION_PLAN.md, per user decision: prefer
                    # the most recent master (by DATE-OBS) rather than an
                    # arbitrary "first found", mirroring the same rule in
                    # base.py's _execute_master_preference. This branch was
                    # unreachable before A13 fixed IMAGETYP normalization
                    # (no row could carry a 'MASTER' label by the time this
                    # step ran), so it had no observable effect until now.
                    masters = candidates[is_master_mask]
                    if len(masters) > 1:
                        parsed_dates = pd.to_datetime(masters[InternalColumns.DATE_OBS], errors='coerce')
                        if parsed_dates.notna().any():
                            final_set = masters.loc[[parsed_dates.idxmax()]]
                        else:
                            final_set = masters.iloc[[0]]
                    else:
                        final_set = masters
                else:
                    # Use all raws
                    final_set = candidates

                return int(final_set[InternalColumns.NUMBER].sum())

            # 4c. Assign Counts
            d_count = resolve_count(dark_candidates)
            b_count = resolve_count(bias_candidates)
            f_count = resolve_count(flat_candidates)
            
            lights.at[idx, 'darks'] = d_count
            lights.at[idx, 'bias']  = b_count
            lights.at[idx, 'flats'] = f_count
            
            # FlatDarks: Match Filter, Gain, Bin -- same criteria as Flats,
            # and routed through the same resolve_count() master-preference
            # logic as the other three calibration types (A11 in
            # REMEDIATION_PLAN.md, per user decision). Previously this
            # omitted the BINNING constraint entirely (matching flat-darks
            # shot at a different binning to the light), and summed master
            # + raw candidates together unconditionally instead of
            # preferring the master -- both inconsistent with how darks,
            # bias and flats are matched just above.
            flat_dark_candidates = cals[
                cals[InternalColumns.IMAGE_TYPE].str.upper().str.contains('DARKFLAT', na=False) & \
                (cals[InternalColumns.FILTER_NAME].str.lower() == str(row[InternalColumns.FILTER_NAME]).lower()) & \
                (cals[InternalColumns.GAIN_MATCH] == row[InternalColumns.GAIN_MATCH]) & \
                (cals[InternalColumns.BINNING] == row[InternalColumns.BINNING])
            ]
            fd_count = resolve_count(flat_dark_candidates)
            lights.at[idx, 'flatDarks'] = fd_count

            if d_count > 0 or b_count > 0 or f_count > 0 or fd_count > 0:
                logger.debug(f"Light Index {idx} ({row.get('filename', 'Unknown')}): Assigned {d_count} Darks, {f_count} Flats, {b_count} Bias, {fd_count} FlatDarks.")

        # --- Stage 5: Reintegration ---
        # Combine enriched lights with authority-validated calibration
        df = pd.concat([lights, cals], ignore_index=True)
        
        state.processed_df = df
        return state