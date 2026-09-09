"""
Standard Metadata Normalization Module - AstroBin Upload Utility v2.1.1

This module implements the first stage of the transformation pipeline: 
'NormalizeHeadersStep'. Its primary responsibility is to take the raw, 
often inconsistent metadata from FITS/XISF headers and transform it into 
 a standardized internal format.

Tasks performed:
1.  **Hardware Overrides**: Mapping custom hardware keywords (e.g., 'EXPTIME') 
    to internal standard keys (e.g., 'exposure').
2.  **Default Injection**: Filling missing metadata with user-defined defaults.
3.  **IMAGETYP Normalization**: Standardizing frame types (LIGHT, FLAT, etc.) 
    using substring matching.
4.  **Type Hardening**: Ensuring critical columns are numeric (float/int) and 
    applying fallbacks for 'NaN' values.
"""

import pandas as pd
import logging
from models import SessionState
from constants import InternalColumns, ImageType


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges columns sharing the same name, keeping the first non-null value
    across each duplicate-named group (left to right).

    B2 in REMEDIATION_PLAN.md: replaces the deprecated
    `df.groupby(level=0, axis=1).first()` (FutureWarning on pandas 2.2,
    removed in pandas 3). The stdlib-suggested direct replacement,
    `df.T.groupby(level=0).T`, is NOT a safe drop-in here: transposing a
    DataFrame that mixes numeric and string columns (which this one
    always does -- e.g. 'exposure' alongside 'imagetyp') upcasts every
    column to object dtype and turns NaN into None, silently reproducing
    the exact object-dtype corruption fixed in A5 -- verified directly
    before writing this. Coalescing within each same-named group's own
    sub-frame instead keeps each group's original, homogeneous dtype,
    since duplicate columns share a name precisely because they represent
    the same logical field. Column order is sorted to match groupby's
    default `sort=True` exactly, since that's what the replaced call
    produced. Verified against the original call with `.equals()`, not
    just eyeballed, across both dtype-mixed and duplicate-plus-unique
    column-order cases.
    """
    if not df.columns.duplicated().any():
        return df
    result = {}
    for name in sorted(df.columns.unique()):
        sub = df[name]
        result[name] = sub if isinstance(sub, pd.Series) else sub.bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(result, index=df.index)


class NormalizeHeadersStep:
    """
    Sanitizes raw FITS/XISF metadata into a standardized internal format.
    """
    def execute(self, state: SessionState) -> SessionState:
        """
        Executes the normalization logic on the raw_df.

        Args:
            state (SessionState): The current pipeline state.
            
        Returns:
            SessionState: The state with a populated and cleaned processed_df.
        """
        logger = logging.getLogger("AstroBinV2")
        logger.info("Initialising headers state")
        
        # Create a working copy of the raw data
        df = state.raw_df.copy()
        config = state.config

        # --- Stage 1: Hardware Overrides ---
        # We process each override in the order specified in the configuration.
        # The first matching hardware key in the list takes precedence.
        for internal_key, hw_keys in config.overrides.items():
            combined_series = None
            found_cols = []
            for hw_key in hw_keys:
                # Case-insensitive search for matching hardware columns
                matching_cols = [c for c in df.columns if c.upper() == hw_key.upper()]
                if matching_cols:
                    source = matching_cols[0]
                    found_cols.append(source)
                    if combined_series is None:
                        combined_series = df[source].copy()
                        logger.debug(f"Applying hardware override: Mapped '{source}' to internal key '{internal_key}'")
                    else:
                        # Coalesce: keep the highest priority value, fill gaps with lower priority
                        logger.debug(f"Applying hardware override: Coalescing '{source}' into internal key '{internal_key}'")
                        combined_series = combined_series.fillna(df[source])
            
            if combined_series is not None:
                df[internal_key] = combined_series
                # Remove original hardware columns to prevent redundancy, 
                # but keep the internal_key column if it was one of the sources.
                for col in found_cols:
                    if col != internal_key and col in df.columns:
                        df.drop(columns=[col], inplace=True)

        # --- Stage 2: Column Standardization ---
        # Normalize all column names to lowercase for consistent internal
        # processing, BEFORE default injection. We must also merge any
        # duplicate columns created by case variations (e.g. 'GAIN' and
        # 'gain') at this point.
        #
        # A8 in REMEDIATION_PLAN.md: default injection used to run first,
        # against the still-uppercase raw columns, and only *then* would
        # everything get lowercased and coalesced. That made a default's
        # survival depend on where its column landed relative to a
        # differently-cased genuine column once both were lowercased --
        # itself dependent on non-obvious pandas append-order behaviour.
        # Normalizing case first and injecting defaults only into columns
        # that are *still* genuinely absent afterwards removes that
        # implicit dependency entirely, regardless of whether the original
        # ordering was ever shown to misbehave on real data (it wasn't,
        # empirically, for the FITS/XISF uppercase-key convention this
        # pipeline actually sees -- but nothing enforced that assumption).
        logger.debug("Normalizing all column names to lowercase")
        df.columns = [c.lower() for c in df.columns]

        # Identify and merge duplicate columns
        if df.columns.duplicated().any():
            logger.debug("Merging duplicate columns")
            df = _coalesce_duplicate_columns(df)

        # --- Stage 3: Default Injection ---
        # For any core metadata still missing after extraction, overrides,
        # and case normalization, inject the user-defined fallback values.
        # Defaults are keyed uppercase in config.ini/AppConfig; lowercase
        # them here to match the now-normalized dataframe.
        for k, v in config.defaults.items():
            k_lower = k.lower()
            if k_lower not in df.columns:
                logger.debug(f"Default Injection: Key '{k}' not found, using default '{v}'")
                df[k_lower] = v

        # --- Stage 3b: Equipment Value Overrides ---
        # [equipmentoverrides] forces a literal display value into a column
        # for every row -- e.g. focname = 'ZWO EAF' when the header only
        # carries 'EAF' (GitHub #5). Runs after default injection so it wins
        # over both the found value and any default. Sentinel/blank entries
        # were already dropped by the loader.
        for k, v in config.equipment_overrides.items():
            k_lower = k.lower()
            logger.debug(f"Equipment Override: forcing '{k_lower}' = '{v}'")
            df[k_lower] = v

        # --- Stage 4: Initial Filtering ---
        # Drop 'MASTERLIGHT' frames.
        # We calculate exposures from individual subs; masters would double the total.
        itype_col = InternalColumns.IMAGE_TYPE
        if itype_col in df.columns:
            logger.debug("Performing initial image type filtering")
            df[itype_col] = df[itype_col].astype(str).str.upper()
            mask_drop = df[itype_col].str.contains('MASTERLIGHT', case=False, na=False) | \
                        df[itype_col].str.contains('MASTER LIGHT', case=False, na=False) | \
                        (df[itype_col] == 'NAN')
            df = df[~mask_drop].copy()

        # --- Stage 5: Master Preference Filtering ---
        # Execute preference before normalization to allow substring matching (FLAT vs MASTERFLAT)
        logger.debug("Executing master preference filtering")
        df = self._execute_master_preference(df)

        # --- Stage 6: IMAGETYP Normalization (Post-Preference) ---
        if itype_col in df.columns:
            logger.debug("Standardizing image type values")
            type_map = {
                'LIGHT': ImageType.LIGHT.value,
                'FLAT': ImageType.FLAT.value,
                'DARK': ImageType.DARK.value,
                'BIAS': ImageType.BIAS.value,
                'MASTERFLAT': ImageType.MASTER_FLAT.value,
                'MASTER FLAT': ImageType.MASTER_FLAT.value,
                'MASTERDARK': ImageType.MASTER_DARK.value,
                'MASTER DARK': ImageType.MASTER_DARK.value,
                'MASTERBIAS': ImageType.MASTER_BIAS.value,
                'MASTER BIAS': ImageType.MASTER_BIAS.value,
                'MASTERDARKFLAT': ImageType.MASTER_DARKFLAT.value,
                'DARKFLAT': ImageType.DARK_FLAT.value,
                'DARK FLAT': ImageType.DARK_FLAT.value
            }
            
            # Apply mappings (longer keywords first to prevent partial matches like 'DARK' matching 'DARKFLAT').
            #
            # Matches are evaluated against a frozen snapshot of the original
            # values, and a row is only ever assigned once. Previously each
            # mask was recomputed against df[itype_col] *after* prior
            # iterations had already mutated it, so a row correctly set to
            # 'MASTERDARK' by the 'MASTER DARK' keyword was then reprocessed
            # by the later, shorter 'DARK' keyword -- which matches
            # 'MASTERDARK' as a substring of its own output -- and clobbered
            # straight back down to plain 'DARK'. In practice this meant
            # every master calibration frame silently lost its master
            # designation, which made the master-preference check in
            # CalibrationMatcherStep.resolve_count() (it looks for a
            # 'MASTER' substring) permanently unreachable, risking
            # double-counted calibration frames whenever a master and its
            # own raw subs coexisted (A13 in REMEDIATION_PLAN.md).
            original_itype = df[itype_col].copy()
            assigned = pd.Series(False, index=df.index)
            for keyword, normalized in sorted(type_map.items(), key=lambda x: len(x[0]), reverse=True):
                mask = original_itype.str.contains(keyword, case=False, na=False) & ~assigned
                if mask.any():
                    logger.debug(f"Converted IMAGETYP keyword '{keyword}' to {normalized}")
                df.loc[mask, itype_col] = normalized
                assigned |= mask

        # --- Stage 7: Core Column Hardening ---
        # Ensure critical columns exist and are strictly typed.
        logger.debug("Reducing headers and hardening core column data types")

        def _configured_default(raw_key, fallback, cast=float):
            """Prefers the user's [defaults] value over the hardcoded fallback.

            Stage 3 above only ever consults config.defaults when a column is
            missing outright. That left a gap: the moment even one frame in a
            batch supplies a header, every *other* frame's blank in that same
            column reverts to the literal below, regardless of what the user
            configured -- even for fields [defaults] exists specifically to
            answer, like "where was this frame shot if it carries no GPS
            data?" A calibration frame with no SITELAT/SITELONG previously
            hardened to 0.0/0.0 -- the middle of the Gulf of Guinea -- instead
            of the observer's own configured site, discovered via the `ic405`
            dataset (250 raw DARKFLAT frames, none of them carrying GPS) in
            AstroBinUploaderRust's parity corpus.

            This is the same defaulting question Stage 3 already answers for
            a wholly-missing column, so it is resolved the same way: prefer
            config.defaults, and only fall back to the hardcoded literal when
            the config file does not define that key at all (as most of the
            table below does not -- BORTLE/SQM/FOCTEMP/CCD-TEMP/FOCRATIO/
            EXPOSURE/XBINNING all already agree with their config default and
            are left as plain literals; IMSCALE/NUMBER/darks/flats/flatDarks/
            bias have no config key to look up).
            """
            raw = config.defaults.get(raw_key)
            if raw is None:
                return fallback
            try:
                return cast(raw)
            except (TypeError, ValueError):
                return fallback

        core_columns = {
            InternalColumns.GAIN: _configured_default('GAIN', 0, int),
            InternalColumns.EGAIN: _configured_default('EGAIN', 1.0),
            InternalColumns.DURATION: 0.0,
            InternalColumns.SENSOR_COOLING: -10.0,
            InternalColumns.FOCAL_LENGTH: _configured_default('FOCALLEN', 500),
            InternalColumns.F_NUMBER: 5.0,
            InternalColumns.PIXEL_SIZE: _configured_default('XPIXSZ', 3.76),
            InternalColumns.SITE_LAT: _configured_default('SITELAT', 0.0),
            InternalColumns.SITE_LONG: _configured_default('SITELONG', 0.0),
            InternalColumns.BORTLE: 4.0,
            InternalColumns.MEAN_SQM: 21.0,
            InternalColumns.TEMPERATURE: 20.0,
            InternalColumns.TARGET: _configured_default('OBJECT', 'Unknown', str),
            InternalColumns.FILTER_NAME: 'No Filter',
            InternalColumns.SITE_NAME: 'Unknown Site',
            InternalColumns.BINNING: 1,
            InternalColumns.HFR: 1.0,
            InternalColumns.MEAN_FWHM: 0.0,
            InternalColumns.IMSCALE: 1.0,
            InternalColumns.NUMBER: 1,
            'darks': 0, 'flats': 0, 'flatDarks': 0, 'bias': 0
        }

        for col, default in core_columns.items():
            if col not in df.columns:
                # Initialize missing core columns with defaults
                df[col] = default
            else:
                # Type-cast existing columns to numeric and fill NaNs
                if col in [InternalColumns.EGAIN, 
                          InternalColumns.SENSOR_COOLING, InternalColumns.FOCAL_LENGTH, 
                          InternalColumns.F_NUMBER, InternalColumns.PIXEL_SIZE, 
                          InternalColumns.SITE_LAT, InternalColumns.SITE_LONG,
                          InternalColumns.BINNING, InternalColumns.BORTLE, 
                          InternalColumns.MEAN_SQM, InternalColumns.HFR,
                          InternalColumns.MEAN_FWHM, InternalColumns.IMSCALE]:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default).astype(float)
                elif col == InternalColumns.DURATION:
                    # Round duration to 2 decimal places for consistent grouping/matching
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default).round(2).astype(float)
                elif col == InternalColumns.NUMBER:
                    # Special handling for NUMBER: preserve existing counts from masters
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1).astype(int)
                elif col == InternalColumns.GAIN:
                    # Gain is strictly a linear integer (e.g. 100, 1, 0)
                    # We prioritize existing numeric GAIN from headers.
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default).round().astype(int)
                elif col == InternalColumns.SITE_NAME:
                    # Standardize Site Names as strings
                    df[col] = df[col].astype(str).replace('nan', default)
                elif col == InternalColumns.TARGET:
                    # A blank OBJECT header previously reached this point and
                    # was left NaN outright -- no branch handled it at all,
                    # unlike every other column here. Same string-column
                    # idiom as SITE_NAME just above.
                    df[col] = df[col].astype(str).replace('nan', default)

        logger.debug("Completed data type conversion and header normalization")
        state.processed_df = df
        return state

    def _execute_master_preference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final authority on calibration hierarchy.
        If a group of frames (same hardware axes) contains a MASTER,
        discard everything else in that group.
        """
        logger = logging.getLogger("AstroBinV2")
        itype_col = InternalColumns.IMAGE_TYPE

        # 1. Identify Calibration vs Light using original labels
        def is_cal(val):
            v = str(val).upper()
            return any(t in v for t in ['FLAT', 'DARK', 'BIAS']) and 'LIGHT' not in v

        cals_mask = df[itype_col].apply(is_cal)
        lights = df[~cals_mask].copy()
        cals = df[cals_mask].copy()
        
        if cals.empty: return df

        # 2. Grouping key must associate 'FLAT' with 'MASTER FLAT'
        def get_group_key(row):
            orig_itype = str(row[itype_col]).upper()
            # Normalize base type: 'MASTER FLAT' -> 'FLAT', 'DARK' -> 'DARK'
            # Stripping 'MASTER' and removing spaces ensures 'MASTER FLAT' groups with 'FLAT'
            base_type = orig_itype.replace('MASTER', '').replace(' ', '').strip()
            
            # Use raw numeric values for precise hardware grouping. This
            # runs before Stage 7's core-column hardening, so GAIN/EGAIN
            # may still be genuinely non-numeric or missing here.
            try:
                gain = int(round(float(row[InternalColumns.GAIN])))
            except (ValueError, TypeError) as e:
                logger.debug(
                    f"Master preference: unparseable GAIN for "
                    f"{row.get(InternalColumns.FILENAME, '<unknown file>')}, using 0 ({e})"
                )
                gain = 0

            try:
                # Reduce precision to 2 decimals to bridge master/raw EGAIN differences
                egain = f"{float(row[InternalColumns.EGAIN]):.2f}"
            except (ValueError, TypeError) as e:
                logger.debug(
                    f"Master preference: unparseable EGAIN for "
                    f"{row.get(InternalColumns.FILENAME, '<unknown file>')}, using 1.00 ({e})"
                )
                egain = "1.00"
                
            binning = str(row[InternalColumns.BINNING]).strip()
            
            # Filter normalization for grouping
            import re
            filter_val = row.get('filter', row.get('FILTER', 'No Filter'))
            filter_name = str(filter_val).lower().strip()
            # Remove common prefixes like 'filter_' or 'filter-'
            filter_name = re.sub(r'^filter[_-]', '', filter_name).strip()
            
            if base_type in ['DARK', 'BIAS']:
                try:
                    duration = f"{float(row[InternalColumns.DURATION]):.2f}"
                except (ValueError, TypeError) as e:
                    logger.debug(
                        f"Master preference: unparseable DURATION for "
                        f"{row.get(InternalColumns.FILENAME, '<unknown file>')}, using 0.00 ({e})"
                    )
                    duration = "0.00"
                return (base_type, gain, egain, binning, duration)
            else:
                return (base_type, gain, egain, binning, filter_name)

        cals['_group_key'] = cals.apply(get_group_key, axis=1)
        
        # 3. Master Preemption: Within each group, if a Master exists, keep ONLY one master.
        final_cals = []
        dropped_count = 0

        for group_key, group in cals.groupby('_group_key'):
            is_master_mask = group[itype_col].astype(str).str.upper().str.contains('MASTER', na=False)
            if is_master_mask.any():
                # KEEP ONLY one master frame for this hardware group, drop all raws.
                #
                # A10 in REMEDIATION_PLAN.md, per user decision: masters
                # should always be the latest available. There normally
                # shouldn't be more than one master per (type, gain, egain,
                # binning, duration/filter) group in the first place -- if
                # there is, prefer the one with the most recent DATE-OBS
                # rather than an arbitrary "first found" (which A9 made
                # deterministic, but deterministic isn't the same as
                # meaningful: it was really just picking whichever file the
                # sorted scan happened to reach first).
                masters = group[is_master_mask]
                if len(masters) > 1:
                    parsed_dates = pd.to_datetime(masters[InternalColumns.DATE_OBS], errors='coerce')
                    if parsed_dates.notna().any():
                        latest_idx = parsed_dates.idxmax()
                        chosen = masters.loc[[latest_idx]]
                        logger.debug(
                            f"Master Preference: {len(masters)} masters found for group "
                            f"{group_key}; kept the most recent (DATE-OBS={parsed_dates.loc[latest_idx]})."
                        )
                    else:
                        # No usable DATE-OBS on any candidate -- fall back
                        # to the first found under the deterministic scan
                        # order (A9), rather than erroring.
                        chosen = masters.iloc[[0]]
                        logger.debug(
                            f"Master Preference: {len(masters)} masters found for group "
                            f"{group_key} but none had a usable DATE-OBS; kept the first "
                            f"found under scan order."
                        )
                else:
                    chosen = masters
                dropped_count += len(group) - len(chosen)
                final_cals.append(chosen)
            else:
                # Keep all raw frames if no master exists
                final_cals.append(group)
        
        if dropped_count > 0:
            logger.debug(f"Master Preference Filter: Dropped {dropped_count} redundant raw/duplicate calibration frames.")
        
        if final_cals:
            cals = pd.concat(final_cals, ignore_index=True).drop(columns=['_group_key'])
        
        return pd.concat([lights, cals], ignore_index=True)
