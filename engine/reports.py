"""
Reporting Module - AstroBin Upload Utility v2.1.1

This module is responsible for transforming aggregated session data into a 
highly detailed, human-readable text report. It mirrors the high-quality 
formatting standards established in legacy versions, providing a clear 
overview of equipment, environmental conditions, and exposure statistics.

The generator supports multi-site sessions and complex mosaics by 
intelligently grouping frames by site, target, and image type.
"""

import logging
import pandas as pd
from typing import Tuple, Union
from datetime import datetime
from constants import ImageType, InternalColumns

def seconds_to_hms(seconds: Union[int, float], logger: logging.Logger, aligned: bool = False) -> str:
    """
    Converts a raw duration in seconds into a formatted HH:MM:SS string.

    Args:
        seconds (float): Total seconds to convert.
        logger (logging.Logger): Application logger for error handling.
        aligned (bool): If True, uses fixed-width padding for tabular display.

    Returns:
        str: Formatted string (e.g., "1 hrs 30 mins 15.00 secs").
    """
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = float(seconds % 60)
        
        # Aligned formatting is used within the ASCII tables for vertical consistency
        if aligned:
            return f"{hours:>6} hrs {minutes:>6} mins {secs:>6.2f} secs"
        return f"{hours} hrs {minutes} mins {secs:.2f} secs"
    except (TypeError, ValueError, OverflowError) as e:
        # 'logger' has been a parameter here, documented as being "for
        # error handling", since the function's signature was written, but
        # was never once used -- this fell back to a silent zero with no
        # trace regardless of what bad input caused it (B3 in
        # REMEDIATION_PLAN.md).
        logger.debug(f"seconds_to_hms could not format {seconds!r}: {e}")
        return "0 hrs 0 mins 0.00 secs"

def get_target_details(group: pd.DataFrame, logger: logging.Logger) -> str:
    """
    Identifies and formats the imaging target name from a group of frames.
    
    Includes specific logic for detecting mosaics based on 'Panel' keywords 
    in the target names.

    Args:
        group (pd.DataFrame): Dataframe containing a subset of Light frames.
        logger (logging.Logger): Application logger.

    Returns:
        str: Formatted target name string.
    """
    target_format = " Target: {}"
    if group.empty: return target_format.format("No target data")
    
    # Extract unique target names, dropping NaNs
    unique_targets = group[InternalColumns.TARGET].dropna().unique()
    
    # Mosaic Detection: If multiple targets contain 'Panel', summarize as a mosaic
    panels = [str(t) for t in unique_targets if 'Panel' in str(t)]
    if panels:
        # Extract the base name (e.g., 'M31' from 'M31 Panel 1')
        base_name = panels[0].split('Panel')[0].strip()
        return target_format.format(f"{base_name} {len(panels)} Panel Mosaic")
    
    # Default behavior: Return the first target found in the group
    return target_format.format(unique_targets[0] if len(unique_targets) > 0 else "Unknown")

def get_equipment_used(group: pd.DataFrame, df: pd.DataFrame, logger: logging.Logger) -> str:
    """
    Constructs a list of hardware and software used during the session.

    Args:
        group (pd.DataFrame): Subset of Light frames for hardware extraction.
        df (pd.DataFrame): Full session dataframe for software version extraction.
        logger (logging.Logger): Application logger.

    Returns:
        str: Multiline string containing the equipment list.
    """
    s = ["\nEquipment used:"]
    fmt = "\t{:<20}: {}"
    
    # Mapping of labels to internal column names
    items = {
        'Telescope': InternalColumns.TELESCOPE,
        'Camera': InternalColumns.CAMERA,
        'Filterwheel': InternalColumns.FILTER_WHEEL,
        'Focuser': InternalColumns.FOCUSER,
        'Rotator': InternalColumns.ROTATOR_NAME
    }
    
    # Extract hardware names from the first light frame (assumes static hardware per target)
    for label, col in items.items():
        if col in group.columns:
            val = group[col].iloc[0]
            if pd.notna(val) and str(val).lower() not in ['none', 'nan', '']:
                s.append(fmt.format(label, val))
            
    # Software version extraction: Collect all unique software strings found across all files
    sw_set = set(group[InternalColumns.SWCREATE].dropna().unique())
    sw_set.update(df[InternalColumns.SWCREATE].dropna().unique())
    
    if sw_set:
        # Sort so that the 'main' software usually appears first
        sw_list = sorted(list(sw_set), reverse=True) 
        s.append(fmt.format("Capture software", sw_list.pop(0)))
        # List additional software modules indented underneath
        for item in sw_list:
            s.append(fmt.format("", item))
            
    return "\n".join(s) + "\n"

def get_observation_period(group: pd.DataFrame, logger: logging.Logger) -> str:
    """
    Summarizes the dates, session counts, and temperature ranges.

    Args:
        group (pd.DataFrame): Subset of Light frames.
        logger (logging.Logger): Application logger.

    Returns:
        str: Formatted observation period summary.
    """
    s = ["\nObservation period:"]
    fmt = "\t{:<25}: {}"
    
    # Extract pre-calculated session statistics from the broadcasted columns
    start = group[InternalColumns.START_DATE].iloc[0] if InternalColumns.START_DATE in group.columns else "N/A"
    end = group[InternalColumns.END_DATE].iloc[0] if InternalColumns.END_DATE in group.columns else "N/A"
    days = group[InternalColumns.NUM_DAYS].iloc[0] if InternalColumns.NUM_DAYS in group.columns else 0
    sessions = group[InternalColumns.SESSIONS].iloc[0] if InternalColumns.SESSIONS in group.columns else 0
    
    s.append(fmt.format("Start date", start))
    s.append(fmt.format("End date", end))
    s.append(fmt.format("Days", int(days)))
    s.append(fmt.format("Observation sessions", int(sessions)))
    
    # Temperature Statistics
    if InternalColumns.TEMP_MIN in group.columns:
        s.append(fmt.format("Min temperature", f"{group[InternalColumns.TEMP_MIN].min():.1f}\u00B0C"))
        s.append(fmt.format("Max temperature", f"{group[InternalColumns.TEMP_MAX].max():.1f}\u00B0C"))
        s.append(fmt.format("Mean temperature", f"{group[InternalColumns.TEMPERATURE].mean():.1f}\u00B0C"))
    
    return "\n".join(s) + "\n"

def format_image_type_table(group: pd.DataFrame, imagetype: str, logger: logging.Logger, light_filters: set = None, light_gains: set = None) -> Tuple[str, float]:
    """
    Constructs an ASCII table summarizing frame counts and exposures for a specific type.
    
    For Light frames, data is grouped by target and filter. 
    For Calibration frames, data is consolidated by filter and gain.

    Args:
        group (pd.DataFrame): The full site-level dataframe.
        imagetype (str): The specific ImageType to format (e.g., 'LIGHT', 'FLAT').
        logger (logging.Logger): Application logger.
        light_filters (set, optional): Set of filter names used in Light frames.
        light_gains (set, optional): Set of linear Gain values used in Light frames.

    Returns:
        Tuple[str, float]: (The formatted ASCII table, Total exposure time in seconds).
    """
    lines = []
    total_exposure = 0.0

    # The caller (generate_full_summary) already scopes `group` to exactly
    # the row types belonging to this category via isin(matches) -- e.g.
    # both 'DARK' and 'MASTERDARK' for the dark category -- so no further
    # type filtering happens here. This used to re-filter for an exact match
    # against `imagetype` (the category's base type only), which silently
    # dropped every row whenever a group contained *only* its MASTER
    # variant and no base-type row -- invisible until A13 in
    # REMEDIATION_PLAN.md stopped IMAGETYP normalization from collapsing
    # every master label down to its base type.
    image_group = group.copy()

    # Calibration Filtering: Only show Calibration for (Filter and/or Gain) that were actually used for Lights
    # This prevents clutter from calibration files that don't belong to the current session.
    if light_filters is not None and "FLAT" in imagetype.upper():
        image_group = image_group[image_group[InternalColumns.FILTER_NAME].astype(str).str.lower().isin(light_filters)]
    
    if light_gains is not None and imagetype != ImageType.LIGHT.value:
        # Strictly exclude calibration frames whose linear Gain doesn't match any Light frame Gain
        image_group = image_group[image_group[InternalColumns.GAIN_MATCH].isin(light_gains)]

    if image_group.empty: return "", 0.0

    # Common grouping keys for the summary table
    table_group_keys = [InternalColumns.FILTER_NAME, InternalColumns.GAIN_MATCH, InternalColumns.DURATION]

    if imagetype == ImageType.LIGHT.value:
        lines.append(f"\n {imagetype}S:")
        # Lights are grouped by Target first
        for target, t_group in image_group.groupby(InternalColumns.TARGET, observed=True):
            lines.append(f" Target: {target}\n")
            header = " {:<8} {:<8} {:<8} {:<12} {:<12} {:<12} {:<12} {:<15} {:<15}"
            lines.append(header.format("Filter", "Frames", "Gain", "Egain", "Mean FWHM", "Sensor Temp", "Mean Temp", "Exposure", "Total Exposure"))
            
            t_exposure_target = 0.0
            
            # Aggregate stats across multiple sessions/nights for this specific target
            summary_agg = t_group.groupby(table_group_keys, observed=True).agg({
                InternalColumns.NUMBER: 'sum',
                InternalColumns.GAIN: 'first',
                InternalColumns.EGAIN: 'mean',
                InternalColumns.MEAN_FWHM: 'mean',
                InternalColumns.SENSOR_COOLING: 'mean',
                InternalColumns.TEMPERATURE: 'mean'
            }).reset_index()

            for _, row in summary_agg.iterrows():
                row_total_exposure = row[InternalColumns.NUMBER] * row[InternalColumns.DURATION]
                t_exposure_target += row_total_exposure
                
                # Format gain for display (using linear integer GAIN)
                gain_val = row[InternalColumns.GAIN]
                gain_str = str(int(round(float(gain_val)))) if pd.notna(gain_val) else "N/A"
                egain_str = f"{float(row[InternalColumns.EGAIN]):.2f} e/ADU"
                
                lines.append(header.format(
                    str(row[InternalColumns.FILTER_NAME]), int(row[InternalColumns.NUMBER]), gain_str, egain_str,
                    f"{row[InternalColumns.MEAN_FWHM]:.2f} arcsec", f"{row[InternalColumns.SENSOR_COOLING]:.1f}\u00B0C", f"{row[InternalColumns.TEMPERATURE]:.1f}\u00B0C",
                    f"{row[InternalColumns.DURATION]:.2f} secs", seconds_to_hms(row_total_exposure, logger, aligned=True)
                ))
            lines.append(f"\n Exposure time for {target}: {seconds_to_hms(t_exposure_target, logger)}\n")
            total_exposure += t_exposure_target
    else:
        # Calibration Frames: Consolidate by filter/gain/exposure (no target grouping)
        #
        # The label reflects what is actually in this table, not a blind
        # lookup. This function's docstring cites "v1.4.7 standards", but the
        # label_map this replaced labelled *every* calibration section
        # MASTERxxx unconditionally -- including sessions built entirely from
        # raw, uncalibrated DARK/FLAT/BIAS frames with no master anywhere in
        # sight. The actual v1.4.7 code (utils.py::process_image_type)
        # labelled each section by its literal IMAGETYP: plain "DARK:" for
        # raw darks, "MASTERDARK:" only when the frames really were masters.
        # That got lost somewhere between v1.4.7 and here; the comment kept
        # citing v1.4.7 while the behaviour stopped matching it.
        #
        # v2.1.1 deliberately consolidates a class's raw and MASTER variants
        # into one table (A13/the comment above, "group raw and master types
        # together for the report layout") -- that grouping is correct and
        # is not what this fixes. Only the header text was wrong. A table
        # containing only raw frames now says so; MASTER only when every row
        # actually is one; a genuinely mixed table (a master used for one
        # gain, raw frames surviving for another because no master covered
        # it) favours MASTER, since it tells the reader master calibration
        # was used for at least part of the data -- the safer thing to
        # under-claim toward is "some of this is raw", not the reverse.
        # `imagetype` here is always the category's base (raw) type --
        # generate_full_summary passes type_tuple[0], and every tuple is
        # (raw, MASTER_raw) -- so pluralising it is what needs the BIAS
        # exception the original label_map already encoded ('BIAS' ->
        # 'MASTERBIAS', not 'MASTERBIASS'; every other type takes a plain S).
        base = imagetype.upper()
        plain_label = base if base == 'BIAS' else f"{base}S"
        actual_types = set(image_group[InternalColumns.IMAGE_TYPE].astype(str).str.upper())
        has_master = any(t.startswith('MASTER') for t in actual_types)
        display_label = f"MASTER{plain_label}" if has_master else plain_label

        lines.append(f"\n {display_label}:\n")
        header = " {:<10} {:<8} {:<10} {:<15} {:<12} {:<15}"
        lines.append(header.format("Filter", "Frames", "Gain", "Egain", "Exposure", "Total Exposure"))

        # Darks and Bias are physically filter-independent, and
        # calibration.py's own candidate matching already reflects that --
        # neither dark_candidates nor bias_candidates constrain on filter.
        # This table's grouping previously always included FILTER_NAME
        # regardless of calibration type, found live against real
        # unprocessed calibration data: some capture software stamps
        # whatever filter happens to be mounted into every dark/bias
        # frame's header too, not just lights'. In that dataset every
        # dark/bias file happened to carry the same filter tag, so the
        # only visible symptom was a misleading label (e.g. a MASTERDARKS
        # row showing "Ha" instead of blank) -- but a filter change
        # between calibration sessions would fragment one logical
        # dark/bias set into multiple rows here, each understating its
        # own Frames count, even though calibration.py's actual matching
        # (and so the acquisition CSV's darks/bias columns) is and was
        # unaffected. FlatDarks does constrain on filter in calibration.py
        # (matching Flats), so it keeps grouping by it here too.
        filter_matters_for_type = 'FLAT' in imagetype.upper()  # covers FLAT and DARKFLAT
        cal_group_keys = (
            [InternalColumns.FILTER_NAME, InternalColumns.GAIN_MATCH, InternalColumns.DURATION]
            if filter_matters_for_type
            else [InternalColumns.GAIN_MATCH, InternalColumns.DURATION]
        )

        summary_agg = image_group.groupby(cal_group_keys, observed=True).agg({
            InternalColumns.NUMBER: 'sum',
            InternalColumns.GAIN: 'first',
            InternalColumns.EGAIN: 'mean'
        }).reset_index()

        for _, row in summary_agg.iterrows():
            row_total_exposure = row[InternalColumns.NUMBER] * row[InternalColumns.DURATION]
            total_exposure += row_total_exposure

            # Format gain for display (using linear integer GAIN)
            gain_val = row[InternalColumns.GAIN]
            gain_str = str(int(round(float(gain_val)))) if pd.notna(gain_val) else "N/A"
            egain_str = f"{float(row[InternalColumns.EGAIN]):.2f} e/ADU"

            if filter_matters_for_type:
                # Blank if there is no real filter. Two distinct sentinels
                # mean "no filter" here: 'No Filter' (the configured
                # [defaults] value, injected only when the FITS/XISF file
                # never had a FILTER column at all) and 'None'
                # (AggregationStep's null-safety fill, applied per-cell
                # when the column exists but this row's value was missing).
                filter_val = str(row[InternalColumns.FILTER_NAME])
                if filter_val in ('No Filter', 'None'): filter_val = ""
            else:
                # Not part of the group key for this calibration type --
                # always blank, regardless of what any individual frame's
                # header happened to record.
                filter_val = ""

            lines.append(header.format(
                filter_val, int(row[InternalColumns.NUMBER]), gain_str, egain_str,
                f"{row[InternalColumns.DURATION]:.2f} secs", seconds_to_hms(row_total_exposure, logger, aligned=True)
            ))
            
    return "\n".join(lines), total_exposure

def generate_full_summary(df: pd.DataFrame, logger: logging.Logger, total_scanned: int) -> str:
    """
    Orchestrates the generation of the full multi-site session report.
    
    This is the primary entry point for the reporting engine. It iterates through 
    sites, generates equipment and observation summaries, and builds detail 
    tables for every image type found.

    Args:
        df (pd.DataFrame): The fully aggregated session dataframe.
        logger (logging.Logger): Application logger.
        total_scanned (int): Total count of raw files identified on disk.

    Returns:
        str: The complete, formatted text report.
    """
    if df.empty: return "No data available for reporting."
    
    report = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report.append(f"Observation session summary\nGenerated {now}")
    
    # Iterate through each Site found in the session
    for site, site_group in df.groupby(InternalColumns.SITE_NAME, observed=True):
        lights = site_group[site_group[InternalColumns.IMAGE_TYPE] == ImageType.LIGHT.value]
        if lights.empty: continue # Skip sites that only have calibration frames
        
        # Determine unique filters and gains for the light frames to filter irrelevant calibration data
        light_filters = set(lights[InternalColumns.FILTER_NAME].astype(str).str.lower().unique())
        light_gains = set(lights[InternalColumns.GAIN_MATCH].unique())
        
        # 1. Target and Site Metadata
        report.append(get_target_details(lights, logger))
        report.append(f"\nSite: {site}")
        report.append(f"\tLatitude: {site_group[InternalColumns.SITE_LAT].iloc[0]:.4f}\u00B0")
        report.append(f"\tLongitude: {site_group[InternalColumns.SITE_LONG].iloc[0]:.4f}\u00B0")
        report.append(f"\tBortle scale: {site_group[InternalColumns.BORTLE].iloc[0]:.1f}")
        report.append(f"\tSQM: {site_group[InternalColumns.MEAN_SQM].iloc[0]:.2f} mag/arcsec²")
        
        # 2. Hardware and Temporal Summaries
        report.append(get_equipment_used(lights, df, logger))
        report.append(get_observation_period(lights, logger))
        
        # 3. Formatted Data Tables (Ordered by importance)
        # We group raw and master types together for the report layout
        order = [
            (ImageType.LIGHT.value,), 
            (ImageType.FLAT.value, ImageType.MASTER_FLAT.value), 
            (ImageType.BIAS.value, ImageType.MASTER_BIAS.value),
            (ImageType.DARK.value, ImageType.MASTER_DARK.value),
            (ImageType.DARK_FLAT.value, ImageType.MASTER_DARKFLAT.value)
        ]
        
        unique_itypes = site_group[InternalColumns.IMAGE_TYPE].unique()
        processed_types = set()

        for type_tuple in order:
            # Find all types in the current group that belong to this category (e.g., DARK + MASTERDARK)
            matches = [u for u in unique_itypes if u in type_tuple]
            if matches:
                # Filter the site_group to include only these specific types for the table
                category_group = site_group[site_group[InternalColumns.IMAGE_TYPE].isin(matches)]
                
                # We use the primary type in the tuple for the formatting logic (determines the MASTER label)
                primary_type = type_tuple[0]
                
                table, exp = format_image_type_table(category_group, primary_type, logger, 
                                                   light_filters=light_filters, light_gains=light_gains)
                if table:
                    report.append(table)
                    report.append(f"\nTotal {primary_type} Exposure Time: {seconds_to_hms(exp, logger)}\n")
                
                processed_types.update(matches)
                    
    # Append global processing statistics
    report.append(f"\n Total number of images processed: {total_scanned}\n")
    return "\n".join(report)