"""
Header Extractor Module - AstroBin Upload Utility v2.1.0

This module manages the high-speed extraction of metadata from multiple file 
formats including FITS, XISF, and CSV. It is optimized for large image sets 
by utilizing multi-process parallelism to bypass the Python GIL during 
compute-intensive XML and binary parsing.

Key Features:
- **Parallel Processing**: Uses ProcessPoolExecutor for concurrent file reads.
- **XISF Support**: Native parsing of PixInsight's XML-based header format.
- **Deep Inspection**: Extracts sub-exposure counts from Master frames by 
  inspecting PixInsight processing history.
"""

import os
import logging
import pandas as pd
from typing import List, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from astropy.io import fits
import struct
import xml.etree.ElementTree as ET
from constants import FITSKeywords


def _init_worker_logging(log_filepath: Optional[str], level: int):
    """
    Configure logging inside a spawned worker process.

    On fork-based platforms (Linux, the default here), a worker process
    inherits a full copy of the parent's already-configured logger --
    including its handlers -- at fork time, so this is a harmless no-op
    (the `if not worker_logger.handlers` guard skips re-adding). On
    spawn-based platforms (macOS default since Python 3.8, Windows always),
    a worker starts a fresh interpreter with no logging configured at all,
    so every per-file parse error logged from inside extract_single_file
    was silently dropped -- not printed, not written to the log file,
    gone (B5 in REMEDIATION_PLAN.md). This runs once per worker via
    ProcessPoolExecutor's `initializer`.
    """
    if not log_filepath:
        return
    worker_logger = logging.getLogger("AstroBinV2")
    if not worker_logger.handlers:
        handler = logging.FileHandler(log_filepath, encoding='utf-8')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(funcName)s - Line: %(lineno)d - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        worker_logger.addHandler(handler)
        worker_logger.setLevel(level)


class HeaderExtractor:
    """
    Orchestrates the discovery and parsing of astronomical metadata.
    """
    def __init__(self, logger: logging.Logger, config: Any):
        """
        Initializes the extractor.

        Args:
            logger (logging.Logger): Active application logger.
            config (Any): Application configuration settings.
        """
        self.logger = logger
        self.config = config

    def extract_from_directories(self, paths: List[str]) -> pd.DataFrame:
        """
        Recursively scans directories and reads headers in parallel.
        
        This method identifies all valid astronomical files and distributes 
        the parsing workload across available CPU cores.

        Args:
            paths (List[str]): Directory paths to scan.
            
        Returns:
            pd.DataFrame: A DataFrame containing raw metadata from all files.
        """
        file_paths = []
        for path in paths:
            self.logger.info(f"Scanning directory: {path}")
            for root, _, files in os.walk(path, followlinks=True):
                for file in files:
                    if file.lower().endswith(('.fits', '.fit', '.fts', '.xisf')):
                        file_paths.append(os.path.join(root, file))

        # Deterministic dispatch order. os.walk's per-directory file order is
        # OS/filesystem-dependent, and without this sort the final row order
        # of raw_df would vary between otherwise-identical runs (see A9 in
        # REMEDIATION_PLAN.md). Every downstream 'first wins' resolution --
        # deduplication's survivor pick, the master-preference filters, the
        # agg('first') columns in AggregationStep, and every .iloc[0] read in
        # reports.py -- depends on this being stable.
        file_paths.sort()

        total = len(file_paths)
        results: Dict[str, Any] = {}

        # Find the main logger's log file, if any, so worker processes can
        # be configured to write to the same file (B5 in
        # REMEDIATION_PLAN.md -- see _init_worker_logging above).
        log_filepath = None
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler):
                log_filepath = h.baseFilename
                break

        # Parallel Execution: Utilize multiple processes for XML/FITS parsing
        # This is significantly faster for XISF files which involve large XML blocks.
        # Completion order under as_completed() is nondeterministic, so results
        # are keyed by their originating path and reassembled in the sorted
        # dispatch order below rather than appended in completion order.
        with ProcessPoolExecutor(
            initializer=_init_worker_logging,
            initargs=(log_filepath, self.logger.level)
        ) as executor:
            futures = {executor.submit(self.extract_single_file, fp): fp for fp in file_paths}
            for i, future in enumerate(as_completed(futures), 1):
                fp = futures[future]
                res = future.result()
                if res:
                    results[fp] = res
                # Real-time console progress update
                print(f"\rScanning files: {i} of {total}...", end="", flush=True)

        print("\n") # Ensure next console output starts on a new line
        headers = [results[fp] for fp in file_paths if fp in results]
        self.logger.info(f"Extraction complete. {len(headers)} valid headers retrieved.")
        return pd.DataFrame(headers)

    def extract_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Loads metadata from a diagnostic CSV file.
        
        Used for re-running the pipeline on previously extracted data 
        without re-scanning the disk.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Metadata with normalized uppercase columns.
        """
        self.logger.info(f"Injecting metadata from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        # Normalize columns to uppercase to match FITS standards
        df.columns = [c.upper() for c in df.columns]
        return df

    def extract_single_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Worker function: Identifies the file format and parses its metadata.
        
        Args:
            filepath (str): Absolute path to the file.

        Returns:
            Optional[Dict[str, Any]]: Dictionary of header keywords, or None on failure.
        """
        logger = logging.getLogger("AstroBinV2")
        logger.debug(f"Processing headers for file: {os.path.basename(filepath)}")
        try:
            if filepath.lower().endswith(('.fits', '.fit', '.fts')):
                hdr = self._read_fits(filepath)
                logger.debug(f"Successfully read FITS header from {filepath}")
            elif filepath.lower().endswith('.xisf'):
                hdr = self._read_xisf(filepath)
                logger.debug(f"Successfully read XISF header from {filepath}")
            else:
                logger.warning(f"Unsupported file format: {filepath}")
                return None
            
            # Post-parsing cleanup: Strip quotes often found in raw FITS string values
            cleaned_hdr = {k: v.strip("'").strip('"') if isinstance(v, str) else v for k, v in hdr.items()}

            # Absolute source path, distinct from FILENAME (basename only).
            # Lets DeduplicateStep key on directory as well as filename, so
            # identically-named captures from different sessions/nights don't
            # collapse into one (A2 in REMEDIATION_PLAN.md). Set after the
            # quote-strip above since it is not raw header text.
            cleaned_hdr[FITSKeywords.SOURCE_PATH] = os.path.abspath(filepath)

            # Horizontal Header Printing (Essential requirement for DEBUG mode)
            logger.debug(f"Recovered Header: {cleaned_hdr}")
            
            return cleaned_hdr
        except Exception as e:
            logger.error(f"Error parsing headers for {filepath}: {str(e)}")
            # Silent failure for individual files to prevent pipeline crashing
            return None

    def _read_fits(self, filepath: str) -> Dict[str, Any]:
        """Reads a standard FITS file header using Astropy."""
        with fits.open(filepath) as hdul:
            # Select the first HDU that actually carries the metadata this
            # pipeline reads, rather than unconditionally using HDU 0.
            # Compressed images (.fits.fz / CompImageHDU) commonly store
            # the real header -- IMAGETYP, EXPOSURE, DATE-OBS, etc. -- on
            # the first image extension, leaving the primary HDU with only
            # structural boilerplate (SIMPLE/BITPIX/NAXIS/EXTEND).
            # Confirmed empirically: a CompImageHDU's IMAGETYP/EXPOSURE
            # land in HDU 1, absent from HDU 0 entirely (A7 in
            # REMEDIATION_PLAN.md). Falls back to HDU 0 if no HDU carries
            # IMAGETYP, preserving today's behaviour for files that
            # legitimately rely on [defaults] for it.
            source_hdu = hdul[0]
            for candidate in hdul:
                if FITSKeywords.IMAGE_TYPE in candidate.header:
                    source_hdu = candidate
                    break

            # Convert header object to a standard Python dictionary
            hdr = dict(source_hdu.header)
            hdr[FITSKeywords.FILENAME] = os.path.basename(filepath)
            # Identify if this is a Master frame with multiple sub-exposures
            hdr[FITSKeywords.NUMBER] = self._get_fit_number(hdr)
            return hdr

    def _read_xisf(self, filepath: str) -> Dict[str, Any]:
        """
        Parses the XML header of a PixInsight XISF file.
        
        Directly reads the XML block from the binary file to avoid 
        loading large image data into memory.
        """
        with open(filepath, 'rb') as f:
            f.read(8) # Skip 'XISF0100' signature
            # Read the 4-byte little-endian length of the XML header
            length = struct.unpack('<I', f.read(4))[0]
            f.read(4) # Skip reserved block
            # Decode the XML block
            xml_str = f.read(length).decode('utf-8', errors='ignore')
            
        root = ET.fromstring(xml_str)
        ns = {'xisf': 'http://www.pixinsight.com/xisf'}
        
        # Collect all FITSKeyword elements into a flat dictionary
        hdr = {kw.get('name'): kw.get('value') for kw in root.findall('.//xisf:FITSKeyword', ns)}
        hdr[FITSKeywords.FILENAME] = os.path.basename(filepath)
        
        # Deep Property Extraction: Look for Gain in PixInsight-specific properties 
        # if it was missing from the standard FITSKeywords.
        if FITSKeywords.GAIN not in hdr:
            gain_prop = root.find(".//xisf:Property[@id='instrument:gain']", ns)
            if gain_prop is not None:
                raw_gain = gain_prop.text
                try:
                    # Smart Extraction: If gain is a decimal < 1, it's likely EGAIN signature
                    val = float(raw_gain)
                    if 0 < val < 1.0:
                        hdr[FITSKeywords.EGAIN] = raw_gain
                    else:
                        hdr[FITSKeywords.GAIN] = raw_gain
                except ValueError:
                    hdr[FITSKeywords.GAIN] = raw_gain
        
        # Filename Fallback: If Gain is still missing (or was a decimal assigned to EGAIN), 
        # try to extract the true linear integer from the filename.
        if FITSKeywords.GAIN not in hdr or str(hdr.get(FITSKeywords.GAIN)).strip() in ['', 'nan', 'None']:
            import re
            fname = os.path.basename(filepath)
            # Match patterns like GAIN-100, gain_100, Gain100
            match = re.search(r'GAIN[_-]?(\d+)', fname, re.IGNORECASE)
            if match:
                hdr[FITSKeywords.GAIN] = match.group(1)

        # Filename Fallback for FILTER
        if FITSKeywords.FILTER not in hdr or str(hdr.get(FITSKeywords.FILTER)).strip() in ['', 'nan', 'None']:
            import re
            fname = os.path.basename(filepath)
            # Match patterns like FILTER-Ha, Filter_OIII, etc.
            match = re.search(r'FILTER[_-]([^_.]+)', fname, re.IGNORECASE)
            if match:
                hdr[FITSKeywords.FILTER] = match.group(1)
        
        # Master Sub-exposure Detection:
        # PixInsight Master frames store the integration count in the ProcessingHistory property.
        hdr[FITSKeywords.NUMBER] = 1
        prop = root.find(".//xisf:Property[@id='PixInsight:ProcessingHistory']", ns)
        if prop is not None and prop.text:
            try:
                hist_root = ET.fromstring(prop.text)
                table = hist_root.find(".//table[@id='images']")
                if table is not None:
                    hdr[FITSKeywords.NUMBER] = int(table.get('rows', 1))
            except (ET.ParseError, ValueError, TypeError) as e:
                # Malformed ProcessingHistory XML or a non-numeric 'rows'
                # attribute -- fall through to the fallback below rather
                # than silently leaving NUMBER at 1 with no trace (B3 in
                # REMEDIATION_PLAN.md).
                logging.getLogger("AstroBinV2").debug(
                    f"Could not parse ProcessingHistory NUMBER for {filepath}: {e}"
                )
        
        # Fallback: If NUMBER is still 1, search FITS comments/history for ImageIntegration count
        if hdr[FITSKeywords.NUMBER] == 1:
            for kw in root.findall('.//xisf:FITSKeyword', ns):
                name = kw.get('name')
                comment = kw.get('comment', '')
                if name in ['COMMENT', 'HISTORY'] and 'ImageIntegration.numberOfImages:' in comment:
                    try:
                        hdr[FITSKeywords.NUMBER] = int(comment.split(':')[-1].strip())
                        break
                    except (ValueError, TypeError) as e:
                        # Comment text matched but the trailing token
                        # wasn't a plain integer -- keep NUMBER at its
                        # current fallback rather than silently discarding
                        # the mismatch with no trace (B3 in
                        # REMEDIATION_PLAN.md).
                        logging.getLogger("AstroBinV2").debug(
                            f"Could not parse numberOfImages comment for {filepath}: {comment!r} ({e})"
                        )
            
        return hdr

    def _get_fit_number(self, hdr: Dict[str, Any]) -> int:
        """
        Scans FITS HISTORY for sub-exposure counts (PixInsight specific).
        
        When PixInsight creates a Master frame, it stores the 'numberOfImages' 
        in a HISTORY card which we can parse to get the true exposure count.
        """
        history = hdr.get('HISTORY', [])
        if isinstance(history, str): history = [history]
        for line in history:
            if 'ImageIntegration.numberOfImages:' in line:
                try:
                    return int(line.split()[-1])
                except (ValueError, TypeError) as e:
                    # Line matched but the trailing token wasn't a plain
                    # integer -- fall through to the default of 1 rather
                    # than silently discarding the mismatch with no trace
                    # (B3 in REMEDIATION_PLAN.md).
                    logging.getLogger("AstroBinV2").debug(
                        f"Could not parse numberOfImages HISTORY line for "
                        f"{hdr.get(FITSKeywords.FILENAME, '<unknown file>')}: {line!r} ({e})"
                    )
        return 1