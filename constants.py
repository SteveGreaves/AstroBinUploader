"""
Centralized Constants Module - AstroBin Upload Utility v2.1.0

This module serves as the single source of truth for all literal strings, 
column names, and configuration keys used throughout the application. 
By centralizing these values, we ensure consistency across the extraction, 
transformation, and export phases of the pipeline.
"""

from enum import Enum

class FITSKeywords:
    """
    Standard FITS Header Keywords.
    
    These constants represent the keys used when searching for metadata 
    within FITS/XISF files. They are typically uppercase as per the 
    FITS standard.
    """
    IMAGE_TYPE = 'IMAGETYP'    # Type of frame (LIGHT, DARK, etc.)
    EXPOSURE = 'EXPOSURE'      # Exposure time in seconds
    DATE_OBS = 'DATE-OBS'      # Observation timestamp
    XBINNING = 'XBINNING'      # Binning factor (e.g., 1x1, 2x2)
    GAIN = 'GAIN'              # Camera gain (can be unitless or dB)
    EGAIN = 'EGAIN'            # Electronic gain (e/ADU)
    INSTRUMENT = 'INSTRUME'    # Camera name
    TELESCOPE = 'TELESCOP'     # Telescope description
    FOCUSER = 'FOCNAME'        # Focuser hardware name
    FILTER_WHEEL = 'FWHEEL'    # Filter wheel name
    ROTATOR_NAME = 'ROTNAME'   # Rotator name
    ROTATOR_ANGLE = 'ROTANTANG' # Rotator mechanical angle
    PIXEL_SIZE = 'XPIXSZ'      # Pixel size in microns
    CCD_TEMP = 'CCD-TEMP'      # Sensor temperature in Celsius
    FOCAL_LENGTH = 'FOCALLEN'  # Optical focal length in mm
    FOCAL_RATIO = 'FOCRATIO'   # Optical f-number (e.g., 5.0)
    SITE = 'SITE'              # Geographical site name
    SITE_LAT = 'SITELAT'       # Site latitude (decimal degrees)
    SITE_LONG = 'SITELONG'     # Site longitude (decimal degrees)
    BORTLE = 'BORTLE'          # Bortle Scale (1-9)
    SQM = 'SQM'                # Sky Quality Meter reading (mag/arcsec^2)
    FILTER = 'FILTER'          # Active filter name
    OBJECT = 'OBJECT'          # Imaging target name
    FOCUSER_TEMP = 'FOCTEMP'   # Ambient/Focuser temperature
    HFR = 'HFR'                # Half Flux Radius (star size measure)
    FWHM = 'FWHM'              # Full Width at Half Maximum
    SWCREATE = 'SWCREATE'      # Capture software name
    FILENAME = 'FILENAME'      # Original filename for traceability
    NUMBER = 'NUMBER'          # Count of sub-exposures (for Master frames)
    IMSCALE = 'IMSCALE'        # Image scale (arcsec/pixel)
    SOURCE_PATH = 'SOURCE_PATH' # Absolute filesystem path, for directory-aware
                                 # deduplication (A2 in REMEDIATION_PLAN.md).
                                 # Not a real FITS/XISF keyword; synthesized by
                                 # the extractor. Absent on CSVs captured by an
                                 # older version -- see extract_from_csv.

class ConfigSections:
    """
    INI Configuration Section Names.
    
    Identifies the primary blocks within config.ini to prevent string 
    fragmentation in the ConfigLoader.
    """
    DEFAULTS = 'defaults'      # Standard fallback values
    OVERRIDE = 'override'      # User-defined keyword remapping
    EQUIPMENT_OVERRIDES = 'equipmentoverrides'  # User-defined value replacement
    FILTERS = 'filters'        # AstroBin filter code database
    SITES = 'sites'            # Local site coordinates database

class InternalColumns:
    """
    Internal Normalized Column Names.
    
    The pipeline converts all raw FITS/CSV headers into these lowercase 
    identifiers. This isolation layer allows the processing logic to 
    remain agnostic of the source file's naming conventions.
    """
    IMAGE_TYPE = 'imagetyp' 
    DURATION = 'exposure'
    BINNING = 'xbinning'
    SENSOR_COOLING = 'ccd-temp'
    MEAN_FWHM = 'fwhm'
    F_NUMBER = 'focratio'
    TEMPERATURE = 'foctemp'
    FOCUSER = 'focname'
    FILTER_WHEEL = 'fwheel'
    TELESCOPE = 'telescop'
    CAMERA = 'instrume'
    MEAN_SQM = 'sqm'
    FOCAL_LENGTH = 'focallen'
    PIXEL_SIZE = 'xpixsz'
    TARGET = 'object'
    SESSIONS = 'sessions'
    START_DATE = 'start_date'
    END_DATE = 'end_date'
    NUM_DAYS = 'num_days'
    SITE_LAT = 'sitelat'
    SITE_LONG = 'sitelong'
    BORTLE = 'bortle'
    ROTATOR_NAME = 'rotname'
    ROTATOR_ANGLE = 'rotantang'
    FILENAME = 'filename'
    SOURCE_PATH = 'source_path'
    NUMBER = 'number'
    DATE_OBS = 'date-obs'
    SITE_NAME = 'site'
    FILTER_NAME = 'filter'
    HFR = 'hfr'
    IMSCALE = 'imscale'
    GAIN = 'gain'
    EGAIN = 'egain'
    SWCREATE = 'swcreate'
    TEMP_MIN = 'temp_min'
    TEMP_MAX = 'temp_max'
    GAIN_MATCH = 'gain_match'  # Used for Integer Gain Handshake

class ImageType(str, Enum):
    """
    Normalized Image Type enumeration.
    
    Standardizes the chaotic variety of IMAGETYP values found in the 
    wild (e.g., 'Light Frame', 'light', 'LIGHT') into predictable constants.
    """
    LIGHT = 'LIGHT'
    FLAT = 'FLAT'
    BIAS = 'BIAS'
    DARK = 'DARK'
    MASTER_LIGHT = 'MASTERLIGHT'
    MASTER_FLAT = 'MASTERFLAT'
    MASTER_DARK = 'MASTERDARK'
    MASTER_BIAS = 'MASTERBIAS'
    MASTER_DARKFLAT = 'MASTERDARKFLAT'
    DARK_FLAT = 'DARKFLAT'

class RegexPatterns:
    """
    Centralized filename-parsing patterns.

    Previously duplicated ad hoc across engine/extractor.py and
    engine/steps/deduplicate.py, so a fix to one copy could leave the other
    behind. Collected here per the future_work.md 'Centralise Regex
    Patterns' item.
    """
    # Splits a WBPP-produced filename into (base capture name, WBPP postfix
    # chain, extension). The postfix chain is only recognized when it
    # *starts* with the calibration marker ('_c' or '_cc') -- WBPP always
    # calibrates before any later stage (cosmetic correction, light
    # pollution suppression, registration), so a real postfix chain is
    # never just '_r' or '_b' on its own. This deliberately excludes a more
    # permissive "any short trailing token" match: several rigs use bare
    # single-letter filter names (R/G/B/S), and 'Target_Filter_R.fits'
    # must not be mistaken for a postfixed 'Target_Filter.fits'.
    #
    # Vocabulary: 'c' (calibrated), 'cc' (cosmetic correction), 'r'
    # (registered/aligned), 'rn' (registered+normalized), 'd' (debayered),
    # 'b' (?), 's' (?) were the tokens implied by the original code's
    # docstring; 'lps' (Light Pollution Suppression) was added after being
    # found in real WBPP output (see REMEDIATION_PLAN.md A1). This list is
    # necessarily incomplete -- WBPP scripts can add their own postfixes --
    # extend it here if a real dataset surfaces one that isn't recognized.
    #
    # A1 in REMEDIATION_PLAN.md: the previous pattern was
    # r'(.+?)(?:_c.*)?(\.xisf|\.fits|\.fit|\.fts)' -- unanchored, and
    # '_c.*' matched a bare '_c' anywhere in the filename (e.g. inside
    # '_calibrated_') and swallowed everything up to the extension,
    # silently merging unrelated captures that happened to contain '_c'.
    WBPP_FILENAME = (
        r'(.+?)'                                    # base capture name (lazy)
        r'(_(?:c|cc)(?:_(?:cc|rn|r|d|b|s|lps))*)?'   # optional postfix chain
        r'(\.xisf|\.fits|\.fit|\.fts)$'              # extension, anchored at end
    )

# Backward compatibility aliases for legacy module support
InternalNames = InternalColumns
StandardizedKeys = InternalColumns
ImageTypes = ImageType
