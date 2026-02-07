# AstroBinUpload Change Log

## v1.4.5 (2026-02-07)
- **Fix**: Upgraded to a more robust date-handling strategy in `aggregate_parameters` to resolve type-conflicts caused by v1.4.4. Now performs conditional string conversion and direct assignment to ensure correct Pandas dtype updates.

## v1.4.4 (2026-02-04)
- **Fix**: Resolved DatetimeArray type mismatch in aggregate_parameters by forcing an explicit string cast on the date-obs column. This prevents crashes on systems where Pandas auto-detects dates with high-precision (e.g., datetime64[us]).

## v1.4.3 (2026-02-03)
- **Logic Restoration**: Re-implemented robust date parsing to prevent session collapsing (Fix for 1900-01-01 default).
- **Calibration Handshake**: Fixed Dark/Bias frame matching by normalizing Gain and Cooling to rounded integers (Fix for float mismatch).
- **Versioning**: Updated all internal file headers to v1.4.3.

## v1.4.2 (2026-02-01)
- **Performance**: Implemented ProcessPoolExecutor for parallel header extraction on high-speed drives.
- **Modularization**: Refactored monolithic code into specialized modules (headers_functions, processing_functions, etc.).
- **Cross-Platform**: Fixed Windows f-string syntax errors for newline characters.

## v1.4.0 (2025-09-25)
- **XISF Support**: Added deep parsing for WBPP ProcessingHistory to extract accurate image counts.
- **Coordinate Logic**: Implemented Euclidean distance alignment for calibration frames and config.ini fallbacks for missing site data.
- **USEOBSDATE**: Added global switch to toggle between header-dates and custom session parsing.
