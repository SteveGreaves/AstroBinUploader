# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.5] - 2026-02-07

### Added
- **Dynamic Hardware Overrides**: Implemented a Search, Replace, and Normalize system for FITS headers via `config.ini`, supporting multi-key variations (e.g. `SQM = AOCSKYQ, AOCSKYQU`) and automatic source pruning.
- **Documentation Audit**: Refined internal documentation, docstrings, and comments across all modules for improved maintainability.
- **Multi-key FITS Overrides**: Added support for multi-key FITS overrides in `config.ini` to handle hardware variations.
- **FITS Header Overrides Documentation**: Documented the user-maintainable FITS header override system for custom hardware support.
- **INI [override] Section**: Added support for an `[override]` section in `config.ini` to manually force specific header values (Exposure, Gain, Filter, etc.) regardless of FITS metadata.
- **XISF Processing History Parsing**: Enhanced `xml_to_data` logic to navigate PixInsight's nested XML and extract the `rows` attribute for accurate Master frame sub-exposure counts.
- **Keyword Aliasing**: Added support for `SITENAME` (mapped to `SITE`) and `FOCUSER` (mapped to `FOCNAME`) for SGP-PRO compatibility.
- **Live Progress Feedback**: Integrated a console counter within the `get_HFR` loop to track progress during LIGHT frame analysis.

### Fixed
- **IMAGETYP Normalization**: Implemented a robust rule to convert any type containing 'light' but NOT 'master' (case-insensitive) to exactly 'LIGHT', ensuring compatibility with varied capture software names while preserving Master frame exclusion.
- **DATE-OBS Timestamping**: Implemented standard removal of fractional seconds from `DATE-OBS` strings to prevent parsing errors during session aggregation.
- **Filter Sanitization**: Added automatic `.strip()` to filter names to prevent duplicate acquisition entries caused by trailing white spaces.
- **Image Type Normalization**: Updated logic to treat `LIGHTFRAME` and `Light Frame` as standard `LIGHT` frames.
- **Installation Path Resolution**: Resolved a bug where the script failed to find `config.ini` when run for the first time from the root directory.

### Changed
- **Session Date Logic**: Refined `USEOBSDATE` parameter handling; when set to `False`, a 5-hour threshold is used to roll early morning images into the previous night's session date.
- **Calibration Association**: Modified `modify_lat_long` to ensure Darks/Flats inherit the coordinates of the nearest Light frame to maintain site consistency.
- **Log Formatting**: Updated `summarize_session` to include total processed image counts and improved temperature statistic alignment.

---

## [1.3.12] - 2025-09-24
### Changed
- Modified `xml_to_data` function to obtain number of images used in a master from modified PixInsight .xisf header structures.

## [1.3.11] - 2024-10-16
### Fixed
- Bug where running the script for the first time from the installation directory would fail.
- Handled filter names with trailing white spaces.
### Added
- New default parameter `USEOBSDATE` for observation session aggregation logic.
- Progress counter to the hfr processing.

## [1.3.10] - 2024-09-29
### Changed
- Deals with the case where a fractional part of a second is present in some date-obs keyword but not in others.
- Deals with the case where the filter names in the light frames have trailing white spaces.

## [1.3.9] - 2024-09-28
### Changed
- Allows the processing of LIGHTFRAMES and Light Frames as well as LIGHT frames.
- Modification in the process headers function.

## [1.3.8] - 2024-09-28
### Changed
- Allows the processing of LIGHTFRAMES as well as LIGHT frames via modification in the process headers function.

## [1.3.7] - 2024-06-16
### Added
- Script can be called from an image directory to process local images while accepting calibration directories as arguments.

## [1.3.6] - 2024-06-16
### Added
- Ability to call the script with `--debug` flag to save dataframes to csv files.
### Fixed
- Corrected error caused by site coordinates and date format of some images.
### Changed
- Modified script to save all output files to a subdirectory of the directory being processed.

## [1.3.5] - 2024-05-05
### Added
- Ability to call the script with no arguments to process the current directory.
- Debug, txt and csv output files are saved to a subdirectory of the directory being processed.

## [1.3.4] - 2024-03-05
### Fixed
- Corrected utf-8 encoding error with logging.
- Reset index on group in summarize session for correct target return.
- Formatted time output; seconds now shown to 2dp.

## [1.3.3] - 2024-03-04
### Fixed
- Corrected error in `aggregate_parameters` where script would fail if no MASTER frames were present.

## [1.3.2] - 2024-02-29
### Added
- Handled FOCUSER and SITENAME SGP-PRO keywords.
- Logic to handle conflicting keyword pairs (EXPTIME/EXPOSURE and LAT-OBS/SITELAT).
- Support for multiple MasterFlat frames for the same filter.
- Logic to ensure calibration frames inherit the nearest light frame location.

## [1.3.1] - 2024-02-26
### Changed
- Modified debugging file dumps to occur after data processing rather than at the end.

## [1.3.0] - 2024-02-12
- Initial Release Version 1.3.0.