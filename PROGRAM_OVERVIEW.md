# AstroBin Upload Utility - Program Overview

## Purpose
The AstroBin Upload Utility is an automated metadata extraction and aggregation tool designed to streamline the "Bulk Upload" process for AstroBin. It scans your imaging directories, identifies light and calibration frames, and produces the specific CSV and text reports required for accurate session documentation.

## Architecture: The Pipeline Pattern
The application separates the work into three surrounding components and a pipeline of six independent, testable Steps.

**Around the pipeline:**

- **Loader** (`engine/loader.py`): Discovers FITS and XISF files and manages configuration profiles (supporting custom `.ini` files via `--config`).
- **Extractor** (`engine/extractor.py`): High-speed parallel parsing of XML and binary headers.
- **Exporter** (`engine/exporter.py`): Generates the final AstroBin-ready CSV and the human-readable session summary.

**The pipeline itself** — the six steps registered in `AstroBinUpload.py`, in execution order. The stage numbers below are the ones used in the source comments:

1.  **Stage 1 — Normalization (`NormalizeHeadersStep`)**: Sanitizes inconsistent metadata, applies user-defined `[override]` and `[equipmentoverrides]` mappings, and fills gaps from `[defaults]`.
2.  **Stage 2 — Optical Parameters (`OpticalParameterStep`)**: Derives resolution and star metrics — Image Scale (IMSCALE) and FWHM from measured or estimated HFR.
3.  **Stage 3 — Deduplication (`DeduplicateStep`)**: Identifies and removes redundant files (e.g. WBPP postfixes).
4.  **Stage 4 — Calibration Matching (`CalibrationMatcherStep`)**: Associates Darks, Flats and Bias frames using the Hybrid Handshake (EGAIN/GAIN).
5.  **Stage 5 — Geocoding (`GeocodeStep`)**: Resolves each set of coordinates to a named site. See Site Resolution below.
6.  **Stage 6 — Aggregation (`AggregationStep`)**: Vectorized statistical reduction of thousands of frames into session-level summaries.

## Key Logic Components

### The Hybrid Handshake
To ensure calibration frames belong to the correct lights, the utility uses a multi-factor "handshake":
- **Primary**: Electronic Gain signature (`E_0.25`).
- **Secondary**: Linear Integer Gain (`G_100`).
- **Required**: Binning and Filter (for Flats).

### Master Preference
If both raw subs and a Master integration exist for the same hardware group, the utility gives "Master Preference" to the integration. It discards the redundant raws and uses the integrated count from the master's history.

### Site Resolution
`GeocodeStep` resolves coordinates to a site in three tiers, stopping at the first that succeeds:

1.  **Smart Proximity Clustering**: GPS readings drift between sessions, so coordinates within `CLUSTER_RADIUS_M` (110 m, measured with a vectorized haversine) are treated as one physical site, and the cluster's centroid — the mean of all its readings — becomes that site's canonical position. Clustering is greedy single-linkage: once a point joins a cluster it stays there.
2.  **`[sites]` database lookup**: A fuzzy coordinate match against the sites already recorded in `config.ini`. A hit short-circuits, so a known site costs nothing and never touches the network.
3.  **External lookup** (`engine/sites.py`, restored in v2.2.0): Only when the coordinates are new. OpenStreetMap's Nominatim supplies the postal address and lightpollutionmap.info's World Atlas 2015 layer supplies the artificial brightness, converted to SQM and then to a Bortle class. The result is written back into `[sites]`, so each site is looked up exactly once.

Tier 3 uses the API key and e-mail address in the standard `[secret]` section. Without a valid key the sky quality comes from `[defaults] BORTLE`/`SQM`; if the address request fails the site details come from `[defaults] SITE`, `SITELAT` and `SITELONG`. Every failure mode — an unedited placeholder key, no network, a refused or malformed response, `geopy`/`requests` not installed — degrades the same way and lets the run finish, making no further network calls.

### Vectorization
All statistical operations are performed using Pandas vectorized logic rather than Python loops, allowing the utility to process thousands of images in seconds.

## Debugging and Testing
The system is built for high transparency and robust error recovery:
-   **Raw Data Capture**: `debug_step_00_RawHeaders.csv` stores the metadata exactly as read from disk. This is the **only supported source** for standard re-testing via the `--test` flag.
-   **Emergency Diagnostics**: A fatal crash writes `emergency_raw_dump.csv`, preserving scanned metadata for immediate recovery using the `--test` flag — provided the crash happened after extraction. A crash before any headers are read has nothing to dump and produces only the log.
-   **Traceability**: Every file's raw header is logged horizontally (DEBUG level) upon extraction.
-   **Sequential Dumps**: Intermediate dataframes are exported after each pipeline step in `--debug` mode for stage-by-stage auditing.
-   **Full Exception Capture**: Global error handling ensures all crashes record a full Python traceback in the log file.

## Usage
The utility runs from a virtual environment created in the project directory (see the README for the one-time setup; **Python 3.10 or later** is required, a floor set by the pinned `astropy` and `numpy`). Call that environment's interpreter directly — there is no need to "activate" anything, and because the interpreter and its libraries live in the same folder they cannot be mismatched:

    .venv/bin/python3 AstroBinUpload.py [directory_paths] [options]

On Windows:

    .venv\Scripts\python AstroBinUpload.py [directory_paths] [options]

| Argument | Purpose |
| --- | --- |
| `directory_paths` | One or more directories to scan recursively for `.fits`, `.fit`, `.fts` or `.xisf` files. **Omit them entirely on a first run**: with no paths and no `config.ini`, a default configuration is generated and the program exits so it can be edited. With no paths and a `config.ini` already present, it reports that a directory is needed and exits. |
| `--config`, `-c` | Use a named configuration file instead of `config.ini`. |
| `--debug` | Verbose logging, and preserve the intermediate dataframe from every step. |
| `--test CSV_FILE` | Diagnostic mode: inject metadata from a CSV of already-extracted headers instead of scanning disk. Looked for first in the run's `AstroBinUploadInfo` directory — so a bare filename replays that run's own `debug_step_00_RawHeaders.csv` or `emergency_raw_dump.csv` — then at the path as given, relative or absolute. |
