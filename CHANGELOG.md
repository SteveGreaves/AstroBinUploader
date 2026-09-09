# Changelog

## [2.2.0] - 2026-09-09
Restores a capability the v2.0.0 rewrite dropped without recording it.

### Restored
- **Reverse geocoding and live sky-quality lookups.** Through v1.4.x,
  `sites_functions.py` resolved a coordinate the `[sites]` database did not
  know: OpenStreetMap's Nominatim supplied the site's postal address, and
  lightpollutionmap.info's World Atlas 2015 layer supplied its artificial
  brightness, converted to SQM and then to a Bortle class. The result was
  written back into `[sites]`, so each site was looked up exactly once. The
  long addresses in a populated `[sites]` section were produced this way;
  they were never meant to be typed by hand.

  The v2.0.0 "clean slate" rewrite removed all of it. No changelog entry
  recorded the loss — that release claimed "absolute visual parity with
  legacy v1.4.x reports", which was true of the *reports* and not of the
  behaviour behind them. v2.1.0 then removed the leftovers as hygiene: the
  `Nominatim` import (described as "unused", which it was, because its caller
  had already gone) and the `[secret]` section holding the API key and the
  contact e-mail. By v2.1.3 the README described the result as a feature —
  "No network calls: since v2.1.0 the program contacts no external service" —
  which documented an accident as a design decision, and misdated it.

  Restored in `engine/sites.py` from the v1.4.x source rather than
  reimplemented: the nine Bortle bands, the brightness→SQM formula and its
  constants, and the 16-character key validation are carried over exactly, so
  a site already recorded in a user's `[sites]` section keeps the
  classification it was given.

- **The `[secret]` configuration section**, and `geopy`/`requests` as
  *optional* dependencies.

### Changed — a generated `config.ini` now explains itself
`_generate_default_config` wrote no comments at all, so the file a new user is
told to edit was a bare list of keys. Every section now carries a short comment
saying what it is for: what `[defaults]` values are used for and what
`USEOBSDATE` decides, the `STANDARD = YOUR_KEYWORD` form of `[override]`, what
`[equipmentoverrides]` is for, which `[secret]` field to edit and what happens
until it is, and that `[sites]` is written by the program. Keys and values are
unchanged, and the generated file still parses identically to
`config.ini.example`.

Comments are attached after every section has been created: `ConfigObj` indexes
`inline_comments` by section as each is added, so setting a comment for a section
that does not exist yet makes `write()` fail with a `KeyError` on that name.

### Changed — the default filter codes are identified as one person's filters
The `[filters]` defaults are the author's own **Astronomik 2 inch round filters**,
named as N.I.N.A. writes them. An AstroBin filter ID identifies a specific
product, so the same filter name in a different brand, size or mounting has a
different ID — a user who keeps the defaults silently uploads someone else's
filters. Only the README's "Astrobin Filter-Code mappings" section said so, which
is the furthest point from where the values are actually edited. Now stated in the
generated `config.ini`, in `config.ini.example`, and in the README's `[filters]`
walkthrough.

### Fixed — `config.ini.example` matched neither the code nor the documentation
The shipped example config had drifted from what the program actually generates:
`USEOBSDATE` was `False` where the code default and the README both say `True`,
a `FWHM` key was listed that is never read (FWHM is derived as `HFR * 2`), and
several sample values disagreed with the generated defaults. It also still
described `[secret]` as "OPTIONAL — delete this whole section", the framing
corrected in v2.2.0's own documentation pass, which had updated `README.md` but
not this file. `[secret]` is a standard section; the fallbacks depend on whether
the API key is valid, not on whether the section is present.

Rebuilt from the generated output with the explanatory comments retained, and
verified by loading both files through `ConfigLoader` and comparing the parsed
configuration.

### Restored — `--test` finds the file where it is documented to
`--test` is how a support request is reproduced: the user sends the
`debug_step_00_RawHeaders.csv` their `--debug` run produced, and the pipeline is
replayed from it without needing their FITS data.

v1.4.x resolved the argument as `os.path.join(output_dir, args.test)`, i.e.
relative to the run's `AstroBinUploadInfo` directory — which is exactly where
`--debug` writes that file, so the bare filename was enough. The v2.0.0 rewrite
replaced this with a plain `pd.read_csv(csv_path)`, resolving against the current
directory instead, and the documented form stopped working. The regression stayed
invisible because `golden_tests/run_golden.py` passes an absolute path, so the
harness never exercised it.

Both forms now work, tried in order: the run's `AstroBinUploadInfo` directory
first, then the path exactly as given. An absolute path satisfies both. This
restores the documented behaviour without breaking the form that v2.0.0–v2.1.3
accepted — which is the one that matters for a CSV received from someone else. If
the file is in neither location, the error names both paths tried.

Also fixed: `--help` described the CSV as needing to "reside in the first
directory path provided", which was never true of the v2.x code.

### Restored — running with no arguments creates config.ini
The README has always documented that calling the script with no arguments
generates a default `config.ini` and exits, giving a new user something to edit.
That has not worked for about two years:

- **v1.3.10 and earlier** guarded it as
  `if len(sys.argv) < 2 and os.path.isfile(CONFIGFILENAME)`, deliberately
  letting a first run with no arguments fall through to config generation — but
  `output_dir = sys.argv[1]` sat between that guard and the generation code, so
  it raised `IndexError` before reaching it.
- **V1.3.12** commented out the `os.path.isfile` term, fixing the crash by
  removing the feature.
- **v1.4.5** moved to `argparse` with `nargs='+'`, making a directory path
  structurally mandatory. From then until v2.1.3 a bare call produced only
  `error: the following arguments are required: directory_paths`, which is what
  a new user met as their very first interaction with the program.

`directory_paths` is now `nargs='*'`, with the first-run branch running before
any path-dependent setup — so the crash is fixed rather than reintroduced. With
no paths and no `config.ini`, the file is generated and the program exits. With
no paths and a `config.ini` already present, it reports that a directory is
needed and exits — the case V1.3.12 was actually concerned with, now handled
explicitly. A named `--config` profile that does not exist is an error, never a
request to create one.

Also fixed: `--help` reported the version as a hardcoded "v2.1.1" regardless of
the actual version, and its examples have been updated to the new call form.

### Changed — how you call the script
The utility is now installed into, and run from, a **virtual environment created
inside the project directory**. The documented command changes from

    python3 AstroBinUpload.py [directory_paths]

to

    .venv/bin/python3 AstroBinUpload.py [directory_paths]

(Windows: `.venv\Scripts\python AstroBinUpload.py`). There is no activation
step — naming that interpreter is enough, from any directory or terminal.

**Nothing in the program changed.** This is purely how it is installed and
invoked. Two reasons for the move:

- On Python 3.11 and later, Debian, Ubuntu, Fedora and Homebrew refuse a plain
  `pip install -r requirements.txt` into the system interpreter with
  `error: externally-managed-environment`. The previously documented
  installation therefore fails outright on a current machine.
- `requirements.txt` pins exact versions (`astropy==6.1.3`, `pandas==2.2.3`,
  `numpy==2.1.1`, `configobj==5.0.8`). Installed system-wide those pins can
  downgrade packages other software depends on; inside a virtual environment
  they cannot reach anything else.

Choosing between installed Python versions is unaffected, and is now recorded
rather than left to chance: create the environment with the interpreter you want
(`python3.11 -m venv .venv`) and `.venv/bin/python3` is permanently that version.
The documented minimum is now **Python 3.10**, which is what `astropy 6.1.3` and
`numpy 2.1.1` require — the README previously said only "Python 3.x".

**Existing installations are unaffected** — if the libraries are already
installed, `python3 AstroBinUpload.py` keeps working exactly as before. The
virtual environment is required for new installations, and recommended when
upgrading, particularly if the pinned library versions change.

### Behaviour when not configured
Unchanged, deliberately, and this is the part that is tested hardest: with no
`[secret]` section the program makes **no network calls at all**, imports
neither optional dependency, and takes the site name from `[defaults] SITE`
and the sky quality from `[defaults] BORTLE`/`SQM` exactly as v2.0.0–v2.1.3
did. Every failure mode — missing credentials, no network, a refused request,
a malformed response, `geopy`/`requests` not installed — degrades to those
same defaults and lets the run finish. A `--test` replay never writes to the
configuration.

The golden corpus has no `[secret]` section and both references are
unchanged, which is what proves the offline path did not move.

### Testing
- `tests/test_sites.py` — 30 cases: the offline guarantee, every Bortle band
  and both sides of every boundary, the brightness conversion, credential
  validation, the `[secret]` section's `KEY = ENDPOINT` shape, and the
  request path driven through an injected fake so the suite never needs a
  network.

## [2.1.3] - 2026-09-09
Found while building a test dataset for the Rust port (`ic405`: a real
February 2022 session including 250 raw `DARKFLAT` frames, none of them
carrying GPS data).

### Fixed
- A configured `[defaults]` value was only ever honoured when a header
  column was missing *entirely*. The moment even one frame in a batch
  supplied the header, every other frame's blank in that same column fell
  back to a hardcoded literal instead — silently discarding the user's
  configuration for exactly the case `[defaults]` exists to cover. Affected
  seven fields where the hardcoded fallback disagreed with a configured
  one: `GAIN`, `EGAIN`, `FOCALLEN`, `XPIXSZ`, `SITELAT`, `SITELONG`,
  `OBJECT`.

  The visible symptom: a calibration frame with no `SITELAT`/`SITELONG`
  reported `Latitude: 0.0000° / Longitude: 0.0000°` — the Gulf of Guinea —
  instead of the observer's own configured site, whenever at least one
  other frame in the same run supplied real coordinates. `OBJECT` had a
  second, narrower bug on top: no fallback branch existed for it at all, so
  a blank target name stayed blank straight through to the report rather
  than falling back to even the hardcoded `'Unknown'`.

  `NormalizeHeadersStep`'s per-cell hardening (`base.py`) now looks up each
  of these seven from `config.defaults` first, falling back to the
  hardcoded literal only when the config file does not define that key.
  The other twelve hardened fields (`BORTLE`, `SQM`, `FOCTEMP`, `CCD-TEMP`,
  `FOCRATIO`, `EXPOSURE`, `XBINNING`, `FILTER`, ...) already agreed with
  their configured default and are unaffected.

  `sadr` and `sh2101_calib` — the two golden references — are unchanged: neither
  carries a blank in any of the seven fields.

## [2.1.2] - 2026-09-08
Found while validating the Rust port against real, unstructured data.

### Fixed
- Every calibration section in the session summary was labelled `MASTERxxx`
  unconditionally — a raw, uncalibrated `DARK`/`FLAT`/`BIAS` session showed
  `MASTERDARKS:`/`MASTERFLATS:`/`MASTERBIAS:` even with no master frame
  anywhere in the data. The code's own comment claimed this matched
  "v1.4.7 standards"; it didn't — v1.4.7's `process_image_type`
  (`utils.py`) labelled each section by its literal `IMAGETYP`, e.g. plain
  `DARK:` for raw darks and `MASTERDARK:` only for genuine masters. That
  behaviour was lost sometime between v1.4.7 and this codebase while the
  comment kept citing it. `format_image_type_table` now derives the label
  from what the table actually contains: plain when every row is raw,
  `MASTERxxx` when every row is a real master, and `MASTERxxx` for a
  genuinely mixed table (a master covering one gain, raw frames surviving
  for another the master doesn't) — the safer thing to over-claim toward
  when a table isn't uniform. The A13 consolidation this replaces (grouping
  a class's raw and MASTER variants into one table) is unchanged; only the
  header text was wrong.

  Both golden references re-blessed: `sadr` has no calibration frames at
  all and is unaffected; `sh2101_calib` is all-raw and its three
  calibration sections now correctly read `FLATS:` / `BIAS:` / `DARKS:`.

### Closed
- **#9** and **#10** — the last two open issues, both about master
  calibration frames, held open since v2.1.0 pending a re-test against real
  data. The reporters' data was never available, so both were reproduced
  from the figures in the reports themselves and verified:
  - sub-exposure counts read correctly from WBPP's
    `PixInsight:ProcessingHistory` property, the XISF `COMMENT`/`HISTORY`
    keywords and FITS `HISTORY` — not the `1` originally reported;
  - two master darks at one gain but different exposures (180 s / 600 s)
    report as two rows with their own counts, not one merged row.

  Both cross-checked byte-for-byte against the independent Rust
  implementation. The tracker now has zero open issues. The optional
  `[calibrationoverrides]` ini fallback suggested in #10 remains unbuilt and
  is tracked in `future_work.md`.

### Documentation
- `README.md` corrected against the current code: the installation list named
  modules gone since v2.0 (`utils.py`, `config_functions.py`, …); `[secrets]`
  and both API reference sections still described network calls removed in
  v2.1.0 (the utility has been fully offline since); `[sites]` claimed to
  update itself automatically, which stopped when those calls went; the config
  walkthrough still seeded the stale `ROTATOR` key fixed in v2.1.1; and
  `[equipmentoverrides]` was undocumented despite shipping in v2.1.1.
- `future_work.md` reviewed item by item against the current code — 7 of its
  11 items are done, 1 partly; the rest are marked open with what remains.

## [2.1.1] - 2026-09-07
Follow-up to v2.1.0, closing the loop with the GitHub issue tracker.

### Added
- **#5** — optional `[equipmentoverrides]` config section. Forces a literal
  display value into a named column (`instrume` / `telescop` / `focname` /
  `fwheel` / `rotname`, or any column) for every frame, applied right after
  default injection. For headers that carry `EAF` rather than `ZWO EAF`.

### Changed
- **#6** — the generated default `[override]` section now maps the ASCOM
  Observing-Conditions keywords out of the box: `SQM = AOCSKYQ, AOCSKYQU` and
  `FOCTEMP = AOCAMBT`, plus `FOCNAME = FOCUSER` and `SWCREATE = CREATOR`. The
  `[override]` plumbing already supported this — only the generated template
  lagged.

### Fixed
- **#3** — the generated default `config.ini` seeded a stale `ROTATOR` key; the
  rotator name has keyed on `ROTNAME` (angle on `ROTANTANG`) since the pipeline
  rewrite. Generator now matches `config.ini.example`.

## [2.1.0] - 2026-09-07
Full remediation of v2.0.3 per `REMEDIATION_PLAN.md`. 14 output-affecting
defects (Bucket A) and 15 hygiene items (Bucket B) fixed, each with its own
commit and verification note. A golden regression harness now guards behaviour.

### Fixed — output-affecting (Bucket A)
- **A1 — Deduplication regex**: the WBPP post-fix pattern was unanchored and
  matched a bare `_c` anywhere in a name (e.g. inside `_calibrated_`), then
  swallowed everything to the extension — silently merging unrelated captures
  into one row. Replaced with an anchored pattern.
- **A2 — Deduplication scope**: keys now include the source directory, so
  identically-named frames from different sessions (`Light_0001.fits`) no
  longer collapse together.
- **A3 — GPS clustering**: an already-clustered point could be stolen by a
  later seed, stripping the first cluster to a single un-averaged point.
  Clustering is now a stable greedy single-linkage pass.
- **A4 — GPS distance**: replaced flat degree-space Euclidean distance (which
  silently shrank the effective radius with latitude) with a haversine metric
  in metres.
- **A5 — Group keys**: hardened against numeric grouping keys being promoted
  to object dtype (which printed `100.0` instead of `100` in the acquisition
  CSV).
- **A6 — Hardware overrides**: a typo'd `[override]` target and list-value
  corruption made the section partly dead.
- **A7 — FITS HDU selection**: metadata is now read from the HDU that carries
  it rather than always HDU 0.
- **A8 — Header normalisation order**: column case is normalised before
  defaults are injected, not after.
- **A9 — Determinism**: file discovery and sorting are now fully deterministic,
  so every `first()` / `iloc[0]` resolution is stable between runs.
- **A10 / A11 — Calibration semantics**: master-preference ("latest wins") and
  flat-dark matching (binning + master preference) made consistent with the
  other calibration classes.
- **A12 — Optical metrics**: `OpticalParameterStep` vectorised without changing
  rounding — builtin round-half-to-even is preserved via an explicit helper
  rather than switching to pandas/numpy rounding.
- **A13 — IMAGETYP normalisation**: stopped a second normalisation pass from
  matching its own output and erasing `MASTER` designations.
- **A14 — Calibration report**: darks and bias are filter-independent and are
  no longer grouped or labelled by filter in the report table.

### Changed / hardened (Bucket B)
- Single source of truth for the version (`_version.py`); the startup version
  handshake and 14 duplicated `__version__` strings were removed.
- Logging robustness: `funcName` in the format, worker logging fixed for
  spawn-based platforms, several silent `except` clauses given debug traces.
- Dead code removed (unused imports, orphaned `pipeline.py`, unused `[secret]`
  config section); deprecated `groupby(axis=1)` replaced with a dtype-safe
  coalesce; input paths validated; magic numbers named; `requirements.txt`
  trimmed to real dependencies.

### Testing
- New `golden_tests/run_golden.py` — replays committed fixtures through the
  `--test` path and byte-compares against blessed references. Ships with two
  fixtures (`sadr`, `sh2101_calib` — 1693 real frames exercising calibration
  matching end to end). `pytest` is now wired into the project venv.

## [2.0.3] - 2026-02-12
### Fixed
- **Optical Metric Type Safety**: Resolved a fatal `TypeError` when processing XISF files by adding `HFR`, `FWHM`, and `IMSCALE` to the mandatory type-hardening list. This ensures these values are always treated as floats, preventing crashes during assignment in Pandas 3.x.

## [2.0.2] - 2026-02-11
### Added
- **Engine Integrity Verification**: Implemented a mandatory version handshake across all internal modules to prevent 'Frankenstein' installations and ensure architectural parity.
- **Architectural Optimization**: Collapsed `utils.py` into the main entry point to reduce external dependencies and streamline initialization.
- **Refined Testing Methodology**: Introduced `debug_step_00_RawHeaders.csv` to capture raw extraction state, providing a stable and primary source for the `--test` diagnostic mode.

### Changed
- **Versioning Strategy**: Transitioned to module-level `__version__` signatures for all core components.

## [2.0.1] - 2026-02-11
### Added
- **Hardened Debugging System**: Complete overhaul of logging and diagnostic output to match and exceed v1.4.0 standards.
- **Horizontal Header Logging**: Every file processed now has its raw recovered header dictionary printed horizontally in the log for immediate verification.
- **Sequential Debug CSVs**: Automatic export of intermediate dataframes after every pipeline step when `--debug` is enabled.
- **Emergency Diagnostic Dumps**: Automatic generation of `CRASH_DIAGNOSTIC.csv` and `emergency_raw_dump.csv` on any fatal error, ensuring data preservation even without debug flags.
- **Advanced Error Tracking**: Global exception handling ensures all exit errors, including full Python tracebacks, are captured in `AstroBinUploader.log`.
- **Smart Proximity Clustering**: Replaced coordinate rounding with distance-based grouping (~110m threshold) and Centroid Averaging, providing superior spatial resolution and resolving GPS drift boundaries.
- **Custom Configuration Support**: Added the `--config` (or `-c`) flag to specify alternative `.ini` files, enabling easy switching between Mono, Color, and Remote equipment profiles.
- **Auto-Config Generation**: Restored the ability to automatically generate a default `config.ini` template if the file is missing.

### Changed
- **Logging Density**: Increased granular milestones for hardware overrides, master preference filtering, and calibration matching logic.
- **Sequential Golden Tests**: Updated testing protocol to mandate sequential execution and summary verification.

## [2.0.0] - 2026-02-10
### Added
- **Hybrid Handshake Matching**: New calibration matching engine that prioritizes high-precision `EGAIN` signatures while falling back to linear `GAIN` for legacy compatibility.
- **Smart XISF Extraction**: Enhanced parser that automatically detects electronic gain signatures in PixInsight `instrument:gain` properties.
- **Deep Master Inspection**: Fallback logic to extract integrated sub-exposure counts from legacy PixInsight history comments.
- **Golden Test Suite**: Expanded reference tests to include mosaics, CSV overrides, and multi-gain datasets.
- **Clean Slate Architecture**: Complete project rewrite focusing on modularity, testability, and performance.
- **Pipeline Pattern**: Introduced a decoupled transformation pipeline where logic is isolated into independent, pluggable `PipelineStep` modules.
- **Strong Typing**: Implemented Python Dataclasses and Enums for configuration (`AppConfig`) and state (`SessionState`) management, eliminating loosely-typed dictionary dependencies.
- **Modernized Engine**: Rebuilt the core execution engine to strictly separate I/O (Extraction), Logic (Steps), and Presentation (Exporter).
- **Comprehensive Documentation**: Added detailed docstrings and comments across the entire codebase to adhere to the Platinum Standard of software engineering.

### Fixed
- **The 200-Flat Bug**: Resolved double-counting of flats by ensuring master frames correctly preempt raw subs through robust hardware grouping.
- **Metadata Drift**: Fixed issues where slight precision differences in headers caused separate site or gain groups.
- **Visual Parity**: Ensured that the modern architecture produces human-readable reports and console output identical to the legacy standard.
- **Data Integrity**: Hardened the numeric pipeline with centralized `pd.to_numeric` conversion and robust fallbacks to project defaults.

### Changed
- **Reporting Engine**: Isolated display logic ensures all outputs (ASCII and CSV) use human-readable linear integer Gains.
- **Aggregation**: Vectorized statistical reduction using Pandas for high-speed processing of large datasets.
- **Repository Rationalization**: Removed all legacy procedural code, establishing a clean root directory structure.
- **Vectorized Performance**: Fully integrated high-speed vectorized operations for aggregation and date-shifting. Achieving a **~62% speed increase** in processing time compared to the legacy iterative baseline (v1.4.5).

---

## [1.4.7] - 2026-02-08

### Added
- **AstroBinProcessor Pipeline**: Introduced a centralized `AstroBinProcessor` class in `pipeline.py` to manage application state and orchestrate the ETL workflow. This modularizes the codebase, separating core logic from the CLI entry point.
- **Constants Management**: Introduced `constants.py` to centralize FITS keywords, configuration labels, and internal column names, eliminating "magic strings" and preventing typo-related logic failures.

### Changed
- **Vectorized Performance Overhaul**: Replaced slow iterative loops in session aggregation with optimized Pandas operations (vectorized date shifting). This results in near-instantaneous processing of large datasets compared to previous versions.
- **Robust FITS Hardening**: Implemented centralized numeric hardening using `pd.to_numeric` across all modules. The utility now handles malformed or non-standard FITS metadata with automated fallbacks to project defaults from `config.ini`.
- **Structural Cleanup**: Rationalized the repository structure, separating the manager logic (`pipeline.py`) from functional utilities, ensuring long-term maintainability.

---

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
- **Parameter Aggregation Robustness**: Fixed a regression in `aggregate_parameters` where missing columns (e.g., `ROTANTANG`) would cause a crash. Implemented a mandatory column injection logic that populates missing DataFrame columns using values from the `[defaults]` section of `config.ini`.
- **Session Summary Formatting**: Resolved an issue where equipment items with missing data were displayed as "nan" in the text summary. Updated `equipment_used` to intelligently filter out "nan", "None", and empty strings.
- **Keyword Synchronization**: Standardized the use of `ROTANTANG` across the codebase, configuration templates, and data type conversion logic, resolving inconsistencies with `ROTATANG`.
- **CLI Cleanliness**: Removed diagnostic `print` statements from the core header processing and directory scanning logic to ensure a professional and clean CLI experience.
- **IMAGETYP Normalization**: Implemented a robust rule to convert any type containing 'light' but NOT 'master' (case-insensitive) to exactly 'LIGHT', ensuring compatibility with varied capture software names while preserving Master frame exclusion.

### Changed
- **Performance Optimization**: Significantly improved code execution speed by optimizing I/O operations and leveraging vectorized Pandas operations for header conditioning and parameter aggregation, ensuring high-speed compatibility with RAID 0 environments.
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
