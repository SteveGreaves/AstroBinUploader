# Release Notes - AstroBin Upload Utility

## [v2.1.1] - 2026-09-07
### GitHub Issue Follow-up

A short release reconciling the codebase with the open issue tracker after v2.1.0.

- **#5 — equipment name overrides**: a new optional `[equipmentoverrides]`
  section lets you force the displayed value for `INSTRUME`, `TELESCOP`,
  `FOCNAME`, `FWHEEL` or `ROTNAME` (or any field). Set `FOCNAME = ZWO EAF` and
  every frame reports that, regardless of what the header actually says.
- **#6 — ASCOM Observing Conditions**: `AOCSKYQU` and `AOCAMBT` now feed SQM and
  focuser temperature without any config change on a freshly generated
  `config.ini`.
- **#3 — rotator**: the auto-generated config no longer carries the obsolete
  `ROTATOR` key; the rotator *name* comes from `ROTNAME`, the angle from
  `ROTANTANG`.

Issues #11 and #4 were already resolved by v2.1.0 (deduplication regex anchor;
`--config` flag). #9 and #10 are believed resolved by v2.1.0's master-frame
handling and await a re-test against the reporters' data.

---

## [v2.1.0] - 2026-09-07
### Remediation Release — Correctness, Determinism & a Regression Harness

v2.1.0 is a correctness release. An audit of v2.0.3 produced
`REMEDIATION_PLAN.md`: 14 defects that could change the numbers in your
acquisition CSV or session summary, and 15 hygiene issues. All are fixed, each
in its own commit with a verification note, and the output is now protected by
a committed golden-regression harness.

**Why upgrade.** Several of the Bucket A defects were silent — they produced a
plausible-looking but wrong result rather than an error:
- WBPP-calibrated frames whose filenames contained `_c` anywhere (for example
  `..._calibrated_...`) could be merged into a single row, under-counting total
  exposure.
- Frames from different sessions that share a default filename
  (`Light_0001.fits`) could collapse together.
- GPS site clustering could split one physical site into several, or leave a
  site's coordinates un-averaged.
- With a null in a grouping key, integer columns could render as `100.0`
  instead of `100` in the acquisition CSV.
- Row order — and therefore every "first wins" pick — was not deterministic
  between otherwise identical runs.

**Calibration matching** now treats darks, bias, flats and flat-darks
consistently: all constrain on binning and all honour master preference
("latest master wins"). Darks and bias are correctly treated as
filter-independent in the report table.

**Under the hood.** The startup version handshake and the fourteen duplicated
`__version__` strings are gone, replaced by a single `_version.py`. Logging is
more robust (worker processes on macOS/Windows now log correctly; previously
silent failures leave a debug trace). Dead code and an unused credential slot
in the example config were removed.

**Testing.** `golden_tests/run_golden.py` replays committed metadata fixtures
through the `--test` path and byte-compares against blessed references — no
external data, runs anywhere the repo is checked out. `pytest` is now
installed in the project virtual environment.

This release is the frozen parity target for a planned standalone port
(`RUST_PORT_PLAN.md`).

---

## [v2.0.3] - 2026-02-12
### Optical Metric Type Safety & History Integrity
v2.0.3 addresses a critical stability issue and restores the project's historical documentation standards.
- **Fix**: Resolved a fatal `TypeError` in the `OpticalParameterStep` by adding `HFR`, `FWHM`, and `IMSCALE` to the mandatory type-hardening list in `NormalizeHeadersStep`. This ensures these values are always treated as floats, preventing crashes in Pandas 3.x when processing XISF files.
- **Documentation**: Restored full chronological history to `CHANGELOG.md`, `ReleaseNotes.md`, and `MEMORY.md`.
- **Standards**: Hardened project mandates in `GEMINI.md` to ensure versioned documents and session memories are always maintained in full detail.

---

## [v2.0.2] - 2026-02-11
### Overview
v2.0.2 is the definitive "Master Release" of the 2.0 series. It represents a complete architectural evolution of the AstroBin Upload Utility, transforming it from a procedural script into a high-performance, modular, and hardened ETL (Extract, Transform, Load) pipeline. This release combines the performance of v2.0.0, the diagnostic visibility of v2.0.1, and the system integrity safeguards of v2.0.2.

### 🚀 Key Architectural Advancements

#### 1. The Pipeline Pattern
The utility has been rebuilt from the ground up using a modular Step-based architecture. This decoupling ensures that each transformation stage (Normalization, Deduplication, Calibration Matching, etc.) is independent, robustly testable, and highly maintainable.

#### 2. The Hybrid Handshake
Matching calibration frames to lights now utilizes a multi-tier logic. The system prioritizes the unique electronic signature (`E_GAIN`) of your camera sensor for high-precision pairing, while maintaining a seamless fallback to linear integer `GAIN` for legacy compatibility.

#### 3. Smart XISF & Master Extraction
Our "PixInsight Aware" parser distinguishes between actual linear gain and electronic signatures. It also performs deep inspection of master frames to accurately extract integrated sub-exposure counts from both modern `ProcessingHistory` and legacy `HISTORY` comments.

### 🛠️ Hardened Debugging & Transparency

#### 4. Automatic Crash Diagnostics
Troubleshooting is now proactive. If the pipeline encounters an error, it automatically generates a `CRASH_DIAGNOSTIC.csv` capturing the data's exact state at failure. An `emergency_raw_dump.csv` is also created if a crash occurs during initial disk scanning.

#### 5. High-Visibility Logging
The logging system has been overhauled for 100% data traceability:
- **Horizontal Header Echo**: Every file read has its full raw metadata printed as a dictionary in the log.
- **Granular Milestones**: Detailed tracking of hardware overrides, master preference filtering, and specific calibration assignments.
- **Advanced Tracebacks**: Every fatal error records a full Python traceback in `AstroBinUploader.log`, eliminating "silent stops."

#### 6. Smart Proximity Clustering
We have replaced simple coordinate rounding with **Distance-Based Clustering** (~110m threshold). This resolves "Site Fragmentation" caused by GPS drift and uses **Centroid Averaging** to calculate the most precise geographical coordinates for your final reports.

### 🛡️ Reliability & Flexibility

#### 7. Engine Integrity Verification
To prevent "Frankenstein" installations, v2.0.2 introduces a mandatory **Version Handshake**. The utility verifies that every internal module is in perfect version parity at startup, ensuring you are always running a consistent and supported build.

#### 8. Refined Testing Methodology
We have simplified the diagnostic workflow. Running with `--debug` now generates a **`debug_step_00_RawHeaders.csv`** file. This file captures the exact metadata read from disk and is the primary supported source for the **`--test`** flag, allowing for 100% accurate reproduction of any imaging session.

#### 9. Custom Configuration Profiles
Specify alternative `.ini` files via the `--config` (or `-c`) flag. This allows for effortless switching between Mono, Color, or Remote observatory profiles without manually renaming files.

#### 10. Streamlined Distribution
The architecture has been optimized by collapsing redundant utility files into the primary entry point, reducing overhead and making the tool easier to audit and deploy.

---

## [v2.0.1] - 2026-02-11
### Hardened Debugging & Site Consolidation
- **Diagnostic Visibility**: Implemented horizontal header logging and sequential debug CSV exports for every pipeline step.
- **Crash Preservation**: Automated generation of emergency diagnostic dumps (`CRASH_DIAGNOSTIC.csv`) on fatal errors.
- **Spatial Precision**: Transitioned from coordinate rounding to distance-based spatial clustering (~110m) with centroid averaging to resolve GPS drift.
- **User Flexibility**: Introduced the `--config` flag for hardware profile switching and restored auto-generation of default configuration templates.

---

## [v2.0.0] - 2026-02-10
### The Pipeline Revolution
- **Architecture**: Complete ground-up rewrite using the **Pipeline Pattern**.
- **Performance**: Full transition to vectorized Pandas operations and parallelized FITS/XISF extraction.
- **Hybrid Matching**: Introduced the **Integer Gain Handshake** prioritizing electronic gain signatures.
- **PixInsight Support**: Enhanced XISF parser with integrated sub-exposure count extraction from master headers.
- **Strong Typing**: Implemented dataclass-based state management and centralized constants.

---

## [v1.4.x] - Legacy Milestones
- **v1.4.7**: Initial introduction of the `AstroBinProcessor` pipeline and centralized `constants.py`.
- **v1.4.5**: Implementation of the search/replace hardware override system and multi-key FITS support.

---

## [v1.3.x] - Foundation
- **v1.3.12**: XISF metadata parsing for master frames.
- **v1.3.6**: Introduction of the `--debug` flag and structured subdirectory output.
- **v1.3.0**: Initial baseline release (Feb 2024).
