# Release Notes - AstroBin Upload Utility

## [v2.2.0] - 2026-09-09
### ⚠️ The way you call the script has changed

**Read this before upgrading.** The utility is now installed into, and run from,
a **virtual environment created inside the project directory**. The command you
type gains one prefix:

| | |
| --- | --- |
| Before | `python3 AstroBinUpload.py "/path/to/data"` |
| Now | `.venv/bin/python3 AstroBinUpload.py "/path/to/data"` |
| Now, on Windows | `.venv\Scripts\python AstroBinUpload.py "/path/to/data"` |

There is **no activation step**. You do not need to "enter" or "switch on" the
environment — naming that interpreter is the whole of it, and it works from any
directory and in any terminal window. Every example in the README has been
updated to the new form.

**Setting it up is a one-time command.** In the directory holding
`AstroBinUpload.py`:

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt

(Windows: `python -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`.)
That creates a `.venv` folder containing a private copy of Python and the four
libraries the utility needs.

**Nothing inside the program changed.** Not one line of the processing code is
different because of this. It affects only how the utility is installed and
invoked.

**Why it changed:**

- The old instructions no longer work on a new machine. On Python 3.11 and
  later, Debian, Ubuntu, Fedora and Homebrew all refuse a plain
  `pip install -r requirements.txt` into the system Python, with
  `error: externally-managed-environment`. New users following the previous
  README hit that error before they could run anything.
- `requirements.txt` pins exact versions (`astropy==6.1.3`, `pandas==2.2.3`,
  `numpy==2.1.1`, `configobj==5.0.8`). Installed system-wide, those pins can
  downgrade packages that other software on your machine depends on. Inside a
  virtual environment they cannot reach anything else.

**Python 3.10 or later is required** — that is what the pinned `astropy` and
`numpy` need, and the README previously said only "Python 3.x". If you have
several versions installed you choose which one the utility uses when you create
the environment (`python3.11 -m venv .venv`), and `.venv/bin/python3` is then
permanently that version. This is more predictable than before, not less: a
system-wide install made it easy to install the libraries under one Python and
run the script under another, which is the commonest cause of
`ModuleNotFoundError`. A virtual environment keeps the two together.

**If you already have the utility working, nothing breaks.** Your libraries are
already installed and `python3 AstroBinUpload.py` continues to run exactly as it
does today. The virtual environment is required for new installations, and
recommended when you upgrade — especially if a future release changes a pinned
library version, since that is the case where a system-wide install can go wrong
silently.

### The generated `config.ini` now explains itself

The configuration file the program creates on first run used to be a bare list of
keys, with no indication of what any of it meant — and the program's own message
tells you to edit it before going further. Each section now carries a short
comment: what the `[defaults]` values are for and what `USEOBSDATE` decides, how
to write an `[override]` mapping, what `[equipmentoverrides]` does, which
`[secret]` field you need to change and what happens until you do, and that
`[sites]` maintains itself.

No keys or values changed — only comments were added.

**The default filter codes are now identified as one person's filters.** They are
Astronomik 2 inch round filters, named as N.I.N.A. writes them. An AstroBin filter
ID identifies a specific product, so the same filter name in another brand, size or
mounting has a different ID: keeping the defaults means uploading someone else's
filters. The generated file, `config.ini.example` and the README now all say so —
previously only one README section did, and it was the furthest point from where
the values are edited.

**`config.ini.example` has been corrected.** The shipped example had drifted from
what the program generates: `USEOBSDATE` was `False` where the code default is
`True` (it decides whether frames taken after midnight count with the previous
evening's session), it listed an `FWHM` key that is never read, and several sample
values disagreed with the generated ones. It also still described `[secret]` as
optional and deletable, which this release's own documentation pass had already
corrected elsewhere. It is now rebuilt from the generated output and verified to
parse to the same configuration.

### Restored — `--test` finds the file where it is documented to

If you support other users, this is the flag you use: they send you the
`debug_step_00_RawHeaders.csv` from a `--debug` run, and you replay their session
through the pipeline without ever needing their image files.

The README has always said the CSV is found inside the run's directory, and in
v1.4.x it was — resolved relative to `AstroBinUploadInfo`, which is precisely where
`--debug` puts it, so the bare filename was enough:

    .venv/bin/python3 AstroBinUpload.py "/my/data" --debug
    .venv/bin/python3 AstroBinUpload.py "/my/data" --test debug_step_00_RawHeaders.csv

The v2.0.0 rewrite quietly changed that to resolve against whatever directory you
happened to be standing in, so the documented form failed with a "no such file"
error and you had to type the full path. It went unnoticed because the golden test
harness passes absolute paths and so never used the short form.

Both now work. The file is looked for in `AstroBinUploadInfo` first, then at the
path exactly as given — so replaying your own run takes the bare filename, while a
CSV someone e-mailed you still works from wherever you saved it:

    .venv/bin/python3 AstroBinUpload.py "/scratch/dir" --test ~/Downloads/their_headers.csv

If the file is in neither place, the error tells you both paths it tried instead of
just reporting it missing.

### Restored — running with no arguments creates config.ini

The README has always told you to run the script with no arguments the first
time, to generate a `config.ini` you can then edit. That stopped working around
two years ago, and until now a bare call gave you this instead:

    error: the following arguments are required: directory_paths

— which is the first thing a new user saw, at the moment they had least idea
what to do, with nothing to indicate that a configuration file was needed or
that the program would create one.

It works again:

    .venv/bin/python3 AstroBinUpload.py

creates a default `config.ini` in the directory you are running from and exits
with the familiar message, so you can edit it and then run again with your data.
If a `config.ini` already exists and you call the script with no arguments, it
now tells you a directory path is needed rather than doing nothing useful.

The history, for the record: the behaviour was guarded in v1.3.10 in a way that
would have crashed before reaching the config generation; V1.3.12 fixed that
crash by removing the feature; and v1.4.5's move to `argparse` made a directory
path mandatory, which locked it out. This release restores the documented
behaviour *and* fixes the original crash, rather than choosing between them.

`--help` also reported the version as a hardcoded "v2.1.1" whatever the real
version was. It now reports the running version.

### Restored — reverse geocoding and live sky quality lookups

Through v1.4.x the utility resolved a coordinate that the `[sites]` section did
not already know: OpenStreetMap's Nominatim supplied the site's postal address,
and lightpollutionmap.info's World Atlas 2015 layer supplied its artificial
brightness, converted to SQM and then to a Bortle class. The result was written
back into `[sites]`, so each site was looked up exactly once. The long postal
addresses in a populated `[sites]` section were produced this way — they were
never meant to be typed by hand.

The v2.0.0 rewrite removed all of it, and no changelog entry recorded the loss.
v2.1.0 then removed the leftovers as hygiene, and by v2.1.3 the README described
the outcome as a feature — "no network calls" — which documented an accident as
a design decision. This release puts the capability back, restored from the
v1.4.x source rather than reimplemented, so **a site already recorded in your
`[sites]` section keeps exactly the classification it was given.**

To use it, add a `[secret]` section with an API key and your e-mail address; a
newly generated `config.ini` now includes one with placeholders. The README's
"Accessing sky quality data" and "Reverse Geocoding" sections explain both,
including how to obtain a key.

**If you do not configure `[secret]`, nothing changes for you.** The program
makes no network calls at all, imports neither optional library, and takes the
site name from `[defaults] SITE` and the sky quality from `[defaults] BORTLE`
and `SQM` — exactly as v2.0.0 to v2.1.3 did. Every failure mode (no credentials,
no network, a refused or malformed response, the optional libraries missing)
falls back the same way and lets the run finish.

`geopy` and `requests` return to `requirements.txt`. They are installed by the
standard install command but imported only when a lookup actually happens.

---

## [v2.1.3] - 2026-09-09
### A configured default was ignored when only some frames lacked the header

`[defaults]` exists to answer "what should a frame with no header for this field
be assigned?" — but that promise was only kept when a column was missing from
the data **entirely**. If even one frame in a batch supplied the header, every
other frame's blank in that same column fell back to a hardcoded value instead,
silently discarding what you had configured.

Seven fields were affected: `GAIN`, `EGAIN`, `FOCALLEN`, `XPIXSZ`, `SITELAT`,
`SITELONG` and `OBJECT`. The remaining hardened fields already agreed with their
configured defaults, so nothing changes for them.

The visible symptom was a session mixing frames that carry GPS data with frames
that do not — 250 raw `DARKFLAT` frames alongside lights, in the case that found
it. The summary reported `Latitude: 0.0000° / Longitude: 0.0000°` — a point in
the Gulf of Guinea — instead of the observer's own configured site.

`OBJECT` had a second problem on top: unlike every other field in that table it
had no fallback at all when its column existed, so a blank target name survived
into the report as an empty value.

---

## [v2.1.2] - 2026-09-08
### Calibration section labels, and the issue tracker closed out

**Fixed — calibration sections were labelled `MASTERxxx` unconditionally.** A
session built entirely from raw `DARK`/`FLAT`/`BIAS` frames — no master frame
anywhere in the data — still printed `MASTERDARKS:`, `MASTERFLATS:` and
`MASTERBIAS:` as its section headers. The label now reflects what the table
actually holds: plain `DARKS:`/`FLATS:`/`BIAS:` when every frame is raw,
`MASTERxxx` when they really are masters, and `MASTERxxx` for a mixed table
(a master covering one gain with raw frames surviving for another).

The code carried a comment claiming this matched "v1.4.7 standards". It
didn't — v1.4.7 labelled each section by its literal `IMAGETYP`. The
behaviour drifted at some point and the comment kept citing a standard the
code no longer followed.

Only the header text changed. Frame counts, exposures, the acquisition CSV
and every other line of the summary are untouched, and the raw/master
consolidation into a single table (A13/A14) is unchanged.

**Issues #9 and #10 closed after re-test.** Both concerned master calibration
frames and were left open in v2.1.1 awaiting verification against real data:

- Sub-exposure counts (`32` reported as `1`) are read correctly from WBPP's
  `PixInsight:ProcessingHistory` property, the XISF `COMMENT`/`HISTORY`
  keywords, and FITS `HISTORY`.
- Two master darks at the same gain but different exposures (180 s and 600 s)
  are reported as two rows with their own counts, not merged into one.

Both were verified against constructed reproductions of the reporters' own
figures, and cross-checked byte-for-byte against the independent Rust
implementation. The tracker now has **zero open issues**.

Note the one deferred piece from #10: the optional `[calibrationoverrides]`
ini fallback, for masters that carry no recoverable count at all, is still
unbuilt — see `future_work.md`.

---

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
`--config` flag). #9 and #10 were believed resolved by v2.1.0's master-frame
handling and awaited a re-test against the reporters' data — that re-test was
done in v2.1.2 and both are now closed.

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
