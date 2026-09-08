# Future Revision Notes

Originally written as a code-quality assessment of **v2.0.3**. Most of it has
since been done — largely by the v2.1.0 remediation pass (`REMEDIATION_PLAN.md`,
Bucket A/B) rather than by working through this list directly.

**Status reviewed against v2.1.2 (2026-09-08).** Each item below was checked
against the current code, not assumed from the changelog. What remains open is
at the bottom.

---

## What's Working Well (Keep These)

- **Pipeline pattern** — each processing stage as an independent step class is clean and extensible
- **SessionState dataclass** — passing shared state through the pipeline is the right approach
- **Parallel file extraction** — `ProcessPoolExecutor` in `engine/extractor.py` is a good choice
- **Pandas vectorization** — bulk operations are efficient throughout
- **Documentation** — docstrings and module-level docs are comprehensive

---

## Done

### 1. Silent Exception Handlers — **done** (B3, v2.1.0)
No bare `except: pass` remains. Every handler names its exception types and
leaves at least a `logger.debug()` trace: the emergency dump
(`AstroBinUpload.py`), XISF `ProcessingHistory` / `numberOfImages` parsing
(`engine/extractor.py`), and `seconds_to_hms` (`engine/reports.py`) all log
what went wrong rather than swallowing it.

### 3. Input Validation — **done** (B10, v2.1.0)
`AstroBinUpload.py` checks every directory argument with `os.path.isdir()`
before use and exits with a clear per-path message. Previously a typo'd path
reached `os.makedirs()` and silently manufactured an empty tree.

### 4. Magic Numbers — **done** (B, v2.1.0)
All three are named constants with comments explaining their origin:
`SESSION_GAP_HOURS = 5` (`aggregate.py`), `CLUSTER_RADIUS_M = 110.0`
(`geocode.py`), `EGAIN_UNSET_TOLERANCE = 0.0001` (`calibration.py`).

### 5. GPS Clustering Algorithm — **done** (A3 + A4, v2.1.0)
Both halves fixed. Distance is now a vectorized haversine in metres
(`_haversine_distance_m`, `EARTH_RADIUS_M = 6371000.0`) rather than Euclidean
on degrees, and clustering is a stable greedy single-linkage pass, so an
already-clustered point can no longer be stolen by a later seed. Note the
suggestion in the original item — "or simply use `geopy.distance.distance()`
which is already a dependency" — was deliberately *not* taken: a per-row
geopy call across thousands of frames is far slower than one vectorized
formula, and geopy was dropped entirely (see item 7).

### 6. Centralise Regex Patterns — **done** (v2.1.0)
`constants.py` now has a `RegexPatterns` class; `WBPP_FILENAME` lives there
and is the single definition used by the deduplication step.

### 7. Dead Code (`geopy`) — **done** (A4 + B7, v2.1.0)
`geopy` is no longer imported anywhere and has been removed from
`requirements.txt`, which was itself trimmed from a 50-package `pip freeze` to
the four packages actually imported.

### 9. Logging Infrastructure — **done** (B4, v2.1.0)
The `FunctionNameFilter` that walked `inspect.stack()` on every log record is
gone, replaced by the standard `%(funcName)s` in the format string.

### 2. Test Suite — **substantially done, not to the shape originally suggested**
The original concern ("regressions from future changes are invisible until a
user reports them") is addressed, but by a different route than the four
bullets proposed:

- `golden_tests/run_golden.py` replays committed fixtures through the whole
  pipeline and byte-compares against blessed references — the integration test
  the item asked for, using captured raw-header CSVs rather than sample FITS
  files (the `--test` path was verified byte-identical to a real disk scan, so
  the substitution is faithful).
- `tests/` holds a small pytest suite (config overrides, import sanity).
- Heaviest coverage now lives *outside* this repo: the Rust port
  (`AstroBinUploaderRust`) runs four differential harnesses against this code
  on every push, comparing pipeline state cell by cell and both output
  artifacts byte for byte, over four CSV fixtures and three binary
  FITS/XISF scenarios.

Still missing from this repo's own suite: per-step unit tests against synthetic
DataFrames, and a direct XISF-header-parser test against a fixture file.

---

## Still Open

### 8. Boolean Parsing in Config Loader — **open**
`engine/loader.py` still accepts only `'true'` (case-insensitively) for
`USEOBSDATE`; `'1'` and `'yes'` are read as false. Low impact — the generated
template writes `True`/`False` — but it would surprise a user who typed `yes`.

### 10. Reports Module Coupling — **open**
`engine/reports.py` still mixes aggregation with string formatting in the same
functions. Only worth doing if a second output format (JSON, HTML) is actually
wanted; it carries real rewrite risk and every output byte is currently pinned
by the golden references and the Rust differential harness.

### 11. ConfigObj Dependency — **open**
Still `configobj`. Migrating to `configparser` or TOML would be a breaking
change to every user's `config.ini`, so it needs a deliberate migration path,
not just a swap. Note the format genuinely uses configobj features
(bracket-depth nesting for `[sites]`, comma-implies-list, no type coercion)
that `configparser` does not provide.

### `[calibrationoverrides]` ini fallback — **open** (from issue #10)
The counting bug behind issue #10 is fixed — sub-exposure counts are read from
WBPP's `PixInsight:ProcessingHistory` property, the XISF `COMMENT`/`HISTORY`
keywords, and FITS `HISTORY`. The reporter's *secondary* suggestion — a
config-level fallback for when a master carries no count at all, e.g.

```ini
[calibrationoverrides]
BiasCount = 32
DarksCount = 32
FlatsCount = 32
```

— was never built. Worth doing only if a real master frame turns up with no
recoverable count.

---

## Files of Interest for a Revision

| File | Notes |
|---|---|
| `engine/reports.py` | Tightly coupled; the remaining data/presentation split candidate |
| `engine/loader.py` | Boolean parsing (item 8); the configobj migration (item 11) starts here |
| `engine/steps/base.py` | Still the largest single step; candidate for splitting |
| `constants.py` | Well-structured; `RegexPatterns` now lives here |
