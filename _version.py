"""
Single source of truth for the application version.

Every other module imports from here. Previously each of the fourteen modules
carried its own hardcoded ``__version__`` string, verified at startup by
``verify_engine_integrity`` to detect mixed-version installations. That check
was removed in v2.1.0: it could not survive ordinary packaging, it made every
partial edit unrunnable, and a genuinely mixed installation fails on an API
mismatch long before a version string is consulted.
"""

__version__ = '2.1.3'
