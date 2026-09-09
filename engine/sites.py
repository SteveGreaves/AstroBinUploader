"""
Site identification over the network - reverse geocoding and sky quality.

Restored in v2.2.0. This capability existed through v1.4.x in
``sites_functions.py`` and was dropped, unannounced, by the v2.0.0 "clean
slate" rewrite; v2.1.0 then removed the leftover ``Nominatim`` import and the
``[secret]`` config section it depended on, describing both as unused. They
were unused only because their caller had already gone. Nothing in any
changelog recorded the loss. The behaviour here is restored from the v1.4.x
source, not reimplemented from memory - see
``sites_functions.v1.reference.py``.

Two lookups, both for a coordinate the local ``[sites]`` database does not
already know:

1. **Reverse geocoding** (Nominatim/OpenStreetMap) turns the coordinate into a
   postal address, which becomes the site name in the report. The long
   addresses in a populated ``[sites]`` section were produced this way; they
   were never meant to be typed by hand.
2. **Sky quality** (lightpollutionmap.info) returns the World Atlas 2015
   artificial brightness at that point, which converts to an SQM figure and
   then to a Bortle class - real measurements for the site rather than
   whatever single number the user put in ``[defaults]``.

The result is written back into ``config.ini`` so the lookup happens once per
site, ever. That write is the reason this module degrades so carefully: it
must never corrupt a working configuration because a network call failed.

**Everything here is optional and fails soft.** No ``[secret]`` section, no
API key, no e-mail address, no network, a refused request, a malformed
response - every one of those falls back to the ``[defaults]`` values and the
pipeline continues. An offline run and a run with no ``[secret]`` section
produce byte-identical output to v2.1.3, which is what keeps the golden
corpus (whose ``golden_config.ini`` has no ``[secret]`` section) valid and
reproducible with no network access at all.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

from constants import ConfigSections, FITSKeywords

# The World Atlas 2015 dataset, and the constants of the standard conversion
# from artificial brightness to mag/arcsec^2. Both are carried over verbatim
# from v1.4.x: changing either would silently reclassify every site.
WORLD_ATLAS_LAYER = 'wa_2015'
_BRIGHTNESS_OFFSET = 0.171168465
_BRIGHTNESS_SCALER = 108000000

# lightpollutionmap.info issues 16-character alphanumeric keys.
_API_KEY_LENGTH = 16

# Nominatim's terms of service require a contact address in the user agent.
# Without one we do not call it at all, rather than send an anonymous request.
_USER_AGENT_TEMPLATE = "AstroBinUpload.py_{email}"


def sqm_to_bortle(sqm: float, logger: logging.Logger) -> int:
    """Converts a Sky Quality Meter reading to its Bortle class.

    The nine bands are those used since v1.4.x. They are asymmetric and the
    boundaries are inclusive as written; they are reproduced exactly rather
    than tidied into a table, because every site already recorded in a user's
    ``[sites]`` section was classified by this function.

    Args:
        sqm (float): Sky brightness in mag/arcsec^2. Larger is darker.
        logger (logging.Logger): Application logger.

    Returns:
        int: Bortle class 1 (pristine) to 9 (inner city). Returns 9 - the
            safest over-estimate of light pollution - if ``sqm`` is not a
            number, matching the v1.4.x fallback.
    """
    logger.info(f"Converting SQM value {sqm} to Bortle scale")
    try:
        if not isinstance(sqm, (int, float)) or isinstance(sqm, bool):
            raise ValueError(f"SQM must be a number, got {type(sqm).__name__}")

        if sqm > 21.99:
            return 1
        elif 21.50 <= sqm <= 21.99:
            return 2
        elif 21.25 <= sqm <= 21.49:
            return 3
        elif 20.50 <= sqm <= 21.24:
            return 4
        elif 19.50 <= sqm <= 20.49:
            return 5
        elif 18.50 <= sqm <= 19.49:
            return 6
        elif 17.50 <= sqm <= 18.49:
            return 7
        elif 17.00 <= sqm <= 17.49:
            return 8
        else:
            return 9
    except Exception as e:
        logger.error(f"Could not convert SQM to Bortle ({e}); assuming Bortle 9")
        return 9


def brightness_to_sqm(artificial_brightness: float) -> float:
    """The World Atlas artificial-brightness to mag/arcsec^2 conversion.

    Split out of the request path in this restoration so the arithmetic is
    unit-testable without a network call; the formula and the 2-dp rounding
    are unchanged from v1.4.x.

    Args:
        artificial_brightness (float): Raw value from the API.

    Returns:
        float: SQM in mag/arcsec^2, rounded to two decimals.
    """
    return round(
        math.log10((artificial_brightness + _BRIGHTNESS_OFFSET) / _BRIGHTNESS_SCALER) / -0.4,
        2,
    )


def is_valid_api_key(api_key: Any) -> bool:
    """True for a 16-character alphanumeric key, as v1.4.x validated it."""
    return isinstance(api_key, str) and len(api_key) == _API_KEY_LENGTH and api_key.isalnum()


def is_valid_api_endpoint(api_endpoint: Any) -> bool:
    """True for any non-blank endpoint string."""
    return isinstance(api_endpoint, str) and bool(api_endpoint.strip())


def get_bortle_sqm(
    lat: float,
    lon: float,
    api_key: str,
    api_endpoint: str,
    logger: logging.Logger,
) -> Tuple[int, float, Optional[str], bool, bool]:
    """Retrieves Bortle class and SQM for one coordinate from lightpollutionmap.info.

    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.
        api_key (str): 16-character alphanumeric key from the [secret] section.
        api_endpoint (str): QueryRaster endpoint URL.
        logger (logging.Logger): Application logger.

    Returns:
        Tuple[int, float, Optional[str], bool, bool]: Bortle class, SQM,
            an error message (None on success), whether the key looked valid,
            and whether the endpoint looked valid. On any failure the first
            two are ``0, 0``, which the caller treats as "use the defaults".
    """
    logger.info("")
    logger.info("GETTING BORTLE SCALE AND SQM VALUE")
    logger.info(f"Retrieving Bortle scale and SQM for coordinates ({lat}, {lon})")

    api_valid = is_valid_api_key(api_key)
    api_endpoint_valid = is_valid_api_endpoint(api_endpoint)

    try:
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise ValueError("Latitude and longitude must be numbers")

        if not api_valid or not api_endpoint_valid:
            logger.error("API Key or Endpoint is invalid/missing.")
            return 0, 0, "Invalid credentials", api_valid, api_endpoint_valid

        # Imported here, not at module scope: `requests` is only needed when a
        # lookup actually happens, and an install without it must still run
        # every offline path rather than failing at import time.
        import requests

        params = {
            'ql': WORLD_ATLAS_LAYER,
            'qt': 'point',
            'qd': f'{lon},{lat}',
            'key': api_key,
        }
        logger.info("Sending request to API endpoint")
        response = requests.get(api_endpoint, params=params, timeout=30)
        response.raise_for_status()

        if response.text.strip() == 'Invalid authentication.':
            logger.error("Authentication error: Missing or invalid API key")
            return 0, 0, "Authentication error", False, api_endpoint_valid

        artificial_brightness = float(response.text)
        sqm = brightness_to_sqm(artificial_brightness)
        bortle_class = sqm_to_bortle(sqm, logger)
        logger.info(f"Retrieved Bortle scale: {bortle_class}, SQM value: {sqm}")
        return bortle_class, sqm, None, api_valid, api_endpoint_valid

    except ImportError:
        logger.warning(
            "The 'requests' package is not installed, so no sky-quality lookup "
            "was attempted. Install it, or remove the [secret] section to "
            "silence this."
        )
        return 0, 0, "requests not installed", api_valid, api_endpoint_valid
    except Exception as e:
        # Deliberately broad, and it always was: a failed sky-quality lookup
        # must degrade to the configured defaults, never abort a run that has
        # already read thousands of frames.
        logger.error(f"Sky quality lookup failed: {e}")
        return 0, 0, f"Request Error: {e}", api_valid, api_endpoint_valid


def reverse_geocode(
    lat: float,
    lon: float,
    email: str,
    logger: logging.Logger,
) -> Optional[str]:
    """Turns a coordinate into a postal address via Nominatim.

    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.
        email (str): Contact address for the user agent, which Nominatim's
            terms of service require. Without one, no request is made.
        logger (logging.Logger): Application logger.

    Returns:
        Optional[str]: The address, or None if the lookup was skipped or
            failed for any reason.
    """
    if not email or not str(email).strip():
        logger.warning(
            "No EMAIL_ADDRESS in [secret]; skipping reverse geocoding "
            "(Nominatim's terms of service require a contact address)."
        )
        return None

    try:
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent=_USER_AGENT_TEMPLATE.format(email=email))
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location is None:
            logger.warning(f"No address found for ({lat}, {lon})")
            return None
        logger.info(f"Using location string: {location.address}")
        return location.address
    except ImportError:
        logger.warning(
            "The 'geopy' package is not installed, so no reverse geocoding was "
            "attempted. Install it, or remove the [secret] section to silence this."
        )
        return None
    except Exception as e:
        logger.warning(f"Geocoding error: {e}. Using default site information")
        return None


class SiteLookup:
    """Resolves unknown coordinates to a named site with real sky-quality data.

    Holds the ``[secret]`` credentials and answers one question per cluster:
    *what is this place, and how dark is it?* When it cannot answer - offline,
    no credentials, a failed request - it says so and the caller uses the
    ``[defaults]`` values, which is precisely what the whole pipeline did
    between v2.0.0 and v2.1.3.

    Attributes:
        enabled (bool): False when no usable ``[secret]`` section was supplied,
            in which case ``resolve`` is never called and no import of
            ``requests``/``geopy`` is attempted.
    """

    def __init__(self, config, logger: logging.Logger):
        """
        Args:
            config (AppConfig): The loaded configuration.
            logger (logging.Logger): Application logger.
        """
        self.config = config
        self.logger = logger

        secret = dict(config.secret or {})
        # The section is written as `KEY = ENDPOINT` pairs plus EMAIL_ADDRESS,
        # so the API key is a *key* of the section, not a value -- the shape
        # v1.4.x's config.ini.example documents and users already have on disk.
        self.email = None
        self.api_key = None
        self.api_endpoint = None
        for k, v in secret.items():
            if str(k).strip().upper() == 'EMAIL_ADDRESS':
                self.email = str(v).strip()
            elif is_valid_api_key(str(k).strip()):
                self.api_key = str(k).strip()
                self.api_endpoint = str(v).strip()

        self.enabled = bool(self.email or self.api_key)
        if not self.enabled and secret:
            self.logger.warning(
                "[secret] is present but carries neither a usable 16-character "
                "API key nor an EMAIL_ADDRESS; no network lookups will be made."
            )

    def resolve(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Looks up one coordinate.

        Args:
            lat (float): Latitude in decimal degrees.
            lon (float): Longitude in decimal degrees.

        Returns:
            Optional[Dict[str, Any]]: ``{'site': str, 'bortle': int, 'sqm': float}``
                for whatever was resolved, with defaults substituted for any
                part that was not. None when nothing at all could be resolved,
                which tells the caller to use the defaults untouched and to
                write nothing back to the configuration.
        """
        if not self.enabled:
            return None

        self.logger.info("")
        self.logger.info("PROCESSING NEW LOCATION")
        self.logger.info(f"Site location does not exist in existing sites: ({lat}, {lon})")

        defaults = self.config.defaults
        site = reverse_geocode(lat, lon, self.email, self.logger) if self.email else None

        bortle = sqm = None
        if self.api_key and self.api_endpoint:
            b, s, error_msg, _, _ = get_bortle_sqm(
                lat, lon, self.api_key, self.api_endpoint, self.logger
            )
            if error_msg:
                self.logger.warning(f"API error: {error_msg}")
            if not (b == 0 and s == 0):
                bortle, sqm = b, s

        if site is None and bortle is None:
            self.logger.warning(
                "Neither the site name nor its sky quality could be resolved; "
                "falling back to the [defaults] values."
            )
            return None

        if bortle is None:
            bortle = defaults.get(FITSKeywords.BORTLE, 4)
            sqm = defaults.get(FITSKeywords.SQM, 21.0)
            self.logger.warning(
                f"Sky quality unavailable, using defaults: Bortle {bortle}, SQM {sqm}"
            )
        if site is None:
            site = defaults.get(FITSKeywords.SITE, 'Unknown Site')

        self.logger.info(f"Processed new location: {site}, Bortle: {bortle}, SQM: {sqm}")
        return {'site': str(site), 'bortle': int(float(bortle)), 'sqm': float(sqm)}

    def save(self, site: str, lat: float, lon: float, bortle: int, sqm: float,
             filepath: str) -> bool:
        """Writes a resolved site into ``[sites]`` so it is never looked up twice.

        The one operation here that touches a file the user owns, so it is the
        most conservative: it refuses to write a site it could not name, it
        never rewrites an entry that already exists, and a failed write is
        logged and swallowed rather than losing the run's real output.

        Args:
            site (str): Resolved site name, used as the [sites] sub-section key.
            lat (float): Latitude to record.
            lon (float): Longitude to record.
            bortle (int): Bortle class to record.
            sqm (float): SQM to record.
            filepath (str): Path to the configuration file to update.

        Returns:
            bool: True if the configuration was written.
        """
        if not site or site == self.config.defaults.get(FITSKeywords.SITE):
            # Nothing was actually resolved -- writing the default site under
            # its own name would poison the database with a bogus entry.
            return False
        try:
            from configobj import ConfigObj

            cfg = ConfigObj(filepath, encoding='utf-8')
            # The loader lower-cases section names for lookup, but the file on
            # disk keeps whatever case the user wrote, so match case-insensitively
            # rather than creating a second, differently-cased [sites] section.
            section_name = next(
                (k for k in cfg.keys() if k.lower() == ConfigSections.SITES), None
            )
            if section_name is None:
                section_name = ConfigSections.SITES
                cfg[section_name] = {}
            if site in cfg[section_name]:
                return False

            cfg[section_name][site] = {
                'latitude': lat,
                'longitude': lon,
                'bortle': bortle,
                'sqm': sqm,
            }
            cfg.write()
            self.logger.info(f"Saved new site to {filepath}: {site}")
            print(f"New observing site recorded in {filepath}: {site}")
            return True
        except Exception as e:
            self.logger.error(f"Could not save the new site to {filepath}: {e}")
            return False
