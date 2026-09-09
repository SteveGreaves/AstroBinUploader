"""
Configuration Loader Module - AstroBin Upload Utility v2.1.1

Responsible for the ingestion and normalization of the 'config.ini' file. 
This module ensures that the application's runtime settings are correctly 
mapped into the strongly-typed AppConfig model, providing validation and 
standardization for user-provided settings.
"""

import logging
from typing import Dict, Any, List
from configobj import ConfigObj
import os
from models import AppConfig
from constants import ConfigSections, InternalColumns

# The set of column names an [override] target can actually reach. Every
# override key ends up as a raw column name (base.py's NormalizeHeadersStep
# Stage 1: `df[internal_key] = combined_series`), which is then lowercased
# along with every other column (Stage 3). So an override only has any
# effect if its key, lowercased, matches a name later pipeline code reads --
# i.e. an InternalColumns value. Anything else silently creates a phantom
# column nothing consumes (A6 in REMEDIATION_PLAN.md: '[override]
# SWCREATOR = CREATOR' in the shipped config -- 'swcreator', not
# 'swcreate' -- has never done anything).
_KNOWN_OVERRIDE_TARGETS = {
    v.lower() for k, v in vars(InternalColumns).items()
    if not k.startswith('_') and isinstance(v, str)
}

class ConfigLoader:
    """
    Manages the loading and type-mapping of application configuration.
    """
    def __init__(self, logger: logging.Logger):
        """
        Initializes the loader.

        Args:
            logger (logging.Logger): Active application logger.
        """
        self.logger = logger

    def load(self, filepath: str) -> AppConfig:
        """
        Loads the config.ini and maps its hierarchical sections to AppConfig.
        
        This method performs key normalization (e.g., stripping spaces) and 
        type conversion (e.g., strings to booleans).

        Args:
            filepath (str): Path to the config file (e.g., 'config.ini').
            
        Returns:
            AppConfig: A validated configuration object.
            
        Raises:
            FileNotFoundError: If the configuration file cannot be found.
        """
        if not os.path.exists(filepath):
            if filepath == 'config.ini':
                self.logger.info("config.ini missing. Generating default configuration template.")
                self._generate_default_config(filepath)
                print(f"\nA new {filepath} file was created. Please edit this before re-running the script.")
                import sys
                sys.exit(0)
            else:
                self.logger.error(f"Custom configuration file missing: {filepath}")
                raise FileNotFoundError(f"The specified configuration file '{filepath}' was not found.")

        # Load the raw INI file using ConfigObj for better section management
        config_obj = ConfigObj(filepath, encoding='utf-8')
        
        # Normalize top-level section keys to lowercase for consistent dictionary access
        normalized = {k.lower(): v for k, v in config_obj.items()}
        
        # Extract and validate the 'USEOBSDATE' flag (Default: True)
        defaults_sec = normalized.get(ConfigSections.DEFAULTS, {})
        use_obs_date = str(defaults_sec.get('USEOBSDATE', 'True')).lower() == 'true'
        
        self.logger.info(f"Configuration loaded and normalized from {filepath}")
        
        return AppConfig(
            defaults=self._normalize_defaults(defaults_sec),
            overrides=self._normalize_overrides(normalized.get(ConfigSections.OVERRIDE, {})),
            equipment_overrides=self._normalize_equipment_overrides(
                normalized.get(ConfigSections.EQUIPMENT_OVERRIDES, {})
            ),
            filters=normalized.get(ConfigSections.FILTERS, {}),
            sites=normalized.get(ConfigSections.SITES, {}),
            # Optional. Absent -> empty dict -> SiteLookup stays offline and the
            # pipeline behaves exactly as it did in v2.0.0-v2.1.3.
            secret=normalized.get(ConfigSections.SECRET, {}),
            use_obs_date=use_obs_date
        )

    def _generate_default_config(self, filepath: str):
        """Creates a fresh config.ini with standard templates."""
        config = ConfigObj(filepath, encoding='utf-8')
        # A generated config is the first thing a new user is told to edit, so
        # it explains itself. Comments attach to sections and do not affect
        # parsing; the values below are unchanged.
        config.initial_comment = [
            "# AstroBinUpload configuration.",
            "#",
            "# Generated automatically on first run. Edit to match your own",
            "# equipment and site, then run the script again with your data",
            "# directory. Keep a backup once it is personalised.",
            "#",
            "# Full documentation of every section is in the README.",
            "",
        ]
        config[ConfigSections.DEFAULTS] = {
            'IMAGETYP': 'LIGHT',
            'EXPOSURE': 0.0,
            'DATE-OBS': '2023-01-01',
            'XBINNING': 1,
            'GAIN': -1,
            'EGAIN': -1,
            'INSTRUME': 'None',
            'TELESCOP': 'None',
            'FOCNAME': 'None',
            'FWHEEL': 'None',
            'ROTNAME': 'None',
            'ROTANTANG': 0,
            'XPIXSZ': 3.76,
            'CCD-TEMP': -10,
            'FOCALLEN': 500,
            'FOCRATIO': 5.0,
            'SITE': 'Unknown Site',
            'SITELAT': 0.0,
            'SITELONG': 0.0,
            'BORTLE': 4,
            'SQM': 21.0,
            'FILTER': 'No Filter',
            'OBJECT': 'No target',
            'FOCTEMP': 20,
            'HFR': 1.6,
            'SWCREATE': 'Unknown package',
            'USEOBSDATE': 'True'
        }
        # Keyword-remap defaults. Kept in sync with config.ini.example.
        # SQM/FOCTEMP aliases resolve the common ASCOM Observing-Conditions
        # keywords out of the box (GitHub #6); FOCNAME/SWCREATE cover the
        # N.I.N.A. spellings.
        config[ConfigSections.OVERRIDE] = {
            'SITE': 'SITENAME',
            'EXPOSURE': 'EXPTIME',
            'INSTRUME': 'CAMERA_MODEL',
            'FOCNAME': 'FOCUSER',
            'SWCREATE': 'CREATOR',
            'SQM': 'AOCSKYQ, AOCSKYQU',
            'FOCTEMP': 'AOCAMBT'
        }
        # Forced display values (GitHub #5). 'None' means "leave as found";
        # set e.g. FOCNAME = ZWO EAF to override for every frame.
        config[ConfigSections.EQUIPMENT_OVERRIDES] = {
            'INSTRUME': 'None',
            'TELESCOP': 'None',
            'FOCNAME': 'None',
            'FWHEEL': 'None',
            'ROTNAME': 'None'
        }
        config[ConfigSections.FILTERS] = {
            'Ha': 4663,
            'SII': 4844,
            'OIII': 4752,
            'Red': 4649,
            'Green': 4643,
            'Blue': 4637,
            'Lum': 2906
        }
        config[ConfigSections.SECRET] = {
            'YOUR_API_KEY': 'https://www.lightpollutionmap.info/QueryRaster/',
            'EMAIL_ADDRESS': 'your_email@example.com',
        }
        config[ConfigSections.SITES] = {}
        # Section comments are attached here, after every section exists:
        # ConfigObj indexes inline_comments by section as each is created, so
        # setting a comment for a section that has not been added yet makes
        # write() fail with a KeyError on that name.

        # These codes are not generic. They are the author's own Astronomik
        # 2 inch round filters, named as N.I.N.A. writes them: an AstroBin
        # filter ID identifies a specific product, so the same filter name in
        # another brand, size or mounting has a different ID. Say so in the
        # generated file, or a user keeps the defaults and silently uploads
        # someone else's filters.
        config.comments[ConfigSections.FILTERS] = [
            '',
            '# Filter name -> AstroBin filter ID.',
            '# These defaults are the author\'s own Astronomik 2 inch round',
            '# filters, named as N.I.N.A. writes them. An AstroBin ID identifies',
            '# a specific filter product, so the same name in a different brand,',
            '# size or mounting has a different ID -- replace these with your own.',
            '# See "Finding AstroBin\'s Numeric ID for Filters" in the README.',
        ]
        config.comments[ConfigSections.DEFAULTS] = [
            "# Values used when a header is missing, or when a frame leaves the",
            "# field blank. These are what a frame falls back to, so set them to",
            "# your own equipment and site rather than leaving the placeholders.",
            "#",
            "# USEOBSDATE = True  aggregate by each frame's calendar date.",
            "#             False  frames taken after midnight count with the",
            "#                    session that started the previous evening.",
        ]
        config.comments[ConfigSections.OVERRIDE] = [
            "",
            "# Read a non-standard header keyword as a standard one:",
            "#   STANDARD_NAME = YOUR_KEYWORD",
            "# A comma-separated list tries each keyword in turn, which is how",
            "# one config covers several capture packages.",
        ]
        config.comments[ConfigSections.EQUIPMENT_OVERRIDES] = [
            "",
            "# Force a display value for equipment fields when the header text",
            "# is unhelpful -- N.I.N.A. writes \"EAF\" where AstroBin expects",
            "# \"ZWO EAF\", for example. \"None\" leaves the value as found.",
        ]
        config.comments[ConfigSections.SECRET] = [
            "",
            "# Sky quality API key and endpoint, plus your e-mail address.",
            "# Only the API key is to be edited: replace YOUR_API_KEY with the",
            "# key itself. The key is obtained by e-mailing the owner of",
            "# lightpollutionmap.info -- see the README.",
            "#",
            "# EMAIL_ADDRESS is sent to the reverse-geocoding provider as a",
            "# courtesy so they can see who is using their API, and must be a",
            "# real address.",
            "#",
            "# Until a valid key is set, Bortle and SQM come from [defaults]",
            "# BORTLE and SQM; if the address lookup fails, the site details",
            "# come from [defaults] SITE, SITELAT and SITELONG. The run always",
            "# completes either way.",
        ]
        config.comments[ConfigSections.SITES] = [
            "",
            "# Written by the program as each new site is resolved, so a site is",
            "# looked up once and never again. You do not normally edit this,",
            "# but a remote site can be added by hand if the lookup cannot run.",
        ]
        # Generated with placeholders, exactly as v1.4.x's get_default_ini_string
        # did: the section is part of a standard configuration, not an extra.
        # Until the user substitutes real credentials the lookups in
        # engine/sites.py cannot run and the [defaults] values are used, so a
        # freshly generated config still works -- it just cannot name a site it
        # has never seen. Restored in v2.2.0 along with the lookups themselves.

        config.write()

    def _normalize_defaults(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardizes keys within the [defaults] section.
        
        Converts keys like 'Image Type' or 'exposure' into standardized 
        uppercase FITS-style keys like 'IMAGETYP' and 'EXPOSURE'.
        """
        return {k.upper().replace(' ', ''): v for k, v in defaults.items()}

    def _normalize_overrides(self, overrides: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Processes the [override] section for keyword remapping.

        Handles both single keyword strings and comma-separated lists,
        ensuring they are all returned as lists of strings.
        """
        result = {}
        for k, v in overrides.items():
            if isinstance(v, list):
                # ConfigObj parses a comma-separated INI value (e.g.
                # 'SQM = AOCSKYQ, AOCSKYQU') into a native Python list
                # *before* this method ever sees it -- this branch was
                # previously unreachable in the isinstance(v, str) check
                # below, so any such value fell through to the scalar
                # branch and got str()'d whole: ["['AOCSKYQ', 'AOCSKYQU']"],
                # a single-element list containing a Python repr that can
                # never match a real column name. That override was
                # therefore entirely dead (A6 in REMEDIATION_PLAN.md).
                result[k] = [str(item).strip() for item in v]
            elif isinstance(v, str):
                # Split comma-separated values and strip whitespace
                result[k] = [item.strip() for item in v.split(',')]
            else:
                # Handle single non-string values by wrapping them in a list
                result[k] = [str(v).strip()]

            if k.lower() not in _KNOWN_OVERRIDE_TARGETS:
                self.logger.warning(
                    f"[override] target '{k}' does not match any recognized "
                    f"internal column and will have no effect. Check for a "
                    f"typo (did you mean one of: "
                    f"{', '.join(sorted(_KNOWN_OVERRIDE_TARGETS))}?)"
                )
        return result

    def _normalize_equipment_overrides(self, raw: Dict[str, Any]) -> Dict[str, str]:
        """
        Processes the [equipmentoverrides] section (GitHub #5).

        Each entry forces a literal value into a named internal column for
        every row, applied after default injection. An entry whose value is
        blank or the sentinel 'None' is treated as "no override" and
        skipped, so the generated template can list every overridable field
        without changing behaviour until the user edits one.
        """
        result = {}
        for k, v in raw.items():
            val = str(v).strip() if v is not None else ''
            if val == '' or val.lower() == 'none':
                continue
            key = k.upper().replace(' ', '')
            if key.lower() not in _KNOWN_OVERRIDE_TARGETS:
                self.logger.warning(
                    f"[equipmentoverrides] target '{k}' does not match any "
                    f"recognized internal column and will have no effect."
                )
            result[key] = val
        return result
