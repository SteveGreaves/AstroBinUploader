"""
Configuration Loader Module - AstroBin Upload Utility v2.0.2

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
            filters=normalized.get(ConfigSections.FILTERS, {}),
            sites=normalized.get(ConfigSections.SITES, {}),
            use_obs_date=use_obs_date
        )

    def _generate_default_config(self, filepath: str):
        """Creates a fresh config.ini with standard templates."""
        config = ConfigObj(filepath, encoding='utf-8')
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
            'ROTATOR': 'None',
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
        config[ConfigSections.OVERRIDE] = {
            'SITE': 'SITENAME',
            'EXPOSURE': 'EXPTIME',
            'INSTRUME': 'CAMERA_MODEL'
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
            'lightpollution_api_key': 'xxxxxxxxxxxxx',
            'EMAIL_ADDRESS': 'id@provider.com'
        }
        config[ConfigSections.SITES] = {}
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
