"""
Models Module - AstroBin Upload Utility v2.1.0

This module defines the core data structures and state containers used 
throughout the application. By employing Python Dataclasses and strong 
typing, we ensure that configuration data and session state are 
well-defined, predictable, and self-documenting.

Core Components:
- **AppConfig**: A typed representation of the config.ini settings.
- **SessionState**: The 'Context' object that flows through the processing pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import pandas as pd

@dataclass
class AppConfig:
    """
    Strongly typed representation of the application configuration.
    
    This class maps the hierarchical sections of 'config.ini' into a 
    structured object, facilitating safe access to settings in the pipeline.
    
    Attributes:
        defaults (Dict[str, Any]): Fallback values for missing metadata (e.g., SITE, BORTLE).
        overrides (Dict[str, List[str]]): User-defined hardware keyword mappings.
        equipment_overrides (Dict[str, str]): Forced literal values for named columns
            (e.g. focname = ZWO EAF), applied after defaults. GitHub #5.
        filters (Dict[str, Any]): Map of filter names to AstroBin numeric IDs.
        sites (Dict[str, Dict[str, Any]]): Database of previously geocoded site coordinates.
        use_obs_date (bool): If True, use calendar capture date; if False, shift overnight sessions.
        precision (int): Decimal precision for coordinate rounding and fuzzy matching.
    """
    defaults: Dict[str, Any]
    overrides: Dict[str, List[str]]
    filters: Dict[str, Any]
    sites: Dict[str, Dict[str, Any]]
    use_obs_date: bool = True
    precision: int = 4
    equipment_overrides: Dict[str, str] = field(default_factory=dict)

@dataclass
class SessionState:
    """
    The state container that flows through the processing pipeline.
    
    Acts as the 'Context' object, carrying DataFrames and configuration between 
    independent transformation steps. This separation of state from logic 
    is a key feature of the Pipeline Pattern.
    
    Attributes:
        config (AppConfig): The active application configuration.
        raw_df (pd.DataFrame): Unfiltered header data exactly as extracted from files.
        processed_df (pd.DataFrame): Normalized and cleaned data undergoing transformations.
        aggregated_df (pd.DataFrame): Final session-level statistics grouped for export.
        total_images_scanned (int): Running counter for the number of files identified.
    """
    config: AppConfig
    raw_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    processed_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    aggregated_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_images_scanned: int = 0
