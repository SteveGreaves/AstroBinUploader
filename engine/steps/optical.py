"""
Optical Parameter Calculation Module - AstroBin Upload Utility v2.0.2

This module calculates critical optical metrics for Light frames that are 
required by AstroBin's technical cards.

Metrics Calculated:
1.  **HFR (Half Flux Radius)**: Extracted from capture software filenames 
    (e.g., N.I.N.A. formatted names) or taken from defaults.
2.  **Image Scale**: Calculated using the focal length and pixel size 
    (arcsec/pixel). Formula: (PixelSize / FocalLength) * 206.265.
3.  **FWHM (Full Width at Half Maximum)**: Derived from the HFR and 
    Image Scale. Formula: HFR * ImageScale * 2.
"""

import pandas as pd
import re
from models import SessionState
from constants import ImageType, InternalColumns

class OpticalParameterStep:
    """
    Step responsible for deriving resolution and star size metrics.
    """
    def execute(self, state: SessionState) -> SessionState:
        """
        Processes Light frames to calculate or extract optical parameters.

        Args:
            state (SessionState): The current pipeline state.
            
        Returns:
            SessionState: The state with populated HFR, FWHM, and Imscale.
        """
        import logging
        logger = logging.getLogger("AstroBinV2")
        logger.info("Processing optical parameters and calculating star metrics")
        
        df = state.processed_df
        if df.empty: return state

        # Use the default HFR value from config as a fallback
        hfr_default = float(state.config.defaults.get('HFR', 1.0))
        
        # We only calculate optical metrics for Light frames
        mask = df[InternalColumns.IMAGE_TYPE] == ImageType.LIGHT.value
        if not mask.any(): return state

        def extract_metrics(row):
            """Internal worker function to process a single frame row."""
            
            # 1. HFR Extraction
            # Many capture tools (like N.I.N.A.) can be configured to put 
            # the HFR in the filename. We attempt to parse this.
            fname = str(row[InternalColumns.FILENAME])
            hfr_match = re.search(r'HFR_([0-9.]+)', fname)
            hfr = float(hfr_match.group(1)) if hfr_match and float(hfr_match.group(1)) > 0 else hfr_default
            
            # 2. Image Scale (arcsec/pixel)
            # Standard Formula: (PixelSize in microns / FocalLength in mm) * 206.265
            try:
                flen = float(row[InternalColumns.FOCAL_LENGTH])
                pix = float(row[InternalColumns.PIXEL_SIZE])
                # Ensure we don't divide by zero if FocalLength is missing or 0
                imscale = pix / flen * 206.265 if flen > 0 else 1.0
            except (ValueError, ZeroDivisionError, TypeError):
                imscale = 1.0
                
            # 3. FWHM Calculation
            # FWHM (Full Width at Half Maximum) is approximately HFR * 2. 
            # We multiply by image scale to convert it to arcseconds.
            fwhm = hfr * imscale * 2 if hfr >= 0.0 else 0.0
            
            return pd.Series({
                InternalColumns.HFR: round(hfr, 2),
                InternalColumns.IMSCALE: round(imscale, 2),
                InternalColumns.MEAN_FWHM: round(fwhm, 2)
            })

        # Apply the calculations to the subset of Light frames
        total_lights = len(df[mask])
        results_list = []
        
        for i, (idx, row) in enumerate(df[mask].iterrows(), 1):
            results_list.append(extract_metrics(row))
            print(f"\rProcessing optical metrics: {i} of {total_lights}...", end="", flush=True)
        
        if total_lights > 0:
            print("\n") # Ensure newline after progress completion

        # Reintegrate the calculated results into the main dataframe
        if results_list:
            results = pd.DataFrame(results_list, index=df[mask].index)
            for col in results.columns:
                df.loc[mask, col] = results[col]
            
        state.processed_df = df
        return state
