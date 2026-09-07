"""
Optical Parameter Calculation Module - AstroBin Upload Utility v2.0.3

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
import logging
from models import SessionState
from constants import ImageType, InternalColumns

# arcsec/pixel = (pixel size in microns / focal length in mm) * ARCSEC_PER_RADIAN
# 206265 is the number of arcseconds in one radian; the standard plate-scale
# formula uses it directly because pixel size (microns) and focal length
# (mm) already carry a compensating factor-of-1000 unit difference.
ARCSEC_PER_RADIAN = 206.265

# FWHM (Full Width at Half Maximum) is approximated here as twice the HFR
# (Half Flux Radius) -- a common rule-of-thumb relationship between the two
# star-size metrics, not a precise physical derivation.
FWHM_TO_HFR_RATIO = 2.0


def _python_round(series: pd.Series, ndigits: int = 2) -> pd.Series:
    """
    Round a Series using Python's builtin round(), not pandas'/numpy's
    .round().

    Builtin round() is decimal-correct round-half-to-even (CPython's
    correctly-rounded dtoa-based algorithm); pandas' .round() is numpy's
    multiply-rint-divide on the binary double representation. They disagree
    at real boundary values -- e.g. round(2.675, 2) == 2.67 (builtin) vs
    pd.Series([2.675]).round(2) == 2.68 -- and the disagreement survives
    '%.2f' formatting into the session summary and acquisition CSV (A12 in
    REMEDIATION_PLAN.md). Kept as .apply() rather than a numpy formula
    because builtin round()'s correctness comes from CPython's dtoa
    algorithm operating on the true decimal value, not a vectorizable
    arithmetic identity.
    """
    return series.apply(lambda x: round(x, ndigits) if pd.notna(x) else x)


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
        logger = logging.getLogger("AstroBinV2")
        logger.info("Processing optical parameters and calculating star metrics")

        df = state.processed_df
        if df.empty: return state

        # Use the default HFR value from config as a fallback
        hfr_default = float(state.config.defaults.get('HFR', 1.0))

        # We only calculate optical metrics for Light frames
        mask = df[InternalColumns.IMAGE_TYPE] == ImageType.LIGHT.value
        if not mask.any(): return state

        lights = df.loc[mask]

        # 1. HFR Extraction (vectorized)
        # Many capture tools (like N.I.N.A.) can be configured to put the
        # HFR in the filename. We attempt to parse this. Semantics
        # preserved exactly from the row-wise version: a non-match or a
        # non-positive parsed value both fall back to hfr_default --
        # '.where(cond, other)' treats a NaN condition (no match) as False
        # exactly like a numeric failed comparison, so both paths collapse
        # into one vectorized call.
        hfr_extracted = pd.to_numeric(
            lights[InternalColumns.FILENAME].astype(str).str.extract(r'HFR_([0-9.]+)', expand=False),
            errors='coerce'
        )
        hfr = hfr_extracted.where(hfr_extracted > 0, hfr_default)

        # 2. Image Scale (arcsec/pixel), vectorized.
        # Standard Formula: (PixelSize in microns / FocalLength in mm) * 206.265
        # Semantics preserved: the original try/except only ever caught a
        # genuinely non-numeric header value (float(x) on an existing NaN
        # succeeds and silently propagates, it doesn't raise) -- so
        # pd.to_numeric(errors='coerce') reproduces the same "NaN survives
        # if focal length is valid" behaviour. Only flen<=0 or flen
        # unparseable falls back to the 1.0 default, matching the original
        # 'if flen > 0 else 1.0' guard exactly (NaN > 0 is False either way).
        flen = pd.to_numeric(lights[InternalColumns.FOCAL_LENGTH], errors='coerce')
        pix = pd.to_numeric(lights[InternalColumns.PIXEL_SIZE], errors='coerce')
        imscale = (pix / flen * ARCSEC_PER_RADIAN).where(flen > 0, 1.0)

        # 3. FWHM Calculation, vectorized.
        # FWHM (Full Width at Half Maximum) is approximately HFR * 2. We
        # multiply by image scale to convert it to arcseconds.
        fwhm = (hfr * imscale * FWHM_TO_HFR_RATIO).where(hfr >= 0.0, 0.0)

        # 4. Rounding -- via _python_round, not pandas'/numpy's .round(),
        # to avoid disagreeing at exact decimal boundaries (A12).
        df.loc[mask, InternalColumns.HFR] = _python_round(hfr)
        df.loc[mask, InternalColumns.IMSCALE] = _python_round(imscale)
        df.loc[mask, InternalColumns.MEAN_FWHM] = _python_round(fwhm)

        logger.debug(f"Computed optical metrics for {len(lights)} light frame(s).")

        state.processed_df = df
        return state
