"""
Geocoding & Site Management Module - AstroBin Upload Utility v2.0.2

This module is responsible for enriching the session metadata with 
geographical site information. It performs two primary functions:

1.  **Coordinate Propagation**: Ensures that calibration frames (which often 
    lack GPS data) inherit the coordinates of the closest Light frame.
2.  **Site Identification**: Maps numerical coordinates to human-readable 
    site names, Bortle scales, and SQM values using the local sites database 
    stored in 'config.ini'.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
from models import SessionState
from constants import InternalColumns, ImageType, ConfigSections

# Mean Earth radius in metres (IUGG value), used by the haversine distance
# below.
EARTH_RADIUS_M = 6371000.0

# Cluster radius for Smart Proximity Clustering: GPS readings within this
# distance of each other are treated as the same physical site. ~110m was
# the value documented (but not, before A4 in REMEDIATION_PLAN.md, actually
# achieved) by the original flat-degree Euclidean approximation.
CLUSTER_RADIUS_M = 110.0

def _haversine_distance_m(lat1, lon1, lat2, lon2):
    """
    Vectorized great-circle distance in metres (haversine formula).

    Accepts scalars or numpy/pandas arrays (elementwise, via numpy
    broadcasting). Verified against geopy.distance.distance (Karney's
    geodesic) to agree within ~0.35m across equatorial, mid-latitude and
    high-latitude test points -- comfortably precise enough for a 110m
    clustering threshold, without the per-call overhead a Python loop of
    geopy calls would carry across potentially thousands of frames (A4 in
    REMEDIATION_PLAN.md).

    A flat lat/lon Euclidean distance (the previous approach) treats a
    degree of longitude as equal to a degree of latitude, which is only
    true at the equator; at 52N (this project's reference test site) a
    degree of longitude is roughly 38% shorter than a degree of latitude,
    and the discrepancy grows with latitude. That silently narrowed the
    effective clustering radius the further north (or south) the data was.
    """
    lat1r, lon1r, lat2r, lon2r = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_M * c

class GeocodeStep:
    """
    Step responsible for coordinate alignment and site metadata lookup.
    """
    def execute(self, state: SessionState) -> SessionState:
        """
        Executes the geocoding enrichment logic using Distance-Based Clustering.

        Args:
            state (SessionState): The current pipeline state.
            
        Returns:
            SessionState: The state with enriched site metadata.
        """
        logger = logging.getLogger("AstroBinV2")
        logger.info("Identifying geographical site data using Smart Proximity Clustering")
        
        df = state.processed_df
        if df.empty: return state

        config = state.config

        # --- Stage 1: Coordinate Propagation ---
        # Calibration frames often miss SITELAT/SITELONG headers. We force 
        # them to align with the nearest Light frames.
        df = self._align_coordinates(df, logger)

        # --- Stage 2: Smart Proximity Clustering ---
        # Instead of arbitrary rounding, we find clusters of coordinates
        # that are close to each other (within CLUSTER_RADIUS_M) and treat
        # them as one site.

        # 1. Identify all unique coordinate pairs from all frames
        coords_df = df[[InternalColumns.SITE_LAT, InternalColumns.SITE_LONG]].copy()
        coords_df[InternalColumns.SITE_LAT] = pd.to_numeric(coords_df[InternalColumns.SITE_LAT], errors='coerce').fillna(0.0)
        coords_df[InternalColumns.SITE_LONG] = pd.to_numeric(coords_df[InternalColumns.SITE_LONG], errors='coerce').fillna(0.0)

        unique_coords = coords_df.drop_duplicates().reset_index(drop=True)
        unique_coords['site_cluster'] = -1

        # 2. Distance-Based Grouping (greedy single-linkage approach)
        cluster_id = 0
        for i in range(len(unique_coords)):
            if unique_coords.at[i, 'site_cluster'] == -1:
                # Start a new cluster, seeded by this point.
                unique_coords.at[i, 'site_cluster'] = cluster_id

                lat_ref = unique_coords.at[i, InternalColumns.SITE_LAT]
                lon_ref = unique_coords.at[i, InternalColumns.SITE_LONG]

                # Find all other points within the threshold, in real metres.
                dist = _haversine_distance_m(
                    unique_coords[InternalColumns.SITE_LAT], unique_coords[InternalColumns.SITE_LONG],
                    lat_ref, lon_ref
                )

                # Only claim points not already claimed by an earlier
                # cluster. A3 in REMEDIATION_PLAN.md: this mask previously
                # had no '== -1' term, so a point already assigned to
                # cluster 0 could be reassigned to cluster 1 by a later
                # seed whose radius also reached it -- stripping cluster 0
                # down to a single un-averaged point and defeating the
                # whole purpose of centroid averaging. Restricting to
                # unclaimed points makes this a standard (order-dependent,
                # but internally consistent) greedy single-linkage
                # clustering: once a point joins a cluster, it stays.
                claim_mask = (dist < CLUSTER_RADIUS_M) & (unique_coords['site_cluster'] == -1)
                unique_coords.loc[claim_mask, 'site_cluster'] = cluster_id
                cluster_id += 1

        # 3. Calculate Cluster Centroids (Averaging)
        # This solves the precision issue by using the mean of all drift readings.
        centroids = unique_coords.groupby('site_cluster').agg({
            InternalColumns.SITE_LAT: 'mean',
            InternalColumns.SITE_LONG: 'mean'
        }).reset_index()
        
        # 4. Map Clusters back to the primary DataFrame
        # Merge clusters to unique_coords, then merge that to the main df
        df = df.merge(
            unique_coords[[InternalColumns.SITE_LAT, InternalColumns.SITE_LONG, 'site_cluster']], 
            on=[InternalColumns.SITE_LAT, InternalColumns.SITE_LONG], 
            how='left'
        )
        
        # Replace original drifting coords with stable Centroid coords
        df = df.drop(columns=[InternalColumns.SITE_LAT, InternalColumns.SITE_LONG]).merge(
            centroids, on='site_cluster', how='left'
        )

        # --- Stage 3: Site Identification (Database Lookup) ---
        sites_db = pd.DataFrame(config.sites).transpose()
        
        # Iterate through distinct clusters to assign site metadata
        cluster_results = {}
        for cid, group in df.groupby('site_cluster'):
            avg_lat = group[InternalColumns.SITE_LAT].iloc[0]
            avg_lon = group[InternalColumns.SITE_LONG].iloc[0]
            
            # Fuzzy Coordinate Lookup in the local DB
            site_info = self._find_site_in_db(sites_db, avg_lat, avg_lon, state.config.precision)
            
            if site_info is not None:
                cluster_results[cid] = {
                    InternalColumns.SITE_NAME: str(site_info.name),
                    InternalColumns.BORTLE: int(site_info.get('bortle', config.defaults.get('BORTLE', 4))),
                    InternalColumns.MEAN_SQM: float(site_info.get('sqm', config.defaults.get('SQM', 21.0)))
                }
            else:
                cluster_results[cid] = {
                    InternalColumns.SITE_NAME: str(config.defaults.get('SITE', 'Unknown Site')),
                    InternalColumns.BORTLE: int(config.defaults.get('BORTLE', 4)),
                    InternalColumns.MEAN_SQM: float(config.defaults.get('SQM', 21.0))
                }
                logger.debug(f"Site Cluster {cid}: No DB match for averaged coords ({avg_lat:.4f}, {avg_lon:.4f}). Used defaults.")

        # Final Assignment
        for cid, metadata in cluster_results.items():
            mask = df['site_cluster'] == cid
            df.loc[mask, InternalColumns.SITE_NAME] = metadata[InternalColumns.SITE_NAME]
            df.loc[mask, InternalColumns.BORTLE] = metadata[InternalColumns.BORTLE]
            df.loc[mask, InternalColumns.MEAN_SQM] = metadata[InternalColumns.MEAN_SQM]

        logger.info(f"Consolidated GPS drift into {cluster_id} unique imaging site(s).")
        
        # Cleanup internal tracking columns before returning
        state.processed_df = df.drop(columns=['site_cluster'])
        return state

    def _align_coordinates(self, df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
        """
        Ensures all frames have coordinates by aligning them to the closest Light frame.
        
        This prevents 'Unknown Site' errors for calibration frames that were 
        captured as part of the same session but lack GPS metadata.
        """
        lights = df[df[InternalColumns.IMAGE_TYPE] == ImageType.LIGHT.value].copy()
        if lights.empty: return df

        # Ensure coordinates are numeric for distance calculations
        lights[InternalColumns.SITE_LAT] = pd.to_numeric(lights[InternalColumns.SITE_LAT], errors='coerce')
        lights[InternalColumns.SITE_LONG] = pd.to_numeric(lights[InternalColumns.SITE_LONG], errors='coerce')

        for i, row in df[df[InternalColumns.IMAGE_TYPE] != ImageType.LIGHT.value].iterrows():
            try:
                plat = pd.to_numeric(row[InternalColumns.SITE_LAT], errors='coerce')
                plon = pd.to_numeric(row[InternalColumns.SITE_LONG], errors='coerce')
                
                if pd.isna(plat) or pd.isna(plon):
                    # Direct fallback: If completely missing, use the first Light frame's location
                    df.at[i, InternalColumns.SITE_LAT] = lights[InternalColumns.SITE_LAT].iloc[0]
                    df.at[i, InternalColumns.SITE_LONG] = lights[InternalColumns.SITE_LONG].iloc[0]
                else:
                    # Geodesic Distance Match: Find the Light frame geographically
                    # closest to this calibration frame. A4 in
                    # REMEDIATION_PLAN.md: a flat lat/lon Euclidean distance
                    # here treats a degree of longitude as equal to a degree
                    # of latitude, which only holds at the equator -- see
                    # _haversine_distance_m's docstring above.
                    dist = _haversine_distance_m(lights[InternalColumns.SITE_LAT], lights[InternalColumns.SITE_LONG], plat, plon)
                    closest = dist.idxmin()
                    df.at[i, InternalColumns.SITE_LAT] = lights.at[closest, InternalColumns.SITE_LAT]
                    df.at[i, InternalColumns.SITE_LONG] = lights.at[closest, InternalColumns.SITE_LONG]
            except Exception: pass
        return df

    def _find_site_in_db(self, db: pd.DataFrame, lat: float, lon: float, precision: int) -> Optional[pd.Series]:
        """
        Performs a fuzzy coordinate lookup in the sites database.
        
        Args:
            db (pd.DataFrame): The site database from config.
            lat (float): Latitude to search for.
            lon (float): Longitude to search for.
            precision (int): Number of decimal places to match.

        Returns:
            Optional[pd.Series]: The matching site row, or None.
        """
        if db.empty: return None
        
        try:
            # Cast DB coordinates to numeric for rounding
            db_lat = pd.to_numeric(db['latitude'], errors='coerce')
            db_lon = pd.to_numeric(db['longitude'], errors='coerce')
            
            # Find a match within the specified decimal precision
            mask = (db_lat.round(precision) == round(lat, precision)) & \
                   (db_lon.round(precision) == round(lon, precision))
            
            matches = db[mask]
            if not matches.empty:
                return matches.iloc[0]
        except Exception: pass
        return None