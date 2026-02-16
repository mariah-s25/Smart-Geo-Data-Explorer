def analyze_geometry(gdf):
    """
    Analyze a GeoDataFrame and return key spatial info as a dictionary.
    """
    info = {}
    
    # Geometry type
    info['geometry_type'] = gdf.geom_type.unique()
    
    # CRS
    info['crs'] = gdf.crs
    
    # Feature count
    info['feature_count'] = len(gdf)
    
    # Bounding box
    info['bounds'] = gdf.total_bounds  # returns [minx, miny, maxx, maxy]
    
    # Spatial extent (area)
    if gdf.geom_type.isin(['Polygon','MultiPolygon']).any():
        # calculate total area in km²
        gdf_projected = gdf.to_crs(epsg=3857)  # project to meters
        total_area_m2 = gdf_projected.geometry.area.sum()
        info['total_area_km2'] = total_area_m2 / 1e6
    else:
        info['total_area_km2'] = None
    
    return info
