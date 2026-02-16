import geopandas as gpd
import pandas as pd
import tempfile
import zipfile
import os
from shapely.geometry import Point


def load_spatial_data(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return _load_csv(uploaded_file)

    elif file_name.endswith(".geojson") or file_name.endswith(".json"):
        return _load_geojson(uploaded_file)

    elif file_name.endswith(".zip"):
        return _load_shapefile_zip(uploaded_file)

    else:
        raise ValueError("Unsupported file format")


def _load_csv(uploaded_file):

    df = pd.read_csv(uploaded_file)
    columns = df.columns

    lat_candidates = ['lat', 'latitude', 'y']
    lon_candidates = ['lon', 'lng', 'longitude', 'x']

    lat_col, lon_col = None, None

    for col in columns:
        if col.lower() in lat_candidates:
            lat_col = col
        elif col.lower() in lon_candidates:
            lon_col = col

    if lat_col is None or lon_col is None:
        raise ValueError("Latitude / Longitude columns not found")

    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    return gdf


def _load_geojson(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    gdf = gpd.read_file(tmp_path)
    return gdf


def _load_shapefile_zip(uploaded_file):

    with tempfile.TemporaryDirectory() as tmpdir:

        zip_path = os.path.join(tmpdir, "data.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]

        if not shp_files:
            raise ValueError("No .shp file found inside ZIP")

        shp_path = os.path.join(tmpdir, shp_files[0])

        gdf = gpd.read_file(shp_path)

    return gdf
