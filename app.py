import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from branca.colormap import LinearColormap
import geopandas as gpd
from shapely.geometry import Point
import io
from datetime import datetime

from engine.loader import load_spatial_data
from engine.geometry import analyze_geometry

# Page config with custom theme
st.set_page_config(
    page_title="Smart Geo Data Explorer", 
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        background: #f8f9fa;
    }
    .insight-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .filter-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'filtered_gdf' not in st.session_state:
    st.session_state.filtered_gdf = None
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'active_dataset' not in st.session_state:
    st.session_state.active_dataset = None
if 'spatial_operation_result' not in st.session_state:
    st.session_state.spatial_operation_result = None

# Sidebar
with st.sidebar:
    st.markdown("### Smart Geo Data Explorer")
    st.markdown("---")
    
    # Dataset Management
    st.markdown("#### Dataset Management")
    
    if st.session_state.datasets:
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox(
            "Active Dataset",
            dataset_names,
            index=dataset_names.index(st.session_state.active_dataset) if st.session_state.active_dataset in dataset_names else 0
        )
        st.session_state.active_dataset = selected_dataset
        st.session_state.gdf = st.session_state.datasets[selected_dataset]
        
        st.caption(f"{len(st.session_state.datasets)} dataset(s) loaded")
        
        if st.button("Clear All Datasets"):
            st.session_state.datasets = {}
            st.session_state.active_dataset = None
            st.session_state.gdf = None
            st.session_state.filtered_gdf = None
            st.rerun()
    
    st.markdown("---")
    
    # Quick Stats
    if st.session_state.gdf is not None:
        st.markdown("#### Quick Stats")
        st.metric("Features", f"{len(st.session_state.gdf):,}")
        st.metric("Columns", len(st.session_state.gdf.columns))
        
        geom_type = st.session_state.gdf.geometry.geom_type.mode()[0] if len(st.session_state.gdf) > 0 else "Unknown"
        st.metric("Geometry", geom_type)
    
    st.markdown("---")
    


# Main header
st.markdown('<h1 class="main-header">Smart Geo Data Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze and visualize your spatial data in seconds</p>', unsafe_allow_html=True)

# File uploader in main area
st.markdown("### 📤 Upload Your Data")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["zip", "geojson", "csv"],
        help="Supported formats: Shapefile (ZIP), GeoJSON, CSV with coordinates"
    )

with col2:
    if uploaded_file is not None:
        dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.split('.')[0])

if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    with st.spinner('Processing your data...'):
        try:
            loaded_gdf = load_spatial_data(uploaded_file)
            
            # Add to datasets collection
            if dataset_name:
                st.session_state.datasets[dataset_name] = loaded_gdf
                st.session_state.active_dataset = dataset_name
                st.session_state.gdf = loaded_gdf
                st.session_state.filtered_gdf = None
                st.success(f"✅ {uploaded_file.name} loaded as '{dataset_name}'!")
            else:
                st.error("Please provide a dataset name")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# Only show tabs if data is loaded
if st.session_state.gdf is not None:
    gdf = st.session_state.gdf
    
    # Apply filters if they exist
    if st.session_state.filtered_gdf is not None:
        gdf = st.session_state.filtered_gdf
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", 
        "Data Preview", 
        "Geometry Analysis",
        "Map View",
        "Advanced Analysis",
        "Export & Reports"
    ])
    
    with tab1:
        st.markdown("### Dataset Overview")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick stats in colored cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Features",
                value=f"{len(gdf):,}",
                delta="Records",
                delta_color="off"
            )
        
        with col2:
            st.metric(
                label="Columns",
                value=len(gdf.columns),
                delta="Attributes",
                delta_color="off"
            )
        
        with col3:
            geom_types = gdf.geometry.geom_type.unique()
            st.metric(
                label="Geometry Types",
                value=len(geom_types),
                delta=", ".join(geom_types[:2]),
                delta_color="off"
            )
        
        with col4:
            null_count = gdf.isnull().sum().sum()
            st.metric(
                label="Missing Values",
                value=null_count,
                delta="⚠️" if null_count > 0 else "✓",
                delta_color="inverse" if null_count > 0 else "off"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Column information
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### Column Information")
            col_info = pd.DataFrame({
                'Column': gdf.columns,
                'Type': gdf.dtypes.astype(str),
                'Null': [gdf[col].isnull().sum() for col in gdf.columns]
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        with col_right:
            st.markdown("#### Data Quality")
            completeness = ((len(gdf) * len(gdf.columns) - gdf.isnull().sum().sum()) / 
                          (len(gdf) * len(gdf.columns)) * 100)
            
            st.progress(completeness / 100)
            st.markdown(f"**Completeness:** {completeness:.1f}%")
            
            if null_count > 0:
                st.warning(f"⚠️ Found {null_count} missing values across the dataset")
            else:
                st.success("✓ No missing values detected!")
    
    with tab2:
        st.markdown("### Data Preview & Filtering")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Advanced Filtering Section
        with st.expander("🔍 Advanced Filters", expanded=False):
            st.markdown('<div class="filter-section">', unsafe_allow_html=True)
            
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                # Attribute filters
                st.markdown("##### Attribute Filters")
                
                # Get all columns except geometry
                filterable_cols = [col for col in gdf.columns if col != 'geometry']
                
                if filterable_cols:
                    selected_filter_col = st.selectbox("Select column to filter", ["None"] + filterable_cols)
                    
                    if selected_filter_col != "None":
                        col_dtype = gdf[selected_filter_col].dtype
                        
                        # Numeric filtering
                        if pd.api.types.is_numeric_dtype(col_dtype):
                            min_val = float(gdf[selected_filter_col].min())
                            max_val = float(gdf[selected_filter_col].max())
                            
                            filter_range = st.slider(
                                f"Filter {selected_filter_col}",
                                min_val, max_val, (min_val, max_val)
                            )
                            
                            if st.button("Apply Numeric Filter"):
                                mask = (gdf[selected_filter_col] >= filter_range[0]) & (gdf[selected_filter_col] <= filter_range[1])
                                st.session_state.filtered_gdf = gdf[mask]
                                st.success(f"Filtered to {len(st.session_state.filtered_gdf)} features")
                                st.rerun()
                        
                        # Categorical filtering
                        else:
                            unique_vals = gdf[selected_filter_col].unique()
                            selected_vals = st.multiselect(
                                f"Select {selected_filter_col} values",
                                unique_vals
                            )
                            
                            if st.button("Apply Categorical Filter") and selected_vals:
                                mask = gdf[selected_filter_col].isin(selected_vals)
                                st.session_state.filtered_gdf = gdf[mask]
                                st.success(f"Filtered to {len(st.session_state.filtered_gdf)} features")
                                st.rerun()
            
            with filter_col2:
                # Spatial filters
                st.markdown("##### Spatial Filters")
                
                spatial_filter_type = st.selectbox(
                    "Spatial Filter Type",
                    ["None", "Bounding Box", "Area Range"]
                )
                
                if spatial_filter_type == "Bounding Box":
                    st.caption("Filter features within a bounding box")
                    
                    gdf_wgs84 = gdf.to_crs(epsg=4326)
                    bounds = gdf_wgs84.total_bounds
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        min_lon = st.number_input("Min Longitude", value=float(bounds[0]), format="%.6f")
                        min_lat = st.number_input("Min Latitude", value=float(bounds[1]), format="%.6f")
                    with col_b:
                        max_lon = st.number_input("Max Longitude", value=float(bounds[2]), format="%.6f")
                        max_lat = st.number_input("Max Latitude", value=float(bounds[3]), format="%.6f")
                    
                    if st.button("Apply Bounding Box Filter"):
                        from shapely.geometry import box
                        bbox = box(min_lon, min_lat, max_lon, max_lat)
                        mask = gdf_wgs84.geometry.intersects(bbox)
                        st.session_state.filtered_gdf = gdf[mask]
                        st.success(f"Filtered to {len(st.session_state.filtered_gdf)} features")
                        st.rerun()
                
                elif spatial_filter_type == "Area Range":
                    if gdf.geometry.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
                        gdf_metric = gdf.to_crs(epsg=3857)
                        areas = gdf_metric.geometry.area / 1e6  # km²
                        
                        min_area, max_area = st.slider(
                            "Area Range (km²)",
                            float(areas.min()), float(areas.max()),
                            (float(areas.min()), float(areas.max()))
                        )
                        
                        if st.button("Apply Area Filter"):
                            mask = (areas >= min_area) & (areas <= max_area)
                            st.session_state.filtered_gdf = gdf[mask]
                            st.success(f"Filtered to {len(st.session_state.filtered_gdf)} features")
                            st.rerun()
                    else:
                        st.info("Area filtering only available for polygon geometries")
            
            # Reset filters
            if st.session_state.filtered_gdf is not None:
                if st.button("Reset All Filters"):
                    st.session_state.filtered_gdf = None
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Search and display options
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search in data", placeholder="Enter search term...")
        with col2:
            num_rows = st.selectbox("Rows to display", [10, 25, 50, 100, "All"])
        
        # Show filter status
        if st.session_state.filtered_gdf is not None:
            st.info(f"🔍 Showing filtered data: {len(gdf)} of {len(st.session_state.gdf)} features")
        
        # Display data
        display_df = gdf if num_rows == "All" else gdf.head(num_rows)
        
        if search_term:
            mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            display_df = display_df[mask]
            st.info(f"Found {len(display_df)} matching records")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
    
    with tab3:
        st.markdown("### Geometry Analysis")
        st.markdown("<br>", unsafe_allow_html=True)
        
        info = analyze_geometry(gdf)
        
        # Key metrics in prominent cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Features")
            st.markdown(f"<h2 style='color: #667eea; margin: auto;'>{info['feature_count']:,}</h2>", unsafe_allow_html=True)
            st.markdown("Total features in dataset")
        
        with col2:
            st.markdown("#### Geometry Type")
            st.markdown(f"<h2 style='color: #764ba2;'>{', '.join(info['geometry_type'])}</h2>", unsafe_allow_html=True)
            st.markdown("Primary geometry")
        
        with col3:
            st.markdown("#### Total Area")
            area_val = f"{info['total_area_km2']:,.2f}" if info['total_area_km2'] else "N/A"
            st.markdown(f"<h2 style='color: #667eea;'>{area_val}</h2>", unsafe_allow_html=True)
            st.markdown("Square kilometers")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Detailed information
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### Coordinate Reference System")
            st.code(str(info['crs']), language='text')
            
            # Geometry validation
            st.markdown("#### ✅ Geometry Validation")
            invalid_geoms = sum(~gdf.geometry.is_valid)
            if invalid_geoms > 0:
                st.warning(f"⚠️ {invalid_geoms} invalid geometries detected")
                if st.button("Fix Invalid Geometries"):
                    gdf.geometry = gdf.geometry.buffer(0)
                    st.success("Geometries fixed!")
            else:
                st.success("✓ All geometries are valid")
        
        with col_right:
            st.markdown("#### Bounding Box")
            bbox_df = pd.DataFrame({
                "Coordinate": ["Min Longitude", "Min Latitude", "Max Longitude", "Max Latitude"],
                "Value": [f"{x:.6f}" for x in info['bounds']]
            })
            st.dataframe(bbox_df, use_container_width=True, hide_index=True)
    
    with tab4:
        st.markdown("### Interactive Map View")
        st.markdown("<br>", unsafe_allow_html=True)

        try:
            if 'geometry' in gdf.columns and not gdf.empty:
                
                # Map controls
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    map_type = st.selectbox(
                        "Map Type",
                        ["Standard", "Choropleth", "Heatmap", "Cluster"]
                    )
                
                with col2:
                    basemap = st.selectbox(
                        "Basemap",
                        ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark Matter"]
                    )
                    
                    basemap_dict = {
                        "OpenStreetMap": "OpenStreetMap",
                        "CartoDB Positron": "CartoDB positron",
                        "CartoDB Dark Matter": "CartoDB dark_matter"
                    }
                    tiles = basemap_dict[basemap]
                
                with col3:
                    # Color by attribute for choropleth
                    if map_type == "Choropleth":
                        numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            color_col = st.selectbox("Color by", numeric_cols)
                        else:
                            st.warning("No numeric columns for choropleth")
                            color_col = None
                
                # Reproject to WGS84
                gdf_wgs84 = gdf.to_crs(epsg=4326)
                
                # Center map
                centroid = gdf_wgs84.geometry.centroid
                map_center = [centroid.y.mean(), centroid.x.mean()]
                
                # Calculate appropriate zoom level
                bounds = gdf_wgs84.total_bounds
                lat_diff = bounds[3] - bounds[1]
                lon_diff = bounds[2] - bounds[0]
                max_diff = max(lat_diff, lon_diff)
                
                if max_diff > 10:
                    zoom = 5
                elif max_diff > 5:
                    zoom = 6
                elif max_diff > 1:
                    zoom = 8
                elif max_diff > 0.5:
                    zoom = 10
                else:
                    zoom = 12
                
                m = folium.Map(location=map_center, zoom_start=zoom, tiles=tiles)
                
                if map_type == "Standard":
                    # Standard GeoJSON layer
                    folium.GeoJson(
                        gdf_wgs84,
                        name="Data Layer",
                        tooltip=folium.GeoJsonTooltip(
                            fields=gdf_wgs84.columns[:5].tolist(),
                            aliases=gdf_wgs84.columns[:5].tolist()
                        ),
                        style_function=lambda x: {
                            'fillColor': '#667eea',
                            'color': '#764ba2',
                            'weight': 2,
                            'fillOpacity': 0.5
                        }
                    ).add_to(m)
                
                elif map_type == "Choropleth" and color_col:
                    # Choropleth map
                    colormap = LinearColormap(
                        colors=['yellow', 'orange', 'red'],
                        vmin=gdf_wgs84[color_col].min(),
                        vmax=gdf_wgs84[color_col].max(),
                        caption=color_col
                    )
                    
                    folium.GeoJson(
                        gdf_wgs84,
                        name="Choropleth Layer",
                        style_function=lambda feature: {
                            'fillColor': colormap(feature['properties'][color_col]) if feature['properties'][color_col] else '#gray',
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.7
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=[color_col] + gdf_wgs84.columns[:4].tolist(),
                            aliases=[color_col] + gdf_wgs84.columns[:4].tolist()
                        )
                    ).add_to(m)
                    
                    colormap.add_to(m)
                
                elif map_type == "Heatmap":
                    # Heatmap
                    gdf_wgs84['centroid'] = gdf_wgs84.geometry.centroid
                    heat_data = [[point.y, point.x] for point in gdf_wgs84['centroid']]
                    
                    HeatMap(
                        data=heat_data,
                        radius=15,
                        blur=25,
                        gradient={
                            0.0: '#440154',
                            0.25: '#31688e',
                            0.5: '#35b779',
                            0.75: '#fde725',
                            1.0: '#fde725'
                        }
                    ).add_to(m)
                
                elif map_type == "Cluster":
                    # Marker cluster
                    from folium.plugins import MarkerCluster
                    
                    gdf_wgs84['centroid'] = gdf_wgs84.geometry.centroid
                    marker_cluster = MarkerCluster().add_to(m)
                    
                    for idx, row in gdf_wgs84.iterrows():
                        folium.Marker(
                            location=[row['centroid'].y, row['centroid'].x],
                            popup=f"Feature {idx}"
                        ).add_to(marker_cluster)
                
                # Add layer control
                folium.LayerControl().add_to(m)
                
                # Display map
                st_folium(m, width=None, height=600)
                
                # Map statistics
                st.markdown("#### Map Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Visible Features", len(gdf_wgs84))
                
                with col2:
                    st.metric("Map Center", f"{map_center[0]:.4f}, {map_center[1]:.4f}")
                
                with col3:
                    st.metric("Lat Span", f"{lat_diff:.4f}°")
                
                with col4:
                    st.metric("Lon Span", f"{lon_diff:.4f}°")

            else:
                st.info("Map view requires a valid geometry column and non-empty data.")

        except Exception as e:
            st.warning(f"Could not generate map view: {str(e)}")
            st.info("Ensure your data has valid geometries and CRS.")
    
    with tab5:
        st.markdown("### Advanced Analysis")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Analysis options
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Numeric Attributes", "Categorical Attributes", "Spatial Distribution", "Geometry Properties", "Correlation Analysis"]
        )
        
        st.markdown("---")
        
        if analysis_type == "Numeric Attributes":
            st.markdown("#### Numeric Attribute Analysis")
            
            numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select attribute to analyze", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Statistical Summary")
                    stats_df = pd.DataFrame({
                        'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
                        'Value': [
                            f"{gdf[selected_col].count():,.0f}",
                            f"{gdf[selected_col].mean():.2f}",
                            f"{gdf[selected_col].median():.2f}",
                            f"{gdf[selected_col].std():.2f}",
                            f"{gdf[selected_col].min():.2f}",
                            f"{gdf[selected_col].max():.2f}",
                            f"{gdf[selected_col].quantile(0.25):.2f}",
                            f"{gdf[selected_col].quantile(0.75):.2f}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("##### Distribution")
                    fig = px.histogram(
                        gdf, 
                        x=selected_col,
                        nbins=30,
                        title=f"Distribution of {selected_col}",
                        color_discrete_sequence=['#667eea']
                    )
                    fig.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Box plot
                st.markdown("##### Box Plot Analysis")
                fig_box = px.box(
                    gdf, 
                    y=selected_col,
                    title=f"Box Plot: {selected_col}",
                    color_discrete_sequence=['#764ba2']
                )
                fig_box.update_layout(height=300)
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Insights
                #st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("**Key Insights:**")
                mean_val = gdf[selected_col].mean()
                median_val = gdf[selected_col].median()
                skewness = "right-skewed" if mean_val > median_val else "left-skewed" if mean_val < median_val else "symmetric"
                st.write(f"- The distribution appears to be **{skewness}**")
                st.write(f"- Range: {gdf[selected_col].min():.2f} to {gdf[selected_col].max():.2f}")
                outlier_threshold = gdf[selected_col].quantile(0.75) + 1.5 * (gdf[selected_col].quantile(0.75) - gdf[selected_col].quantile(0.25))
                outliers = len(gdf[gdf[selected_col] > outlier_threshold])
                if outliers > 0:
                    st.write(f"- ⚠️ Detected approximately **{outliers}** potential outliers")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No numeric attributes found in the dataset for statistical analysis.")
        
        elif analysis_type == "Categorical Attributes":
            st.markdown("#### Categorical Attribute Analysis")
            
            categorical_cols = gdf.select_dtypes(include=['object', 'category']).columns.tolist()
            categorical_cols = [col for col in categorical_cols if col != 'geometry']
            
            if len(categorical_cols) > 0:
                selected_col = st.selectbox("Select categorical attribute to analyze", categorical_cols)
                
                value_counts = gdf[selected_col].value_counts()
                total_count = len(gdf[selected_col].dropna())
                unique_count = gdf[selected_col].nunique()
                missing_count = gdf[selected_col].isnull().sum()
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Unique Values", f"{unique_count:,}")
                
                with col2:
                    st.metric("Most Common", value_counts.index[0] if len(value_counts) > 0 else "N/A")
                
                with col3:
                    st.metric("Mode Frequency", f"{value_counts.iloc[0]:,}" if len(value_counts) > 0 else "0")
                
                with col4:
                    st.metric("Missing Values", f"{missing_count:,}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Visualization section
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("##### Top 10 Categories")
                    
                    top_10 = value_counts.head(10)
                    freq_df = pd.DataFrame({
                        'Category': top_10.index,
                        'Count': top_10.values,
                        'Percentage': (top_10.values / total_count * 100).round(2)
                    })
                    freq_df['Percentage'] = freq_df['Percentage'].astype(str) + '%'
                    st.dataframe(freq_df, use_container_width=True, hide_index=True)
                
                with col_right:
                    st.markdown("##### Distribution")
                    if value_counts.empty:
                        st.info(f"No data available for column `{selected_col}`.")
                    else:
                        top_n = min(10, unique_count)
                        fig_bar = px.bar(
                            x=value_counts.head(top_n).index,
                            y=value_counts.head(top_n).values,
                            labels={'x': selected_col, 'y': 'Count'},
                            title=f"Top {top_n} Categories",
                            color=value_counts.head(top_n).values,
                            color_continuous_scale='Viridis'
                        )
                        fig_bar.update_layout(showlegend=False, height=350)
                        st.plotly_chart(fig_bar, use_container_width=True)
                
                # Pie chart
                st.markdown("##### Proportion Analysis")
                
                if unique_count > 10:
                    top_categories = value_counts.head(9)
                    others_count = value_counts.iloc[9:].sum()
                    
                    pie_data = pd.DataFrame({
                        'Category': list(top_categories.index) + ['Others'],
                        'Count': list(top_categories.values) + [others_count]
                    })
                else:
                    pie_data = pd.DataFrame({
                        'Category': value_counts.index,
                        'Count': value_counts.values
                    })
                
                fig_pie = px.pie(
                    pie_data,
                    values='Count',
                    names='Category',
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Diversity metrics
                st.markdown("##### Diversity Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    proportions = value_counts / total_count
                    shannon_index = -sum(proportions * np.log(proportions))
                    st.metric("Shannon Diversity", f"{shannon_index:.3f}")
                    st.caption("Higher values indicate more diversity")
                
                with col2:
                    simpson_index = 1 - sum(proportions ** 2)
                    st.metric("Simpson's Diversity", f"{simpson_index:.3f}")
                    st.caption("0 = no diversity, 1 = infinite diversity")
                
                with col3:
                    max_diversity = np.log(unique_count) if unique_count > 0 else 1
                    evenness = shannon_index / max_diversity if max_diversity > 0 else 0
                    st.metric("Evenness", f"{evenness:.3f}")
                    st.caption("How evenly distributed categories are")
                
                # Cross-analysis
                numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 0:
                    st.markdown("##### Cross-Analysis with Numeric Attributes")
                    
                    numeric_col = st.selectbox(
                        "Select numeric attribute for comparison",
                        numeric_cols,
                        key='cat_numeric_cross'
                    )
                    
                    grouped_stats = gdf.groupby(selected_col)[numeric_col].agg(['count', 'mean', 'median', 'std']).round(2)
                    grouped_stats = grouped_stats.sort_values('mean', ascending=False).head(10)
                    grouped_stats.columns = ['Count', 'Mean', 'Median', 'Std Dev']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(grouped_stats, use_container_width=True)
                    
                    with col2:
                        top_categories = value_counts.head(10).index
                        filtered_gdf = gdf[gdf[selected_col].isin(top_categories)]
                        
                        fig_box_cat = px.box(
                            filtered_gdf,
                            x=selected_col,
                            y=numeric_col,
                            title=f"{numeric_col} by {selected_col}",
                            color=selected_col,
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig_box_cat.update_layout(showlegend=False, height=400)
                        fig_box_cat.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_box_cat, use_container_width=True)
                
            else:
                st.info("No categorical attributes found in the dataset for analysis.")
        
        elif analysis_type == "Spatial Distribution":
            st.markdown("#### Spatial Distribution Analysis")
            
            try:
                gdf_wgs84 = gdf.to_crs(epsg=4326)
                
                gdf_wgs84['centroid'] = gdf_wgs84.geometry.centroid
                gdf_wgs84['lon'] = gdf_wgs84.centroid.x
                gdf_wgs84['lat'] = gdf_wgs84.centroid.y
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Spatial Extent")
                    extent_stats = pd.DataFrame({
                        'Direction': ['North', 'South', 'East', 'West'],
                        'Coordinate': [
                            f"{gdf_wgs84['lat'].max():.6f}°",
                            f"{gdf_wgs84['lat'].min():.6f}°",
                            f"{gdf_wgs84['lon'].max():.6f}°",
                            f"{gdf_wgs84['lon'].min():.6f}°"
                        ],
                        'Span': [
                            f"{gdf_wgs84['lat'].max() - gdf_wgs84['lat'].min():.6f}°",
                            "",
                            f"{gdf_wgs84['lon'].max() - gdf_wgs84['lon'].min():.6f}°",
                            ""
                        ]
                    })
                    st.dataframe(extent_stats, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("##### Centroid Statistics")
                    centroid_stats = pd.DataFrame({
                        'Metric': ['Mean Latitude', 'Mean Longitude', 'Median Latitude', 'Median Longitude'],
                        'Value': [
                            f"{gdf_wgs84['lat'].mean():.6f}°",
                            f"{gdf_wgs84['lon'].mean():.6f}°",
                            f"{gdf_wgs84['lat'].median():.6f}°",
                            f"{gdf_wgs84['lon'].median():.6f}°"
                        ]
                    })
                    st.dataframe(centroid_stats, use_container_width=True, hide_index=True)
                
                # Density heatmap
                st.markdown("##### Feature Density Heatmap")
                if 'lat' in gdf_wgs84.columns and 'lon' in gdf_wgs84.columns:
                    centroid = gdf_wgs84.geometry.centroid
                    map_center = [centroid.y.mean(), centroid.x.mean()]
                    m = folium.Map(location=map_center, zoom_start=8, tiles="OpenStreetMap")

                    heat_data = gdf_wgs84[['lat', 'lon']].values.tolist()

                    HeatMap(
                        data=heat_data,
                        radius=15,
                        blur=10,
                        max_zoom=12,
                        gradient={
                            0.0: '#440154',
                            0.25: '#31688e',
                            0.5: '#35b779',
                            0.75: '#fde725',
                            1.0: '#fde725'
                        }
                    ).add_to(m)

                    viridis_colormap = LinearColormap(
                        colors=['#440154', '#31688e', '#35b779', '#fde725'],
                        index=[0, 0.25, 0.5, 1],
                        vmin=0,
                        vmax=1,
                        caption='Density'
                    )
                    viridis_colormap.add_to(m)

                    st_folium(m, width=None, height=500)
                
            except Exception as e:
                st.error(f"Could not perform spatial distribution analysis: {str(e)}")
        
        elif analysis_type == "Geometry Properties":
            st.markdown("#### Geometry Properties Analysis")
            
            try:
                gdf_metric = gdf.to_crs(epsg=3857)
                
                if gdf.geometry.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
                    gdf_metric['area_km2'] = gdf_metric.geometry.area / 1e6
                    gdf_metric['perimeter_km'] = gdf_metric.geometry.length / 1000
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Area", f"{gdf_metric['area_km2'].sum():.2f} km²")
                        st.metric("Mean Area", f"{gdf_metric['area_km2'].mean():.2f} km²")
                    
                    with col2:
                        st.metric("Largest Feature", f"{gdf_metric['area_km2'].max():.2f} km²")
                        st.metric("Smallest Feature", f"{gdf_metric['area_km2'].min():.2f} km²")
                    
                    with col3:
                        st.metric("Total Perimeter", f"{gdf_metric['perimeter_km'].sum():.2f} km")
                        st.metric("Mean Perimeter", f"{gdf_metric['perimeter_km'].mean():.2f} km")
                    
                    st.markdown("##### Area Distribution")
                    fig = px.histogram(
                        gdf_metric,
                        x='area_km2',
                        nbins=30,
                        title="Distribution of Feature Areas",
                        labels={'area_km2': 'Area (km²)'},
                        color_discrete_sequence=['#667eea']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("##### Top 10 Largest Features")
                    top_10 = gdf_metric.nlargest(10, 'area_km2')[['area_km2', 'perimeter_km']].reset_index(drop=True)
                    top_10.index = top_10.index + 1
                    top_10.columns = ['Area (km²)', 'Perimeter (km)']
                    st.dataframe(top_10, use_container_width=True)
                    
                elif gdf.geometry.geom_type.iloc[0] in ['LineString', 'MultiLineString']:
                    gdf_metric['length_km'] = gdf_metric.geometry.length / 1000
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Length", f"{gdf_metric['length_km'].sum():.2f} km")
                    
                    with col2:
                        st.metric("Mean Length", f"{gdf_metric['length_km'].mean():.2f} km")
                    
                    with col3:
                        st.metric("Longest Feature", f"{gdf_metric['length_km'].max():.2f} km")
                    
                    st.markdown("##### Length Distribution")
                    fig = px.histogram(
                        gdf_metric,
                        x='length_km',
                        nbins=30,
                        title="Distribution of Feature Lengths",
                        labels={'length_km': 'Length (km)'},
                        color_discrete_sequence=['#764ba2']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("Point geometries - showing spatial clustering analysis")
                    
                    gdf_wgs84 = gdf.to_crs(epsg=4326)
                    gdf_wgs84['lon'] = gdf_wgs84.geometry.x
                    gdf_wgs84['lat'] = gdf_wgs84.geometry.y
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Points", len(gdf))
                        st.metric("Latitude Range", f"{gdf_wgs84['lat'].max() - gdf_wgs84['lat'].min():.4f}°")
                    
                    with col2:
                        st.metric("Mean Latitude", f"{gdf_wgs84['lat'].mean():.6f}°")
                        st.metric("Longitude Range", f"{gdf_wgs84['lon'].max() - gdf_wgs84['lon'].min():.4f}°")
                    
            except Exception as e:
                st.error(f"Could not calculate geometry properties: {str(e)}")
        
        elif analysis_type == "Correlation Analysis":
            st.markdown("#### Correlation Analysis")
            
            numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                st.markdown("##### Correlation Matrix")
                corr_matrix = gdf[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlation Heatmap",
                    zmin=-1, zmax=1
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("##### Scatter Plot Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_var = st.selectbox("X-axis variable", numeric_cols, key='x_var')
                
                with col2:
                    y_var = st.selectbox("Y-axis variable", [col for col in numeric_cols if col != x_var], key='y_var')
                
                fig_scatter = px.scatter(
                    gdf,
                    x=x_var,
                    y=y_var,
                    trendline="ols",
                    title=f"{x_var} vs {y_var}",
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.markdown("##### Strong Correlations (|r| > 0.7)")
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            strong_corr.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': f"{corr_matrix.iloc[i, j]:.3f}"
                            })
                
                if strong_corr:
                    st.dataframe(pd.DataFrame(strong_corr), use_container_width=True, hide_index=True)
                else:
                    st.info("No strong correlations (|r| > 0.7) found between variables.")
                
            else:
                st.info("Need at least 2 numeric attributes for correlation analysis.")
    
    
    
    with tab6:
        st.markdown("### Export & Reports")
        st.markdown("<br>", unsafe_allow_html=True)
        
        report_type = st.selectbox(
            "Select Export Type",
            ["Data Export", "Analysis Report", "Map Export"]
        )
        
        st.markdown("---")
        
        if report_type == "Data Export":
            st.markdown("#### Data Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### CSV Export")
                csv_data = gdf.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"{st.session_state.active_dataset}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("##### GeoJSON Export")
                geojson_data = gdf.to_json()
                st.download_button(
                    label="📥 Download GeoJSON",
                    data=geojson_data,
                    file_name=f"{st.session_state.active_dataset}_{datetime.now().strftime('%Y%m%d')}.geojson",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                st.markdown("##### Excel Export")
                # Convert to Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Drop geometry for Excel
                    df_export = pd.DataFrame(gdf.drop(columns=['geometry']))
                    df_export.to_excel(writer, sheet_name='Data', index=False)
                
                excel_data = output.getvalue()
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"{st.session_state.active_dataset}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        elif report_type == "Analysis Report":
            st.markdown("#### Generate Analysis Report")
            
            include_overview = st.checkbox("Include Dataset Overview", value=True)
            include_stats = st.checkbox("Include Statistical Analysis", value=True)
            include_geometry = st.checkbox("Include Geometry Analysis", value=True)
            
            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
                    try:
                        report_content = f"""
# Geo Data Analysis Report
**Dataset:** {st.session_state.active_dataset}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""
                        if include_overview:
                            report_content += f"""
## Dataset Overview

- **Total Features:** {len(gdf):,}
- **Total Columns:** {len(gdf.columns)}
- **Geometry Types:** {', '.join(gdf.geometry.geom_type.unique())}
- **Missing Values:** {gdf.isnull().sum().sum()}

### Columns
"""
                            for col in gdf.columns:
                                report_content += f"- {col} ({gdf[col].dtype})\n"
                        
                        if include_stats:
                            numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
                            if numeric_cols:
                                report_content += f"""

## Statistical Analysis

"""
                                for col in numeric_cols[:5]:  # Limit to 5 columns
                                    report_content += f"""
### {col}
- Mean: {gdf[col].mean():.2f}
- Median: {gdf[col].median():.2f}
- Std Dev: {gdf[col].std():.2f}
- Min: {gdf[col].min():.2f}
- Max: {gdf[col].max():.2f}

"""
                        
                        if include_geometry:
                            info = analyze_geometry(gdf)
                            
                            # Handle None value for total_area_km2
                            area_text = f"{info['total_area_km2']:.2f} km²" if info['total_area_km2'] is not None else "N/A (not applicable for this geometry type)"
                            
                            report_content += f"""
## Geometry Analysis

- **Feature Count:** {info['feature_count']:,}
- **Geometry Type:** {', '.join(info['geometry_type'])}
- **Total Area:** {area_text}
- **CRS:** {info['crs']}
- **Bounding Box:** 
  - Min Lon: {info['bounds'][0]:.6f}
  - Min Lat: {info['bounds'][1]:.6f}
  - Max Lon: {info['bounds'][2]:.6f}
  - Max Lat: {info['bounds'][3]:.6f}

"""
                        
                        st.markdown(report_content)
                        
                        # Download button
                        st.download_button(
                            label="Download Report (Markdown)",
                            data=report_content,
                            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
        
        elif report_type == "Map Export":
            st.markdown("#### Map Export")
            st.info("Export interactive map as HTML")
            
            map_title = st.text_input("Map Title", value=f"{st.session_state.active_dataset} Map")
            
            if st.button("Generate Map"):
                with st.spinner("Generating map..."):
                    try:
                        gdf_wgs84 = gdf.to_crs(epsg=4326)
                        
                        centroid = gdf_wgs84.geometry.centroid
                        map_center = [centroid.y.mean(), centroid.x.mean()]
                        
                        m = folium.Map(location=map_center, zoom_start=10, tiles="OpenStreetMap")
                        
                        folium.GeoJson(
                            gdf_wgs84,
                            name=map_title,
                            tooltip=folium.GeoJsonTooltip(
                                fields=gdf_wgs84.columns[:5].tolist(),
                                aliases=gdf_wgs84.columns[:5].tolist()
                            )
                        ).add_to(m)
                        
                        folium.LayerControl().add_to(m)
                        
                        # Save to HTML
                        html_data = m._repr_html_()
                        
                        st.success("✅ Map generated!")
                        
                        st.download_button(
                            label="Download Map (HTML)",
                            data=html_data,
                            file_name=f"map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating map: {str(e)}")

else:
    # Empty state
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px;'>
            <h2 style='color: #667eea;'>Get Started</h2>
            <p style='font-size: 1.1rem; color: #6c757d;'>
                Upload a spatial dataset above to begin your analysis<br>
                Supported Formats: Shapefiles, GeoJSON, and CSV files
            </p>
        </div>
        """, unsafe_allow_html=True)
