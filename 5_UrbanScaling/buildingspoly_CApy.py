import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import os
import tqdm # Import the main library

# --- Configuration ---

dir_path = "/Users/9oy/Documents/Data/IM3/Buildings/MAv1"
output_gpkg_path = "/Users/9oy/Documents/Data/IM3/buildingspoly_CApy.gpkg"

try:
    os.chdir(dir_path)
except FileNotFoundError:
    print(f"Error: Directory not found at {dir_path}. Please check the path.")
    exit()

# --- File Handling and Data Loading ---

files = [f for f in os.listdir('.') if f.endswith('.csv')]
files.sort()

if len(files) < 5:
    raise FileNotFoundError(f"Only found {len(files)} CSV files. Need at least 5.")

fname = files[4]
stname = fname.replace(".csv", "")
file_to_read = fname
print(stname)

# Use pandas' fast CSV reader
df = pd.read_csv(file_to_read)

# Drop the first column (equivalent to R's drop=1)
if len(df.columns) > 1:
    df = df.iloc[:, 1:].copy()
    print(df.shape)
else:
    print("Warning: DataFrame has only one column. Did not perform drop=1.")


# --- Centroid Processing (Vectorized) ---

# Split 'Centroid' column and convert to numeric
df[['lat_str', 'lon_str']] = df['Centroid'].str.split('/', expand=True)

df['lat'] = pd.to_numeric(df['lat_str'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon_str'], errors='coerce')

df.drop(columns=['Centroid', 'lat_str', 'lon_str'], inplace=True, errors='ignore')


# --- Polygon Processing (Optimized Function with Progress Bar) ---

# Register the progress_apply method with pandas (The correct equivalent of R's pblapply)
tqdm.tqdm.pandas()

def create_polygon_from_footprint(footprint_str):
    """
    Converts a coordinate string ('lat1/lon1_lat2/lon2_...') to a Shapely Polygon.
    Implements the R code logic: (lon, lat) order and closing the ring.
    """
    if pd.isna(footprint_str) or not isinstance(footprint_str, str):
        return None

    try:
        # Split coordinate pairs (e.g., "lat1/lon1")
        coords_str_list = footprint_str.split('_')
        coords_list = []
        
        for coord_str in coords_str_list:
            # Assuming format "lat/lon"
            lat_str, lon_str = coord_str.split('/')
            lat, lon = float(lat_str), float(lon_str)
            # st_polygon uses (lon, lat) order, hence the swap
            coords_list.append((lon, lat))

        if len(coords_list) < 3:
            return None

        # Close the polygon ring by adding the first coordinate at the end
        if coords_list[0] != coords_list[-1]:
            coords_list.append(coords_list[0])

        return Polygon(coords_list)

    except Exception:
        # Handle malformed data
        return None

print("Processing polygon geometries...")
# Use progress_apply for a progress bar
df['geometry'] = df['Footprint2D'].progress_apply(create_polygon_from_footprint)


# --- GeoDataFrame Creation and Export ---

# Filter out null geometries and convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df.dropna(subset=['geometry']), geometry='geometry', crs="EPSG:4326")

# Select the required columns and ensure 'lat'/'lon' are included
columns_to_keep = [
    "ID", "State_Abbr", "Area", "Area2D", "Height", "NumFloors",
    "WWR_surfaces", "BuildingType", "Standard", "lat", "lon", "geometry"
]
# Select only columns that exist in the DataFrame
final_gdf = gdf[[col for col in columns_to_keep if col in gdf.columns]]


# Write to GeoPackage (GPKG), using stname as the layer name
print(f"Writing final GeoDataFrame to {output_gpkg_path} with layer name: {stname}...")
final_gdf.to_file(output_gpkg_path, driver="GPKG", layer=stname)

print("Processing complete!")