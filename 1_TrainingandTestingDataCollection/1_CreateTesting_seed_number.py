import sys
import os
import geopandas as gpd
import rasterio
import numpy as np
from rasterio import features, windows
from shapely.geometry import box
from pyproj import Transformer
from shapely.ops import transform
import cv2
import glob
from pathlib import Path

## This script was run from the terminal using the command: 
## for i in {1000..1003}; do python CreateTesting_seed_number.py $i 500 & done

UtilityScriptFolder = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Script"
if UtilityScriptFolder not in sys.path:
    sys.path.insert(0, UtilityScriptFolder)
from BPspatlibv1 import *

numpyseed = int(sys.argv[1])  
numtiles = int(sys.argv[2])  

# File paths
# Directory where NEW VALIDATION TILES will be saved
trainbasedir = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TestNooverlap" 
TRAINING_DATA_FOLDER = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/Archive/central" 
pathtomsafile = "/Users/9oy/Documents/Projects/IM3/Data/MSA_notPR.gpkg"
nlcd = "/Users/9oy/Documents/Data/US/Environmental/NLCD/Annual/Annual_NLCD_LndCov_2015_CU_C1V0.tif" ## 2015 NLCD because of consisten
tmpfilepath = os.path.join("/Users/9oy/Documents/Projects/IM3/EvaluationP/junk","tmp")
buildingsfilepath = "/Users/9oy/Documents/Data/IM3/buildingspoly_CApy.gpkg" ## This file was generated using the buildingspoly_CApy.py script that polygonizes the csv in Model America version 1 dataset for California.
popfile = "/Users/9oy/Documents/Projects/IM3/Data/PopSSP5_2015_proj_30_LA_bilnear.tif"



np.random.seed(numpyseed)
os.chdir(UtilityScriptFolder)

# ==============================================================================
# SPATIAL OVERLAP UTILITIES
# ==============================================================================

def get_tiff_bounds(filepath: Path) -> tuple | None:
    """Reads a GeoTIFF and returns its bounding box (left, bottom, right, top)."""
    try:
        with rasterio.open(filepath) as src:
            return src.bounds
    except rasterio.RasterioIOError:
        print(f"Warning: Could not open GeoTIFF file for bounds check: {filepath.name}. Skipping.")
        return None
    except Exception as e:
        # Catching other potential issues like missing georeferencing
        print(f"Error reading bounds for {filepath.name}: {e}")
        return None

def check_overlap(bbox1: tuple, bbox2: tuple) -> bool:
    """Checks if two bounding boxes (l, b, r, t) overlap."""
    l1, b1, r1, t1 = bbox1
    l2, b2, r2, t2 = bbox2

    # Check for non-overlap along the X-axis (l1 >= r2 or l2 >= r1)
    if l1 >= r2 or l2 >= r1:
        return False
    # Check for non-overlap along the Y-axis (b1 >= t2 or b2 >= t1)
    if b1 >= t2 or b2 >= t1:
        return False

    # If neither non-overlap condition is met, they must overlap
    return True

def load_existing_training_bounds(training_folder: str) -> list[tuple]:
    """Loads bounding boxes of all existing GeoTIFFs in the training folder."""
    p_train = Path(training_folder)
    if not p_train.exists():
        print(f"Warning: Training data folder not found at {training_folder}. Proceeding without overlap check.")
        return []

    print(f"Loading bounds from existing training data in: {training_folder}")
    training_bounds = []
    
    # Assuming training tiles are .tif or .tiff files
    tiff_files = [f for f in p_train.rglob('*') if f.suffix.lower() in ['.tif', '.tiff']]
    
    for filepath in tiff_files:
        bounds = get_tiff_bounds(filepath)
        if bounds:
            training_bounds.append(bounds)
    
    print(f"Successfully loaded {len(training_bounds)} bounding box(es) from training data.")
    return training_bounds

# ==============================================================================
# INITIALIZATION
# ==============================================================================

if not os.path.isdir(trainbasedir):
    os.mkdir(trainbasedir)
    os.mkdir(os.path.join(trainbasedir,"central"))
    os.mkdir(os.path.join(trainbasedir,"surrounding"))
    os.mkdir(os.path.join(trainbasedir,"target"))
    os.mkdir(os.path.join(trainbasedir,"Frac"))

# Load existing training bounds once before the generation loop
TRAINING_BOUNDS = load_existing_training_bounds(TRAINING_DATA_FOLDER)
print("-" * 50)


# ==============================================================================
# GEOSPATIAL HELPER FUNCTIONS
# ==============================================================================

def calculate_intersection_area(geometry, pixel_bounds):
    return geometry.intersection(pixel_bounds).area

def get_pixel_bbox(row, col, transform):
    """Helper to get the bounding box of a single pixel"""
    # Uses rasterio's transform to calculate the corner coordinates
    left, top = transform * (col, row)
    right, bottom = transform * (col + 1, row + 1)
    return box(left, bottom, right, top)

# Generate NLCD Tiff for LA:
msa = gpd.read_file(pathtomsafile)[["GEOID","NAME","geometry"]]
LA = msa.loc[msa.NAME.str.contains("San Diego"),] ## Since titles selected here won't overlap with the training data
outnlcd = os.path.join(tmpfilepath,"nlcd2015.tif")

# NOTE: Since the clipraster function (BPspatlibv0) is not provided, 
# we rely on the existing file check logic.
if not os.path.isfile(outnlcd):
    # This call relies on BPspatlibv0.clipraster, which must be imported/defined.
    # Assuming 'from BPspatlibv0 import *' works and clipraster is available.
    print("NLCD clip file not found, attempting to clip...")
    clipraster(nlcd,LA,outnlcd,tmpfilepath,numpyseed,numtiles) 
else:
    print(f"NLCD clip file found at {outnlcd}.")


# Get Indices for Pixels Where we have LCLU Data
with rasterio.open(outnlcd) as src:
    ras = src.read(1)
    vrows, vcols = np.where((ras >= 21) & (ras <= 24))
    ras=None
        
# ==============================================================================
# MAIN TILE GENERATION LOOP
# ==============================================================================

count = 0
while count < numtiles:
    
    # 1. Select a random pixel location
    idx = np.random.choice(np.arange(len(vrows)), 1, replace=False)
    rowfull = vrows[idx][0]
    colfull = vcols[idx][0]
    
    # Define the central window (256x256)
    central_window = windows.Window(colfull, rowfull, 256, 256)

    try:
        with rasterio.open(outnlcd) as src:
            # Calculate the bounding box for the PROPOSED TILE
            proposed_transform = src.window_transform(central_window)
            
            # The bounds of the 256x256 central tile
            # (left, bottom, right, top) for the central tile
            l, b, r, t = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
            
            # Use the transform to find the actual bounds of the window
            bbox_proposed = src.window_bounds(central_window)
            
            
            # OVERLAP CHECK
            has_overlap = False
            if TRAINING_BOUNDS: # Only run if training data was loaded
                for bbox_train in TRAINING_BOUNDS:
                    if check_overlap(bbox_proposed, bbox_train):
                        has_overlap = True
                        break # Found an overlap, no need to check others
            
            if has_overlap:
                print(f"Skipping tile at row {rowfull}, col {colfull}: Overlaps with training data.")
                continue # Skip this iteration and try a new random index
           
            print("Iterating: " + str(count) + " (Non-overlapping)")

            # Check surrounding tiles for NoData (unchanged logic)
            xoff = [-64,0,64,-64,0,64,-64,0,64]
            yoff = [-64,-64,-64,0,0,0,64,64,64]
            raslist = list()
            rastranslist = list()
            nodatacheck = list()
            
            for j in range(9):
                ww = windows.Window(colfull+xoff[j], rowfull + yoff[j], 256, 256)
                ras=src.read(1,window=ww)
                raslist.append(ras)
                rastranslist.append(src.window_transform(ww))
                # Check for nodata (assuming 0 is the NoData/background value)
                nodatacheck.append((ras == 0).any()) 
            
            nodatacheck = np.array(nodatacheck[4]).any() ## We just rely on the central tile for the no data check.
            
            if not nodatacheck:
                
                # Write Central Tile
                kwargs = src.meta.copy()
                kwargs.update({
                    'height': 256,
                    'width': 256,
                    'transform': rastranslist[4]})
                
                out_path_central = os.path.join(trainbasedir, "central", f"LULC_{numpyseed}_{count}.tif")
                with rasterio.open(out_path_central, 'w', **kwargs) as dst:
                    dst.write(raslist[4],indexes=1)
                
                # Write Surrounding Tiles: This step is not used.
                outsurround = np.stack(raslist[0:4] + raslist[5:])
                kwargs.update({
                    'count': 8 # 8 surrounding tiles
                })
                
                out_path_surrounding = os.path.join(trainbasedir, "surrounding", f"LULC_{numpyseed}_{count}.tif")
                with rasterio.open(out_path_surrounding, 'w', **kwargs) as dst:
                    for band in range(1, 8 + 1):
                        dst.write(outsurround[band-1,:,:], band)

                # Generate Target, Population, and Fractional Coverage Data
                
                # Re-open central tile to ensure bounds are correct for subsequent steps
                with rasterio.open(out_path_central) as src_central:
                    #print(src_central.crs)
                    bounds = src_central.bounds
                    geom = box(*bounds)
                    # Transform bounds to EPSG:4326 for Buildings and Popfile checks
                    transformer = Transformer.from_crs(src_central.crs,"EPSG:4326", always_xy=True)
                    geomt = transform(transformer.transform, geom)
                    
                    # Read buildings data
                    polyb = gpd.read_file(buildingsfilepath, bbox = geomt,layer="CA",engine="pyogrio",use_arrow=True)
                    polyb = polyb.to_crs(src_central.crs) # Project buildings back to NLCD CRS
                    
                    # Target raster setup
                    raster_meta = src_central.meta.copy()
                    raster_shape = src_central.shape
                    raster_transform = src_central.transform
                    raster_meta.update({"dtype":rasterio.float32,"count":1})
                    
                    # Calculate Heights and Fractional Footprint per pixel:
                    weighted_sum = np.zeros(raster_shape)
                    weight_sum = np.zeros(raster_shape)
                    
                    for idx, row1 in polyb.iterrows():
                        geometry = row1.geometry
                        height = row1['Height']
                        
                        mask = features.rasterize(
                            [(geometry, 1)],
                            out_shape=raster_shape,
                            transform=raster_transform,
                            all_touched=True,
                            dtype=np.float32
                        )
                        
                        rows_idx, cols_idx = np.nonzero(mask)
                        
                        for r, c in zip(rows_idx, cols_idx):
                            pixel_bounds = get_pixel_bbox(r,c,raster_transform)
                            intersection_area = calculate_intersection_area(geometry, pixel_bounds)
                            
                            weighted_sum[r, c] += intersection_area * height
                            weight_sum[r, c] += intersection_area
                            
                    # Calculate mean building height (result) and fractional cover (result1)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        result = np.where(weight_sum > 0, weighted_sum / weight_sum, 0)
                        # Assuming NLCD pixel size is 30x30 meters (900 sq units)
                        result1 = weight_sum / (30*30) 
                        result1 = np.where(result1 > 1, 1, result1) # Cap fraction at 1.0

                    # Write Target (Mean Height)
                    outputtarget = os.path.join(trainbasedir, "target", f"LULC_{numpyseed}_{count}.tif")
                    with rasterio.open(outputtarget, 'w', **raster_meta) as dst:
                        dst.write(result.astype(np.float32), 1)
                        
                    # Write Fractional Cover
                    outputtarget0 = os.path.join(trainbasedir, "Frac", f"LULC_{numpyseed}_{count}.tif")
                    with rasterio.open(outputtarget0, 'w', **raster_meta) as dst:
                        dst.write(result1.astype(np.float32), 1)
                        
                count += 1 # Increment only if all files were successfully generated
            else:
                print(f"Skipping tile at row {rowfull}, col {colfull}: Contains NoData in surrounding area.")
                
    except Exception as e:
        print(f"Failed to process tile {count}. Error: {e}")
        print("Continuing...")
        continue

print(f"\nFinished generating {count} non-overlapping validation tiles for seed {numpyseed}.")
