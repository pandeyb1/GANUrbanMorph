import sys
import os
import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio import features
from rasterio.windows import Window
import numpy as np
from shapely.geometry import box
from pyproj import Transformer
from shapely.ops import transform
import cv2
import glob
from tqdm import tqdm
from scipy.ndimage import maximum_filter
import pandas as pd
from matplotlib import pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter, maximum_filter
import shapely
from osgeo import gdal
import shutil
import random


# 1. SETUP: Paths, Variables, and Seed
# Define core paths and parameters for the data generation process.
UtilityScriptFolder = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Script_Submission"
numpyseed = 100
trainbasedir = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Script_Submission/TestFolder/TrainingData"
testbasedir = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Script_Submission/TestFolder/TestData"
pathtomsafile = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/Boundary/US_urb_area_2010.shp"
nlcd = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/Annual_NLCD_LndCov_2010_CU_C1V1.tif"
bf = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/Updated_IM3_images/BF_2010.tif"
bh = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/IM3_MeanHeight_BuildingFraction_2010_2020/MeanHeight_2010_LA.tif"
comm = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/HISDAC/Count/C/Count_2010_C.tif"
gov = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/HISDAC/Count/GV/Count_2010_GV.tif"
ind = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/HISDAC/Count/I/Count_2010_I.tif"
resi = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/HISDAC/Count/RI/Count_2010_RI.tif"
reso = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/HISDAC/Count/RO/Count_2010_RO.tif"
lumaj = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/HISDAC/Majority/Majority/Majority_2010.tif"

tmpfilepath = os.path.join(trainbasedir,"tmp")
numtiles = 10000
TILE_SIZE = 256
os.chdir(UtilityScriptFolder)
from BPspatlibv0 import *
np.random.seed(numpyseed)

# 2. INITIALIZE DIRECTORIES & CLEANUP
# Create all necessary output folders and clear any files from previous runs.
if not os.path.isdir(trainbasedir):
    os.mkdir(trainbasedir)
for subfolder in ["central", "BFrac", "BHeight","tmp","LU"]:
    folder_path = os.path.join(trainbasedir, subfolder)
    os.makedirs(folder_path, exist_ok=True)
    pattern = os.path.join(folder_path, f"LULC_{numpyseed}_*")
    for f in glob(pattern):
        os.remove(f)

#3. PRE-COMPUTATION 
# Clip the national NLCD raster to the Los Angeles Urban Area for localized analysis.

msa = gpd.read_file(pathtomsafile)
la_geom = msa.loc[msa.NAME10.str.contains("Los Angeles--Long Beach"),]
outnlcd = os.path.join(tmpfilepath, "nlcd2010LA.tif")
with rasterio.open(nlcd) as src:
    basecrs = src.crs
if not os.path.isfile(outnlcd):
    clipraster(nlcd, outnlcd, polygon=la_geom.to_crs(basecrs), polygon_path=None, tmp_dir=tmpfilepath)

# Read metadata from the clipped NLCD raster.
with rasterio.open(outnlcd) as src:
    nlcd_meta = src.meta.copy()
    nlcd_transform = src.transform
    nlcd_shape = src.shape
    valid_lulc_mask = src.read(1) > 0 # Mask of pixels with valid data.
    nlcd_data = src.read(1) 
    ref_crs = src.crs
    ref_width = src.width
    ref_height = src.height

## Align BF and BH
bffileint = os.path.join(tmpfilepath,"BF2010a.tif")
bhfileint = os.path.join(tmpfilepath,"BH2010a.tif")
bffileout = os.path.join(tmpfilepath,"BF2010.tif")
bhfileout = os.path.join(tmpfilepath,"BH2010.tif")
align_raster_to_reference(bf,outnlcd,bffileint)
align_raster_to_reference(bh,outnlcd,bhfileint)

## Clip BF and BH
clipraster(bffileint,bffileout, la_geom.to_crs(basecrs), tmp_dir= tmpfilepath)
clipraster(bhfileint,bhfileout, la_geom.to_crs(basecrs), tmp_dir= tmpfilepath)

#Align Hisdac data
[align_raster_to_reference(i,outnlcd,os.path.join(tmpfilepath,os.path.basename(i)),resampling=gdal.GRA_NearestNeighbour) for i in [comm,gov,ind,resi,reso,lumaj]]
lumajout = os.path.join(tmpfilepath,"Majority_2010c.tif")
clipraster(os.path.join(tmpfilepath,os.path.basename(lumaj)),lumajout, la_geom.to_crs(basecrs), tmp_dir=tmpfilepath)

with rasterio.open(lumajout) as src:
    #2,3,5,6 [Commer,Industrial,ResidentialI,ResidentialO,]
    ludat = src.read(1)
    ludat[np.isin(ludat,[1,4,7,8])] = 0

selected_tiles = sample_tiles_with_all_values(valid_lulc_mask,ludat.astype(np.int8),numtiles,numpyseed)
print(f"\nPre-computation complete. Selected {len(selected_tiles)} tiles for generation.")

# 5. FINAL DATA GENERATION
print("\n Starting Final Tile Generation")

for count, (rowfull, colfull) in enumerate(tqdm(selected_tiles, desc="Generating Tiles")):
    try:
        with rasterio.open(outnlcd) as src:
            window = Window(colfull, rowfull, TILE_SIZE, TILE_SIZE)
            raslist = src.read(1, window=window)
            raslist[np.isnan(raslist)] = 0 
            rastranslist = src.window_transform(window)
            central_kwargs = src.meta.copy()
            nlcd_meta = central_kwargs.copy()
            central_kwargs.update({
                'height': TILE_SIZE, 'width': TILE_SIZE, 'transform': rastranslist,'nodata': None
            })
            central_path = os.path.join(trainbasedir, "central", f"LULC_{numpyseed}_{count}.tif")
            with rasterio.open(central_path, 'w', **central_kwargs) as dst:
                dst.write(raslist, indexes=1)

        with rasterio.open(bffileout) as src:
            window = Window(colfull, rowfull, TILE_SIZE, TILE_SIZE)
            raslist = src.read(1, window=window)
            raslist[np.isnan(raslist)] = 0 
            raslist[raslist > 1] = 1
            rastranslist = src.window_transform(window)
            central_kwargs = nlcd_meta.copy()
            central_kwargs.update({
                'height': TILE_SIZE, 'width': TILE_SIZE, 'transform': rastranslist,'dtype':'float32','nodata': None
            })
            central_path = os.path.join(trainbasedir, "BFrac", f"LULC_{numpyseed}_{count}.tif")
            with rasterio.open(central_path, 'w', **central_kwargs) as dst:
                dst.write(raslist, indexes=1)
        
        with rasterio.open(bhfileout) as src:
            window = Window(colfull, rowfull, TILE_SIZE, TILE_SIZE)
            raslist = src.read(1, window=window)
            raslist[np.isnan(raslist)] = 0 
            rastranslist = src.window_transform(window)
            central_kwargs = nlcd_meta.copy()
            central_kwargs.update({
                'height': TILE_SIZE, 'width': TILE_SIZE, 'transform': rastranslist,'dtype':'float32','nodata': None
            })
            central_path = os.path.join(trainbasedir, "BHeight", f"LULC_{numpyseed}_{count}.tif")
            with rasterio.open(central_path, 'w', **central_kwargs) as dst:
                dst.write(raslist, indexes=1)
            
        with rasterio.open(lumajout) as src:
            window = Window(colfull, rowfull, TILE_SIZE, TILE_SIZE)
            raslist = src.read(1, window=window)
            raslist[np.isnan(raslist)] = 0
            rastranslist = src.window_transform(window)
            central_kwargs = nlcd_meta.copy()
            central_kwargs.update({
                'height': TILE_SIZE, 'width': TILE_SIZE, 'transform': rastranslist,'nodata': None
            })
            central_path = os.path.join(trainbasedir, "LU", f"LULC_{numpyseed}_{count}.tif")
            with rasterio.open(central_path, 'w', **central_kwargs) as dst:
                dst.write(raslist, indexes=1)

    except Exception as e:
        print(f"Failed to process tile {count} at (row={rowfull}, col={colfull}). Error: {e}")


source_base_dir = trainbasedir
dest_base_dir = testbasedir
subfolders = ['BFrac', 'BHeight', 'central', 'LU']
num_files_to_move = 2000

def create_test_dataset(source_dir, dest_dir, subfolder_list, num_to_move, seed=42):
    """
    Randomly selects files from a source directory and moves them to a destination
    directory, maintaining the subfolder structure.

    Args:
        source_dir (str): The path to the main training data folder.
        dest_dir (str): The path to the main test data folder to be created.
        subfolder_list (list): A list of strings with the subfolder names.
        num_to_move (int): The number of unique files to move.
        seed (int): A seed for the random number generator for reproducibility.
    """
    print("Starting the process...")
    random.seed(seed)

    print(f"Creating destination directory structure at: '{dest_dir}'")
    for folder in subfolder_list:
        path = os.path.join(dest_dir, folder)
        os.makedirs(path, exist_ok=True)

    try:
        sample_source_folder = os.path.join(source_dir, subfolder_list[0])
        all_filenames = os.listdir(sample_source_folder)
        print(f"Found {len(all_filenames)} total files in the source directory.")
    except FileNotFoundError:
        print(f"ERROR: The source directory '{sample_source_folder}' was not found.")
        return

    if len(all_filenames) < num_to_move:
        print(f"ERROR: Cannot select {num_to_move} files from the {len(all_filenames)} available files.")
        return

    selected_filenames = random.sample(all_filenames, num_to_move)
    print(f"Randomly selected {len(selected_filenames)} unique filenames to move.")

    print("Moving files...")
    moved_count = 0
    for filename in selected_filenames:
        for folder in subfolder_list:
            source_path = os.path.join(source_dir, folder, filename)
            dest_path = os.path.join(dest_dir, folder, filename)

            if os.path.exists(source_path):
                shutil.move(source_path, dest_path)
            else:
                print(f"Warning: File not found and could not be moved: {source_path}")

        moved_count += 1
        if moved_count % 100 == 0:
            print(f"  ... moved {moved_count}/{num_to_move} sets of files ...")

    print(f"\nSuccess! Moved {moved_count} files for each of the {len(subfolder_list)} categories.")
create_test_dataset(trainbasedir,testbasedir,subfolders,num_files_to_move,numpyseed)