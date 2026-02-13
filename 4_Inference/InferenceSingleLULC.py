import os
from glob import glob
import sys
import pandas as pd
import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import re
from torch.nn import functional as F
import torch
import torch.nn as nn
import time
import geopandas as gpd
import scipy
import numpy.ma as ma

UtilityScriptFolder = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Script"
os.chdir(UtilityScriptFolder)
from BPspatlibv0 import * # Import custom library for GIS operations.
def rescale_array(arr, new_min, new_max):
    """
    Rescales a NumPy array linearly to a new custom range [new_min, new_max].
    """
    old_min = np.min(arr)
    old_max = np.max(arr)
    
    # Handle the case where the input array has a single value
    if old_max - old_min == 0:
        return np.full_like(arr, (new_min + new_max) / 2)
    
    # Apply the linear scaling formula
    return (arr - old_min)

os.chdir("//Users/9oy/Documents/Projects/IM3/EvaluationP/Script/4_Inference")
import Generator

nlcdLA10 = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/nlcd2010LA.tif"
pathtomsafile = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/Boundary/US_urb_area_2010.shp"
nlcdCONUS20 = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/Annual_NLCD_LndCov_2020_CU_C1V1.tif"
nlcdLA20 = os.path.join("/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/", "nlcd2020LA.tif")
bf2010 = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BF2010.tif"
bh2010 = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BH2010.tif"
bf2020 = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/Updated_IM3_images/BF_2020.tif"
bh2020 = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/IM3_MeanHeight_BuildingFraction_2010_2020/MeanHeight_2020_LA.tif"
tmpfilepath = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/"
bffileint = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BF_2020a.tif"
bhfileint = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BH_2020a.tif"
bffileout20 = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BF_2020.tif"
bhfileout20 = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BH_2020.tif"

msa = gpd.read_file(pathtomsafile)
la_geom = msa.loc[msa.NAME10.str.contains("Los Angeles--Long Beach"),]

with rio.open(nlcdLA10) as src:
    basecrs = src.crs
if not os.path.isfile(nlcdLA20):
    clipraster(nlcdCONUS20, nlcdLA20, polygon=la_geom.to_crs(basecrs), polygon_path=None, tmp_dir="/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/")

align_raster_to_reference(bf2020,nlcdLA20,bffileint)
align_raster_to_reference(bh2020,nlcdLA20,bhfileint)

clipraster(bffileint,bffileout20, la_geom.to_crs(basecrs), tmp_dir= tmpfilepath)
clipraster(bhfileint,bhfileout20, la_geom.to_crs(basecrs), tmp_dir= tmpfilepath)

with rio.open(bffileout20) as src:
    bf2020data = src.read(1)
with rio.open(bhfileout20) as src:
    bh2020data = src.read(1)* 0.3048 
    bh2020data[bh2020data>75] = 75

#lr = [0.0001,0.0002,0.0005]
lr = [0.0002]

outpath = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/4_Inference/Outputs"
for lri in lr:
    params = {
        'model': f"/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/2_EvalP/v1/BF/LULCCond_BFgenerator_withLatentVector_100L_epoch_1000_LR_{lri}.pth",
        'ProjLULC': nlcdLA20,
        'model1': f"/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/2_EvalP/v1/BH/BFCond_BHgenerator_withLatentVector_100L_epoch_1000_LR_{lri}.pth",
        "BFfile":bf2010,
        "BHfile": bh2010,
        "CurLULC": nlcdLA10,
        "seed":1234
    }

    inference = Generator.MorphInf(params)
    inference.initinference()
    inference.initmodel()
    inference.runinference()
    inference.BH = rescale_array(inference.BH,0,75)
    #inference.BH[inference.BH>75] = 75
    cmask1 = inference.ProjLULC != inference.CurLULC
    cmask2 = np.isin(inference.ProjLULC, [22,23,24])
    cmask = cmask1 & cmask2

    origbf = bf2020data[cmask]
    origbh = bh2020data[cmask]
    predbf = inference.BF[cmask]
    predbh = inference.BH[cmask]

    outdf= pd.DataFrame({"origbf":origbf,"origbh":origbh,"predbf":predbf,"predbh":predbh})
    outdf.to_csv(os.path.join(outpath,f"Evaldf_{lri}.csv"))

    print(np.corrcoef(origbf,predbf)[0,1])
    print(np.corrcoef(origbh,predbh)[0,1])


def add_equation_and_r2(ax, slope, intercept, r_value, x_pos=0.05, y_pos_eq=0.95, y_pos_r2=0.90, color='black', fontsize=12):
    r_squared = r_value#**2
    if intercept >= 0:
        equation_text = r'$y = {:.2f}x + {:.2f}$'.format(slope, intercept)
    else:
        # Use abs(intercept) to avoid double negative (e.g., + -5 becomes - 5)
        equation_text = r'$y = {:.2f}x - {:.2f}$'.format(slope, abs(intercept))
    r2_text = r'$r = {:.2f}$'.format(r_squared) # LaTeX for math
    ax.text(x_pos, y_pos_eq, equation_text, transform=ax.transAxes,
            fontsize=fontsize, color=color, verticalalignment='top')
    ax.text(x_pos, y_pos_r2, r2_text, transform=ax.transAxes,
            fontsize=fontsize, color=color, verticalalignment='top')

fig, ax = plt.subplots(figsize=(8, 6))
plt.hexbin(origbf,predbf,bins="log")#alpha=0.1)
plt.xlabel("Real",fontsize=14)
plt.ylabel("Generated",fontsize=14)
slp, interc, rval,pval,stnderr = scipy.stats.linregress(np.array(origbf),np.array(predbf))
add_equation_and_r2(ax, slp, interc, rval, x_pos=0.05, y_pos_eq=0.99, y_pos_r2=0.95, color='k', fontsize=14)
plt.ylim(0,1.1)
plt.colorbar()
plt.suptitle("Building Footprint",fontsize=16)
plt.title("in Pixels with New Development (2010-2020)\n n=150,166")
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
x= origbh#[(origbh <=15) & (predbh <=15)]
y= predbh#[(origbh <=15) & (predbh <=15)]
plt.hexbin(x,y,bins="log")#,alpha=0.005)
plt.xlabel("Real",fontsize=14)
plt.ylabel("Generated",fontsize=14)
slp, interc, rval,pval,stnderr = scipy.stats.linregress(x,y)
add_equation_and_r2(ax, slp, interc, rval, x_pos=0.05, y_pos_eq=0.95, y_pos_r2=0.91, color='k', fontsize=14)
plt.colorbar()
plt.suptitle("Building Height",fontsize=16)
plt.title("in Pixels with New Development (2010-2020) \n n=150,166")
plt.show()

def calculate_window_averages(data: np.ndarray, mask: np.ndarray, window_size: int = 256) -> np.ndarray:
    """
    Slides a window over the data/mask arrays without overlap, calculates the 
    average of data elements where the mask is True in each window.
    """
    s = window_size
    H, W = data.shape[0] // s, data.shape[1] // s

    # 1. Trim arrays to full block size and reshape into blocks (H_blocks, W_blocks, s, s)
    d_blocks = data[:H*s, :W*s].reshape(H, s, W, s).transpose(0, 2, 1, 3).reshape(-1, s, s)
    m_blocks = mask[:H*s, :W*s].reshape(H, s, W, s).transpose(0, 2, 1, 3).reshape(-1, s, s)
    
    # 2. Calculate the masked mean for each flattened block (d[m].mean())
    # Note: If a mask is all False, d[m] is empty, and .mean() returns NaN, which is safe.
    averages = [d[m].mean() for d, m in zip(d_blocks, m_blocks)]
    return(np.array(averages))


test1 = calculate_window_averages(inference.BF,cmask,256)
test2 = calculate_window_averages(bf2020data,cmask,256)

a=ma.masked_invalid(test2)
b=ma.masked_invalid(test1)
msk = (~a.mask & ~b.mask)
fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(a[msk],b[msk],c="k",alpha=0.75)
plt.axline((0,0),slope=1,c="gray")
slp, interc, rval,pval,stnderr = scipy.stats.linregress(b[msk],a[msk])
add_equation_and_r2(ax, slp, interc, rval, x_pos=0.05, y_pos_eq=0.95, y_pos_r2=0.88, color='k', fontsize=14)
plt.xlabel("Real Average Building Footprint: 2020",fontsize=14)
plt.ylabel("Generated Average Building Footprint: 2020",fontsize=14)
plt.suptitle("Evaluating Newly Developed Pixels (2010-2020)",fontsize=18)
plt.title("Average Building Footprint Growth\n within 256x256 pixels region",fontsize=15)
plt.tight_layout()
plt.show()

test1 = calculate_window_averages(inference.BH,cmask,256)
test2 = calculate_window_averages(bh2020data,cmask,256)

import numpy.ma as ma
a=ma.masked_invalid(test2)
b=ma.masked_invalid(test1)
msk = (~a.mask & ~b.mask)
fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(a[msk],b[msk],c="k",alpha=0.75)
plt.axline((0,0),slope=1,c="gray")
slp, interc, rval,pval,stnderr = scipy.stats.linregress(a[msk],b[msk])
add_equation_and_r2(ax, slp, interc, rval, x_pos=0.05, y_pos_eq=0.95, y_pos_r2=0.88, color='k', fontsize=14)
plt.xlabel("Real Average Building Heights(m): 2020",fontsize=14)
plt.ylabel("Generated Average Building Heights(m): 2020",fontsize=14)
plt.suptitle("Evaluating Newly Developed Pixels (2010-2020)",fontsize=18)
plt.title("Average Building Height Growth\n within 256x256 pixels region",fontsize=15)
plt.tight_layout()
plt.show()

def export_to_geotiff(output_array: np.ndarray, output_path: str, source_path: str):
    """
    Writes a 2D NumPy array to a GeoTIFF file, copying georeferencing (CRS, 
    transform) from a specified source file.
    
    Args:
        output_array (np.ndarray): The 2D array data to write.
        output_path (str): The full path for the output GeoTIFF file.
        source_path (str): The path to an existing GeoTIFF file to copy metadata from.
    """
    try:
        with rio.open(source_path, 'r') as src:
            profile = src.profile
    except rio.RasterioIOError as e:
        print(f"Error opening source file {source_path}: {e}")
        return

    # 2. Update the profile with the new array's dimensions and data type
    profile.update(
        dtype=output_array.dtype,
        height=output_array.shape[0],
        width=output_array.shape[1],
        count=1,
        nodata=np.nan  # Set NaN as NoData
    )

    # 3. Write the array to the new GeoTIFF file
    try:
        with rio.open(output_path, 'w', **profile) as dst:
            # Write the 2D array (assuming band 1)
            dst.write(output_array, 1)
        print(f"Successfully exported data to {output_path}")
    except Exception as e:
        print(f"Error writing output file {output_path}: {e}")

