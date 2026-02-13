import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from glob import glob
from shapely.ops import transform
from functools import partial
from shapely import geometry
import scipy
import os
import pyogrio
import pandas as pd
import rasterio
import statsmodels.api as sm
def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2,slope,intercept,p_value,std_err

# Set directory
if(not os.path.isfile("/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/HeteroScalingLA.csv")):
    buildingsfile = "/Users/9oy/Documents/Data/IM3/buildingspoly_CApy.gpkg" ## generated using buildingspoly_CApy.py
    buildingsfilecrs = pyogrio.read_info(buildingsfile)["crs"]

    extentbpolygon = "/Users/9oy/Documents/Projects/IM3/Scripts/Analysis/Validation/Data/raster_extent.shp"
    extentpoly = gpd.read_file(extentbpolygon,engine="pyogrio",use_arrow=True).to_crs(buildingsfilecrs)#("ESRI:102003")
    minx, miny, maxx, maxy = extentpoly.total_bounds
    bbox_tuple = (minx, miny, maxx, maxy)
    buildings = gpd.read_file(buildingsfile,engine="pyogrio",use_arrow=True,bbox=bbox_tuple) ## 210
    buildings = buildings.to_crs("ESRI:102003")
    buildings["FootprintArea"] = buildings.area
    buildings['Heightm'] = buildings['Height'] * 0.3048
    buildings.loc[buildings['Heightm'] > 75, 'Heightm'] = 75

    buildings['Volume'] = buildings['FootprintArea'] * buildings['Heightm'] ## ft to m conversion
    ### Note Buildings Height is in Ft

    minX, minY, maxX, maxY = buildings.total_bounds
    # Define iterations
    iters = np.append(np.array(50),np.arange(100, 30000, 100))#np.arange(500, 30000, 500) # 
    means = np.empty(len(iters))
    varnc = np.empty(len(iters))

    # Iterate over cell sizes
    for i, it in enumerate(iters):
        print(it)
        # Generate fishnet grid
        # Create a fishnet
        x, y = (minX, minY)
        geom_array = []
        # Polygon Size
        square_size = it
        while y <= maxY:
            while x <= maxX:
                geom = geometry.Polygon([(x,y), (x, y+square_size), (x+square_size, y+square_size), (x+square_size, y), (x, y)])
                geom_array.append(geom)
                x += square_size
            x = minX
            y += square_size
        fish = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs(buildings.crs)
        fish['ID'] = range(1, len(fish) + 1)
        # Perform intersection with buildings and calculate summary statistics
        test = fish.sjoin(buildings, how="left")

        test_ = test.groupby('ID_left')['Volume'].sum().reset_index()
        test_ = test_[test_['Volume'] > 0]
        means[i] = test_['Volume'].mean()
        varnc[i] = test_['Volume'].std()**2

    means = np.append(means,buildings['Volume'].mean())
    varnc = np.append(varnc,buildings['Volume'].std()**2)
    iters = np.append(iters,np.sqrt(buildings.FootprintArea.min()))

    outdf = pd.DataFrame({"Resolution": iters,"Mean":means,"Variance":varnc})
    outdf.to_csv("/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/HeteroScalingLA.csv")
else:
    outdf = pd.read_csv("/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/HeteroScalingLA.csv")

fig, ax = plt.subplots(figsize=(8, 6), nrows=1, ncols=1)
plt.scatter(outdf.Mean, outdf.Variance, c=outdf.Resolution/1000, s=60, alpha=0.7, edgecolors="none")
plt.xscale("log")
plt.yscale("log")
log_x = np.log(outdf.Mean)
log_y = np.log(outdf.Variance)
linregress_result1 = scipy.stats.linregress(log_x, log_y)
slope, intercept, r_value, p_value, std_err = linregress_result1
# Regression line
x_line = np.logspace(np.log10(outdf.Mean.min()), np.log10(outdf.Mean.max()), num=100)
log_y_line = slope * np.log(x_line) + intercept
y_line = np.exp(log_y_line)
plt.plot(x_line, y_line, color="k", lw=2.5, alpha=0.5)
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Grid Resolution (km)', rotation=270, fontsize=16)
plt.ylabel(r'${\sigma^2}$', fontsize=20)
plt.xlabel(r'${\mu}$', fontsize=20)
plt.title("Building Volume: 2015", fontsize=22)

X = sm.add_constant(log_x) 
Y = log_y
model = sm.OLS(Y, X)
results = model.fit()
ci_95 = np.array(results.conf_int(alpha=0.01))
print(ci_95)
# Regression line
log_y_line_low = ci_95[1,0] * np.log(x_line) + ci_95[0,0]
y_line_low = np.exp(log_y_line_low)
plt.plot(x_line, y_line_low, color="gray", lw=2.5, alpha=0.5,linestyle="--")
# Regression line
log_y_line_high= ci_95[1,1] * np.log(x_line) + ci_95[0,1]
y_line_high = np.exp(log_y_line_high)
plt.plot(x_line, y_line_high, color="gray", lw=2.5, alpha=0.5,linestyle="--")

# Annotations
equation_text = r'$\sigma^2 = {:.2f} \cdot \mu^{{{:.2f}}}$'.format(np.exp(intercept), slope)
r2_text = r'$R^2 = {:.3f}$'.format(r_value**2)

plt.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')
plt.text(0.05, 0.90, r2_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')
plt.show()

with rasterio.open("/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BF2020_GAN.tif") as src:
    GANBF2020 = src.read(1)
    GANBF2020[GANBF2020>1] = 1 

with rasterio.open("/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BH2020_GAN.tif") as src:
    GANBH2020 = src.read(1)

with rasterio.open("/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/ChangeMask2010_2020.tif") as src:
    cmask = src.read(1)

GANBV2020 = GANBF2020 * 30 * 30 * GANBH2020

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
    variances = [d[m].std()**2 for d, m in zip(d_blocks, m_blocks)]
    averages = [d[m].mean() for d, m in zip(d_blocks, m_blocks)]
    return(np.array(averages),np.array(variances))

meancmask, varscmask = calculate_window_averages(GANBV2020,cmask.astype(bool),256)
GAN2020 = pd.DataFrame({"Mean":meancmask,"Variance": varscmask}).dropna().reset_index(drop=True)
slope, intercept, r_value, p_value, std_err = linregress_result1
GAN2020["varpredicted"] = np.exp(slope * np.log(GAN2020.Mean) + intercept)
GAN2020["varpredicted_low"] = np.exp(ci_95[1,0] * np.log(GAN2020.Mean) + ci_95[0,0])
GAN2020["varpredicted_high"] = np.exp(ci_95[1,1] * np.log(GAN2020.Mean) + ci_95[0,1])
xerr_min = GAN2020["varpredicted"] - GAN2020["varpredicted_low"]
xerr_max = GAN2020["varpredicted_high"] - GAN2020["varpredicted"]
xerr = np.array([xerr_min.values, xerr_max.values])

plt.scatter(GAN2020.Variance,GAN2020.varpredicted,c="k",alpha=0.25)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Generated Variance")
plt.ylabel("Expected Variance")
plt.grid()
plt.errorbar(y=GAN2020.varpredicted,x=GAN2020.Variance,yerr = xerr, fmt='o',alpha=0.5,c='k')
plt.xlim(10**4, 10**7)
plt.ylim(10**4, 10**7)
plt.plot([10**4,10**7],[10**4,10**7],c='gray')
linregress_result = scipy.stats.linregress(np.log(GAN2020.Variance), np.log(GAN2020.varpredicted))
slope, intercept, r_value, p_value, std_err = linregress_result
equation_text = r'$\beta = {:.2f}$'.format(slope)
r2_text = r'$R^2 = {:.3f}$'.format(r_value**2)
X = sm.add_constant(np.log(GAN2020.Variance)) 
Y = np.log(GAN2020.varpredicted)
model = sm.OLS(Y, X)
results = model.fit()
ci_95 = np.array(results.conf_int(alpha=0.05))
plt.text(0.02, 0.75, equation_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')
plt.text(0.02, 0.65, r2_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')
plt.text(0.02, 0.70, f"95% CI: [{ci_95[1,0]:.2f}, {ci_95[1,1]:.2f}]", transform=ax.transAxes, fontsize=10,verticalalignment='top')
plt.show()

np.sum(GAN2020.varpredicted > GAN2020.Variance)
np.sum(GAN2020.varpredicted_low > GAN2020.Variance)


linregress_result = scipy.stats.linregress(GAN2020.Variance, GAN2020.varpredicted_low)
slope, intercept, r_value, p_value, std_err = linregress_result
print(linregress_result)

linregress_result = scipy.stats.linregress(GAN2020.Variance, GAN2020.varpredicted_high)
slope, intercept, r_value, p_value, std_err = linregress_result
print(linregress_result)

x_line = np.logspace(np.log10(GAN2020.Mean.min()), np.log10(GAN2020.Mean.max()), num=100)
log_y_line = slope * np.log(x_line) + intercept
y_line = np.exp(log_y_line)
plt.plot(x_line, y_line, color="k", lw=2.5, alpha=0.5)
