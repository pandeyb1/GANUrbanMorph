import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import box
import scipy
from shapely.ops import transform
from shapely import geometry
import sys,os
import glob
# Set directory
dir = "/lustre/or-scratch/cades-birthright/9oy/"
#Edited
dir = "/Users/9oy/Documents/Projects/IM3/Data"

#Edited
#i = int(sys.argv[1])
#i = 10

# Read buildings and MSA data

bfname = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/Data/buildings.gpkg"
msafname= "/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/Data/nhgis0039_shape/nhgis0039_shapefile_tl2015_us_cbsa_2015/US_cbsa_2015.shp"

class bvms:
    def __init__(self,msafname,bfname):
        self.msa = gpd.read_file(msafname)
        self.msa = self.msa[self.msa['LSAD']=="M1"].copy()
        self.msa = self.msa[~self.msa['NAME'].str.contains('PR')]
        self.metronames = [x for x in self.msa['NAME']]
        self.GEOID = [x for x in self.msa['GEOID']]
        self.statenames = [x.split(",")[1].replace(" ", "") for x in self.metronames]
    def processmetro(self,metro):
        self.id = self.metronames.index(metro)
        self.selmetro = self.msa[self.msa['NAME'] == metro]
        self.extent = self.selmetro.total_bounds
        self.states = self.metronames[self.id].split(",")[1].replace(" ","")
        self.multistatescheck = len(self.states) > 2
        if self.multistatescheck:
            ##Multiple States
            self.indstates = self.states.split("-")
            #print(self.indstates)
            for i in range(len(self.indstates)):
                #print("Reading Buildings File")
                buildings = gpd.read_file(bfname,layer=self.indstates[i],engine='pyogrio', use_arrow=True)
                #print("Reading Buildings File Done!")
                self.selmetrogeo = self.selmetro.to_crs(buildings.crs)
                selbuildings = buildings[buildings.intersects(self.selmetrogeo.iloc[0].geometry)].copy().to_crs("ESRI:102003")
                selbuildings["FootPrintArea"] = selbuildings["Area2D"] * 0.09290304
                selbuildings['Heightm'] = selbuildings['Height'] * 0.3048 
                selbuildings['Volume'] = selbuildings['FootPrintArea'] * selbuildings['Heightm']
                if i == 0:
                    self.fullmetrobuildings = selbuildings.copy()
                else:
                    self.fullmetrobuildings = pd.concat([self.fullmetrobuildings,selbuildings])
        else:
            ## Single States
            #print("Reading Buildings File")
            buildings = gpd.read_file(bfname,layer=self.states,engine='pyogrio', use_arrow=True)
            #print("Reading Buildings File Done!")
            self.selmetrogeo = self.selmetro.to_crs(buildings.crs)
            selbuildings = buildings[buildings.intersects(self.selmetrogeo.iloc[0].geometry)].copy().to_crs("ESRI:102003")
            selbuildings["FootPrintArea"] = selbuildings["Area2D"] * 0.09290304
            selbuildings['Heightm'] = selbuildings['Height'] * 0.3048 
            selbuildings.loc[selbuildings['Heightm'] > 75, 'Heightm'] = 75
            selbuildings['Volume'] = selbuildings['FootPrintArea'] * selbuildings['Heightm'] 
            self.fullmetrobuildings = selbuildings.copy()
    
test = bvms(msafname,bfname)
mnames = test.metronames
print(len(mnames))

def runmetro(i,mnames):
    #print(i)
    if i ==210:
        outfile = "/Users/9oy/Documents/Projects/IM3/Scripts/Analysis/Validation/Data/MetroBuildings/" +"Louisville_Jefferson County, KY-IN" + ".gpkg"
    else:
        outfile = "/Users/9oy/Documents/Projects/IM3/Scripts/Analysis/Validation/Data/MetroBuildings/" + mnames[i] + ".gpkg"
    if os.path.exists(outfile) == False:
        test = bvms(msafname,bfname)
        test.processmetro(mnames[i])
        test.fullmetrobuildings.to_file(outfile)
    return(0)

from joblib import Parallel, delayed
results = Parallel(n_jobs=7, verbose=10)(delayed(runmetro)(i, mnames) for i in range(len(mnames)))
