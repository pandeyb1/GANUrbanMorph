import geopandas as gpd
from osgeo import gdal
from shapely import geometry
import shapely
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from rasterio import features
from glob import glob##
import os##
import numpy as np ## 
from pyproj import Transformer
from shapely.ops import transform

def genfish(total_bounds,cellsize,boundcrs):
    """
    This function generates a fishnet taking extent coordinates, cell size, and CRS as inputs.
    """
    minX, minY, maxX, maxY = total_bounds
    x, y = (minX, minY)
    geom_array = []
    square_size = cellsize
    while y < maxY:
        while x < maxX:
            geom = geometry.Polygon([(x,y), (x, y+square_size), (x+square_size, y+square_size), (x+square_size, y), (x, y)])
            geom_array.append(geom)
            x += square_size
        x = minX
        y += square_size
    fishnet = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs(boundcrs)
    fishnet['ID'] = range(1, len(fishnet) + 1)
    assert fishnet.intersects(shapely.Polygon([(minX, minY), (maxX, minY),(maxX, maxY), (minX, maxY)])).all()
    return(fishnet)

def clipraster(inputrasterpath,polygon,outputrasterpath,tmpfilepath=None,uid1=0,uid2=0):
    """
    This function clips an input raster using a polygon geometry (geopandas polygon dataframe or a polygon file) and generates an output raster.
    Note that it requires 
    """
    if tmpfilepath != None:
        temp_shapefile = os.path.join(tmpfilepath,'temp_polygon' + str(uid1) + str(uid2) +'.shp')
        polygon.to_file(temp_shapefile)
        gdal.Warp(outputrasterpath, inputrasterpath, 
                cutlineDSName=temp_shapefile,
                cropToCutline=True,
                dstNodata=0)  # Set nodata value to 0
        for ext in ['.shp', '.shx', '.dbf', '.prj','cpg']:
            temp_file = temp_shapefile.replace('.shp', ext)
            if os.path.exists(temp_file):
                os.remove(temp_file)
    else:
        temp_shapefile = tmpfilepath
        gdal.Warp(outputrasterpath, inputrasterpath, 
                cutlineDSName=temp_shapefile,
                cropToCutline=True,
                dstNodata=0)  # Set nodata value to 0
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            temp_file = temp_shapefile.replace('.shp', ext)
            if os.path.exists(temp_file):
                os.remove(temp_file)
    return(None)

def get_pixel_bbox(row, col, transform):
    """
    Calculate the bounding box of a pixel given its row and column.
    
    Args:
    row (int): The row number of the pixel.
    col (int): The column number of the pixel.
    transform (Affine): The affine transform of the raster.
    
    Returns:
    box: A Shapely box representing the bounding box of the pixel.
    """
    # Calculate the coordinates of the pixel corners
    x_top_left = transform.c + col * transform.a
    y_top_left = transform.f + row * transform.e
    
    x_bottom_right = x_top_left + transform.a
    y_bottom_right = y_top_left + transform.e
    
    # Create and return a Shapely box
    return box(min(x_top_left, x_bottom_right),
               min(y_top_left, y_bottom_right),
               max(x_top_left, x_bottom_right),
               max(y_top_left, y_bottom_right))