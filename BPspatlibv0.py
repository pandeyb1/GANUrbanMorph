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
from collections import defaultdict

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

def clipraster(inputrasterpath, outputrasterpath, polygon=None, polygon_path=None, tmp_dir=None):
    """
    This function clips an input raster using a polygon geometry (geopandas polygon dataframe or a polygon file) and generates an output raster.
    """
    if polygon is not None:
        if tmp_dir is None:
            raise ValueError("tmp_dir must be provided if polygon is a GeoDataFrame.")
        temp_shapefile = os.path.join(tmp_dir, 'temp.shp')
        polygon.to_file(temp_shapefile)
    elif polygon_path is not None:
        temp_shapefile = polygon_path
    else:
        raise ValueError("Either polygon or polygon_path must be provided.")

    gdal.Warp(outputrasterpath, inputrasterpath,
              cutlineDSName=temp_shapefile,
              cropToCutline=True,
              dstNodata=0)

    if polygon is not None:
        # clean up temp files
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            try:
                os.remove(temp_shapefile.replace('.shp', ext))
            except FileNotFoundError:
                pass

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

def rasters_align(input_raster_path, reference_raster_path, tolerance=1e-6):
    """Check if input raster aligns with reference raster."""
    input_ds = gdal.Open(input_raster_path)
    ref_ds = gdal.Open(reference_raster_path)

    # Get basic metadata
    in_proj = input_ds.GetProjection()
    in_gt = input_ds.GetGeoTransform()
    in_size = (input_ds.RasterXSize, input_ds.RasterYSize)

    ref_proj = ref_ds.GetProjection()
    ref_gt = ref_ds.GetGeoTransform()
    ref_size = (ref_ds.RasterXSize, ref_ds.RasterYSize)

    # Compare projection
    if in_proj != ref_proj:
        return False

    # Compare geotransform with some tolerance
    for i in range(6):
        if abs(in_gt[i] - ref_gt[i]) > tolerance:
            return False

    # Compare size
    if in_size != ref_size:
        return False

    return True

def align_raster_to_reference(input_raster_path, reference_raster_path, output_raster_path=None,resampling=gdal.GRA_Bilinear):
    """Align input raster to reference raster if not already aligned. Returns aligned raster path."""
    
    if rasters_align(input_raster_path, reference_raster_path):
        print("Rasters already aligned.")
        return input_raster_path  # No need to reproject

    print("Rasters not aligned. Reprojecting and aligning...")

    # Open reference raster
    ref_ds = gdal.Open(reference_raster_path)
    ref_proj = ref_ds.GetProjection()
    ref_gt = ref_ds.GetGeoTransform()
    ref_width = ref_ds.RasterXSize
    ref_height = ref_ds.RasterYSize

    # Calculate bounds from geotransform
    xmin = ref_gt[0]
    xres = ref_gt[1]
    xmax = xmin + xres * ref_width
    ymax = ref_gt[3]
    yres = ref_gt[5]
    ymin = ymax + yres * ref_height

    # Default output path
    if output_raster_path is None:
        base, ext = os.path.splitext(input_raster_path)
        output_raster_path = base + "_aligned.tif"

    # Set up warp options
    warp_options = gdal.WarpOptions(
        format='GTiff',
        outputBounds=[xmin, ymin, xmax, ymax],
        width=ref_width,
        height=ref_height,
        dstSRS=ref_proj,
        resampleAlg=resampling
    )

    # Perform reprojection/alignment
    gdal.Warp(output_raster_path, input_raster_path, options=warp_options)

    print(f"Reprojected and aligned raster saved to: {output_raster_path}")
    return output_raster_path

def sample_tiles_with_all_values(mask: np.ndarray, data: np.ndarray, N: int, TILE_SIZE: int, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if mask.shape != data.shape:
        raise ValueError("mask and data must have the same shape")
    
    rows, cols = mask.shape

    # We only consider top-left corners where the tile fits inside the array boundaries
    valid_tile_positions = []
    tile_value_map = defaultdict(list)  # value -> list of valid top-left positions of tiles containing that value

    # Scan all possible top-left positions for tiles of size TILE_SIZE
    for r in range(rows - TILE_SIZE + 1):
        for c in range(cols - TILE_SIZE + 1):
            # Check if the entire tile mask is valid (all ones)
            tile_mask = mask[r:r+TILE_SIZE, c:c+TILE_SIZE]
            if np.all(tile_mask == 1):
                # Extract tile data values
                tile_data = data[r:r+TILE_SIZE, c:c+TILE_SIZE]
                # Get unique values in this tile
                unique_vals_in_tile = np.unique(tile_data)
                # Store this tile position for all values it contains
                for val in unique_vals_in_tile:
                    tile_value_map[val].append((r, c))
                valid_tile_positions.append((r, c))
    
    unique_values = list(tile_value_map.keys())

    if len(unique_values) > N:
        raise ValueError(f"Cannot cover all {len(unique_values)} unique values with only N={N} samples")

    # Step 1: Ensure at least one tile per unique value is included
    selected_tiles = []
    for val in unique_values:
        candidates = tile_value_map[val]
        chosen = candidates[np.random.randint(len(candidates))]
        selected_tiles.append(chosen)

    # Remove duplicates in selected_tiles if the same tile covers multiple values
    selected_tiles = list(set(selected_tiles))

    remaining_slots = N - len(selected_tiles)

    if remaining_slots > 0:
        # Candidates for remaining slots are valid_tile_positions excluding already selected
        selected_set = set(selected_tiles)
        remaining_choices = [pos for pos in valid_tile_positions if pos not in selected_set]

        if len(remaining_choices) <= remaining_slots:
            # Take all remaining if fewer than needed
            selected_tiles.extend(remaining_choices)
        else:
            chosen_indices = np.random.choice(len(remaining_choices), remaining_slots, replace=False)
            for idx in chosen_indices:
                selected_tiles.append(remaining_choices[idx])

    return np.array(selected_tiles)

# UTILITY FUNCTIONS
def calculate_intersection_area(geometry, pixel_bounds):
    """Calculates the area of intersection between two geometries."""
    if geometry.is_valid:
        output = geometry.intersection(pixel_bounds).area
    else:
        output = shapely.make_valid(geometry).intersection(pixel_bounds).area
    return output

def get_pixel_bbox(r, c, transform):
    """Gets the shapely bounding box of a raster pixel."""
    top_left = transform * (c, r)
    bottom_right = transform * (c + 1, r + 1)
    return box(top_left[0], top_left[1], bottom_right[0], bottom_right[1])

def rasterize_chunk(buildings_chunk, out_shape, transform):
    """Worker function to rasterize a chunk of buildings for parallel processing."""
    shapes = ((geom, h) for geom, h in zip(buildings_chunk.geometry, buildings_chunk.Height))
    return features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.float32,
    )