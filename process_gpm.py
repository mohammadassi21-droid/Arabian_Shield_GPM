import numpy as np
from osgeo import gdal
from scipy.ndimage import distance_transform_edt, uniform_filter

print("Opening slope.tif...")
ds = gdal.Open("slope.tif")
if ds is None:
    raise FileNotFoundError("slope.tif not found in repository root!")

band = ds.GetRasterBand(1)
slope = band.ReadArray()

# 1. Extract structural edge pixels (slope > 8 degrees for steep scarps/wadis)
print("Extracting structural edge pixels...")
edges = np.where(slope > 8, 1, 0).astype(np.uint8)

# 2. Calculate Euclidean Distance (Distance to Lineaments in meters)
print("Calculating Euclidean Distance...")
dist_pixels = distance_transform_edt(edges == 0)
dist_meters = (dist_pixels * 30.0).astype(np.float32)  # 30m SRTM pixel size

# 3. Calculate Lineament Density (Focal moving window sum / ~2000m radius)
print("Calculating Lineament Density...")
density = uniform_filter(edges.astype(np.float32), size=67)

# Helper function to write GeoTIFF rasters with spatial references
def save_raster(filename, data_array, reference_ds):
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        filename, 
        reference_ds.RasterXSize, 
        reference_ds.RasterYSize, 
        1, 
        gdal.GDT_Float32
    )
    out_ds.SetGeoTransform(reference_ds.GetGeoTransform())
    out_ds.SetProjection(reference_ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(data_array)
    out_ds.FlushCache()
    out_ds = None

print("Exporting lineament_distance.tif...")
save_raster("lineament_distance.tif", dist_meters, ds)

print("Exporting lineament_density.tif...")
save_raster("lineament_density.tif", density, ds)

print("Processing complete! Ready for download.")