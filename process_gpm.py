import os

import numpy as np
from osgeo import gdal
from scipy import ndimage


def open_slope_raster():
    candidate_paths = ["slope.tif", "slope"]
    for path in candidate_paths:
        if os.path.exists(path):
            dataset = gdal.Open(path, gdal.GA_ReadOnly)
            if dataset is not None:
                return dataset
    raise FileNotFoundError("No slope raster found. Expected 'slope.tif' or 'slope'.")


def write_geotiff(path, array, geotransform, projection, data_type):
    rows, cols = array.shape
    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(path, cols, rows, 1, data_type)
    if output is None:
        raise RuntimeError(f"Failed to create output raster: {path}")

    output.SetGeoTransform(geotransform)
    output.SetProjection(projection)

    band = output.GetRasterBand(1)
    band.WriteArray(array)
    band.FlushCache()
    output = None


def main():
    dataset = open_slope_raster()
    band = dataset.GetRasterBand(1)
    slope_array = band.ReadAsArray().astype(np.float32)
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    mask = np.isfinite(slope_array) & (slope_array > 8.0)
    structural_edges = mask.astype(np.uint8)

    distance_from_edges = ndimage.distance_transform_edt(~structural_edges.astype(bool)) * 30.0
    lineament_density = ndimage.uniform_filter(structural_edges.astype(np.float32), size=67)

    write_geotiff(
        "lineament_distance.tif",
        distance_from_edges.astype(np.float32),
        geotransform,
        projection,
        gdal.GDT_Float32,
    )
    write_geotiff(
        "lineament_density.tif",
        lineament_density.astype(np.float32),
        geotransform,
        projection,
        gdal.GDT_Float32,
    )

    dataset = None


if __name__ == "__main__":
    main()
