import csv
import os

import numpy as np
from osgeo import gdal, ogr
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


def validate_gold_prospectivity(raster_path, vector_path):
    if not os.path.exists(vector_path):
        print(f"Notice: validation vector '{vector_path}' not found; skipping mineral occurrence validation.")
        return

    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if raster is None:
        print(f"Notice: could not open raster '{raster_path}' for validation; skipping mineral occurrence validation.")
        return

    vector_data = ogr.Open(vector_path)
    if vector_data is None:
        print(f"Notice: could not open vector '{vector_path}' for validation; skipping mineral occurrence validation.")
        return

    band = raster.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float32)
    geotransform = raster.GetGeoTransform()
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    records = []
    layer = vector_data.GetLayer()
    if layer is None:
        print(f"Notice: vector layer in '{vector_path}' is empty; skipping mineral occurrence validation.")
        return

    for feature_idx, feature in enumerate(layer):
        geom = feature.GetGeometryRef()
        if geom is None:
            continue

        if geom.GetGeometryType() == ogr.wkbPoint:
            x, y, _ = geom.GetPoint()
            point_geometries = [(x, y)]
        elif geom.GetGeometryType() == ogr.wkbMultiPoint:
            point_geometries = []
            for point_idx in range(geom.GetGeometryCount()):
                sub_geom = geom.GetGeometryRef(point_idx)
                if sub_geom is not None and sub_geom.GetGeometryType() == ogr.wkbPoint:
                    x, y, _ = sub_geom.GetPoint()
                    point_geometries.append((x, y))
        else:
            point_geometries = []

        for x, y in point_geometries:
            col = int((x - geotransform[0]) / geotransform[1])
            row = int((y - geotransform[3]) / geotransform[5])
            if 0 <= col < cols and 0 <= row < rows:
                value = float(array[row, col])
                records.append({
                    "feature_id": feature_idx,
                    "x": x,
                    "y": y,
                    "prospectivity_value": value,
                })

    if not records:
        print("Notice: no valid gold occurrence points intersected the raster; writing empty validation summary.")
        with open("gold_prospectivity_validation.csv", "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["feature_id", "x", "y", "prospectivity_value"])
            writer.writerow([])
            writer.writerow(["statistic", "value"])
            writer.writerow(["mean", ""])
            writer.writerow(["median", ""])
            writer.writerow(["min", ""])
            writer.writerow(["max", ""])
            writer.writerow(["p75", ""])
        return

    values = np.array([record["prospectivity_value"] for record in records], dtype=np.float32)
    summary = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p75": float(np.percentile(values, 75)),
    }

    with open("gold_prospectivity_validation.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["feature_id", "x", "y", "prospectivity_value"])
        for record in records:
            writer.writerow([
                record["feature_id"],
                record["x"],
                record["y"],
                record["prospectivity_value"],
            ])
        writer.writerow([])
        writer.writerow(["statistic", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])

    print(f"Validated {len(records)} gold occurrence points; summary saved to gold_prospectivity_validation.csv")


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

    density_nonzero = lineament_density[lineament_density > 0]
    density_mean = np.mean(density_nonzero) if density_nonzero.size > 0 else 0.0
    fuzzy_density = 1.0 / (1.0 + (density_mean / (lineament_density + 1e-6)) ** 2)

    dist_mean = np.mean(distance_from_edges)
    fuzzy_distance = 1.0 / (1.0 + (distance_from_edges / (dist_mean + 1e-6)) ** 2)

    fuzzy_and = np.minimum(fuzzy_distance, fuzzy_density)
    fuzzy_or = np.maximum(fuzzy_distance, fuzzy_density)
    gamma = 0.75
    fuzzy_gamma = (fuzzy_and ** (1.0 - gamma)) * (fuzzy_or ** gamma)

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
    write_geotiff(
        "fuzzy_lineament_density.tif",
        fuzzy_density.astype(np.float32),
        geotransform,
        projection,
        gdal.GDT_Float32,
    )
    write_geotiff(
        "fuzzy_lineament_distance.tif",
        fuzzy_distance.astype(np.float32),
        geotransform,
        projection,
        gdal.GDT_Float32,
    )
    write_geotiff(
        "fuzzy_structural_overlay.tif",
        fuzzy_gamma.astype(np.float32),
        geotransform,
        projection,
        gdal.GDT_Float32,
    )

    validate_gold_prospectivity("fuzzy_structural_overlay.tif", "mods_gold_occurrences.geojson")

    dataset = None


if __name__ == "__main__":
    main()
