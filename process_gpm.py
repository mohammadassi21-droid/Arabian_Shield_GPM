import csv
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, ogr
from scipy import ndimage
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REMOTE_SENSING_DIR = PROJECT_ROOT / "07_remote_sensing"


def copy_alteration_rasters_to_root():
    alteration_files = [
        "alteration_index.tif",
        "iser_index.tif",
        "ccpi_index.tif",
    ]
    for filename in alteration_files:
        source = REMOTE_SENSING_DIR / filename
        destination = PROJECT_ROOT / filename
        if not destination.exists():
            if not source.exists():
                raise FileNotFoundError(f"Alteration raster not found: {source}")
            shutil.copy2(source, destination)


def find_raster(filename):
    candidate_paths = [
        PROJECT_ROOT / filename,
        REMOTE_SENSING_DIR / filename,
        SCRIPT_DIR / filename,
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find raster '{filename}' in the project root or 07_remote_sensing.")


def open_slope_raster():
    candidate_paths = [PROJECT_ROOT / "slope.tif", PROJECT_ROOT / "slope", SCRIPT_DIR / "slope.tif", SCRIPT_DIR / "slope"]
    for path in candidate_paths:
        if path.exists():
            dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
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
        return None

    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if raster is None:
        print(f"Notice: could not open raster '{raster_path}' for validation; skipping mineral occurrence validation.")
        return None

    vector_data = ogr.Open(vector_path)
    if vector_data is None:
        print(f"Notice: could not open vector '{vector_path}' for validation; skipping mineral occurrence validation.")
        return None

    band = raster.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float32)
    geotransform = raster.GetGeoTransform()
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    records = []
    layer = vector_data.GetLayer()
    if layer is None:
        print(f"Notice: vector layer in '{vector_path}' is empty; skipping mineral occurrence validation.")
        return None

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
        return []

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
    return values


def evaluate_roc_auc(raster_path, positive_values, geotransform, projection):
    if positive_values is None or len(positive_values) == 0:
        print("Notice: no positive occurrence values available for ROC/AUC evaluation; skipping model assessment.")
        return None

    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if raster is None:
        print(f"Notice: could not open '{raster_path}' for ROC/AUC evaluation; skipping model assessment.")
        return None

    data = raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    valid_data = data[np.isfinite(data)]
    if valid_data.size == 0:
        print("Notice: raster contains no valid pixels for ROC/AUC evaluation; skipping model assessment.")
        return None

    n_pos = len(positive_values)
    rng = np.random.default_rng(42)
    rows, cols = data.shape
    row_idx = rng.integers(0, rows, size=n_pos)
    col_idx = rng.integers(0, cols, size=n_pos)
    negative_values = np.array([
        float(data[row_idx[i], col_idx[i]]) for i in range(n_pos)
    ], dtype=np.float32)

    positive_scores = np.asarray(positive_values, dtype=np.float32)
    negative_scores = np.asarray(negative_values, dtype=np.float32)
    labels = np.concatenate([np.ones(len(positive_scores), dtype=int), np.zeros(len(negative_scores), dtype=int)])
    scores = np.concatenate([positive_scores, negative_scores])

    sorted_indices = np.argsort(scores)
    scores_sorted = scores[sorted_indices]
    labels_sorted = labels[sorted_indices]

    thresholds = np.unique(scores_sorted)
    tpr = []
    fpr = []
    for threshold in thresholds:
        predicted_positive = scores >= threshold
        tp = np.sum(predicted_positive & (labels == 1))
        fp = np.sum(predicted_positive & (labels == 0))
        fn = np.sum((~predicted_positive) & (labels == 1))
        tn = np.sum((~predicted_positive) & (labels == 0))
        tpr_value = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr.append(tpr_value)
        fpr.append(fpr_value)

    if len(tpr) > 1:
        tpr = np.asarray(tpr)
        fpr = np.asarray(fpr)
        auc_score = float(np.trapz(tpr, fpr))
    else:
        auc_score = 0.5

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})", color="darkorange", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="navy", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Fuzzy Structural Overlay ROC Evaluation")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("roc_curve_evaluation.png", dpi=300)
    plt.close(fig)

    positive_mean = float(np.mean(positive_scores)) if positive_scores.size > 0 else 0.0
    negative_mean = float(np.mean(negative_scores)) if negative_scores.size > 0 else 0.0
    summary_lines = [
        f"AUC: {auc_score:.6f}",
        f"Mean prospectivity at gold occurrences: {positive_mean:.6f}",
        f"Mean prospectivity at random background points: {negative_mean:.6f}",
        f"Median prospectivity at gold occurrences: {float(np.median(positive_scores)):.6f}",
        f"Median prospectivity at background points: {float(np.median(negative_scores)):.6f}",
        f"Positive samples: {len(positive_scores)}",
        f"Background samples: {len(negative_scores)}",
    ]
    with open("model_performance_summary.txt", "w", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(summary_lines) + "\n")

    print(f"ROC/AUC evaluation complete. AUC={auc_score:.6f}; summary saved to model_performance_summary.txt")
    return auc_score


def load_feature_stack(feature_names):
    arrays = []
    valid_mask = None
    reference_geotransform = None
    reference_projection = None
    reference_shape = None

    for filename in feature_names:
        raster_path = find_raster(filename)
        dataset = gdal.Open(str(raster_path), gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"Could not open feature raster: {raster_path}")

        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray().astype(np.float32)
        if reference_shape is None:
            reference_shape = array.shape
            reference_geotransform = dataset.GetGeoTransform()
            reference_projection = dataset.GetProjection()
        elif array.shape != reference_shape:
            raise ValueError(f"Feature raster dimensions do not match: {raster_path}")

        band_nodata = band.GetNoDataValue()
        band_valid = np.isfinite(array)
        if band_nodata is not None:
            band_valid &= array != band_nodata
        valid_mask = band_valid if valid_mask is None else valid_mask & band_valid
        arrays.append(array)
        dataset = None

    return (
        np.stack(arrays, axis=-1),
        valid_mask,
        reference_geotransform,
        reference_projection,
    )


def get_gold_occurrence_pixels(vector_path, geotransform, raster_shape, valid_mask):
    vector_data = ogr.Open(str(vector_path))
    if vector_data is None:
        raise FileNotFoundError(f"Could not open occurrence vector: {vector_path}")

    layer = vector_data.GetLayer()
    if layer is None:
        raise ValueError(f"Occurrence vector has no readable layer: {vector_path}")

    rows, cols = raster_shape
    pixels = set()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue

        if geom.GetGeometryType() == ogr.wkbPoint:
            point_geometries = [geom]
        elif geom.GetGeometryType() == ogr.wkbMultiPoint:
            point_geometries = [
                geom.GetGeometryRef(index)
                for index in range(geom.GetGeometryCount())
            ]
        else:
            point_geometries = []

        for point in point_geometries:
            if point is None:
                continue
            x, y, _ = point.GetPoint()
            col = int((x - geotransform[0]) / geotransform[1])
            row = int((y - geotransform[3]) / geotransform[5])
            if 0 <= row < rows and 0 <= col < cols and valid_mask[row, col]:
                pixels.add(row * cols + col)

    vector_data = None
    return np.asarray(sorted(pixels), dtype=np.int64)


def run_random_forest(feature_stack, valid_mask, geotransform, projection, occurrence_path, feature_names):
    rows, cols, feature_count = feature_stack.shape
    positive_indices = get_gold_occurrence_pixels(
        occurrence_path,
        geotransform,
        (rows, cols),
        valid_mask,
    )
    if positive_indices.size == 0:
        raise ValueError("No valid gold occurrence pixels were found for Random Forest training.")

    valid_indices = np.flatnonzero(valid_mask.ravel())
    negative_candidates = np.setdiff1d(valid_indices, positive_indices, assume_unique=False)
    if negative_candidates.size == 0:
        raise ValueError("No valid background pixels are available for Random Forest training.")

    rng = np.random.default_rng(42)
    negative_count = min(positive_indices.size, negative_candidates.size)
    negative_indices = rng.choice(negative_candidates, size=negative_count, replace=False)
    sample_indices = np.concatenate([positive_indices, negative_indices])
    labels = np.concatenate([
        np.ones(positive_indices.size, dtype=np.uint8),
        np.zeros(negative_indices.size, dtype=np.uint8),
    ])

    max_training_samples = 100000
    if sample_indices.size > max_training_samples:
        selected = rng.choice(sample_indices.size, size=max_training_samples, replace=False)
        sample_indices = sample_indices[selected]
        labels = labels[selected]

    training_features = feature_stack.reshape(-1, feature_count)[sample_indices]
    model = RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(training_features, labels)

    probability_map = np.full((rows, cols), -9999.0, dtype=np.float32)
    valid_features = feature_stack.reshape(-1, feature_count)[valid_indices]
    probability_map.ravel()[valid_indices] = model.predict_proba(valid_features)[:, 1]
    write_geotiff(
        PROJECT_ROOT / "random_forest_prospectivity.tif",
        probability_map,
        geotransform,
        projection,
        gdal.GDT_Float32,
    )

    importance_order = np.argsort(model.feature_importances_)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.barh(
        np.asarray(feature_names)[importance_order],
        model.feature_importances_[importance_order],
        color="steelblue",
    )
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest Gini Feature Importance")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "rf_feature_importance.png", dpi=300)
    plt.close(fig)

    oob_scores = model.oob_decision_function_[:, 1]
    finite_oob = np.isfinite(oob_scores)
    if np.count_nonzero(finite_oob) > 1 and np.unique(labels[finite_oob]).size == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(labels[finite_oob], oob_scores[finite_oob])
        roc_auc = auc(false_positive_rate, true_positive_rate)
    else:
        false_positive_rate = np.array([0.0, 1.0])
        true_positive_rate = np.array([0.0, 1.0])
        roc_auc = 0.5

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    ax.plot(false_positive_rate, true_positive_rate, label=f"OOB ROC (AUC = {roc_auc:.3f})", color="darkorange")
    ax.plot([0, 1], [0, 1], "--", color="navy")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Random Forest ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "rf_roc_curve.png", dpi=300)
    plt.close(fig)

    print(f"Random Forest OOB score: {model.oob_score_:.4f}")
    print(f"Random Forest outputs saved to {PROJECT_ROOT}")


def main():
    copy_alteration_rasters_to_root()
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
        PROJECT_ROOT / "lineament_distance.tif",
        distance_from_edges.astype(np.float32),
        geotransform,
        projection,
        gdal.GDT_Float32,
    )
    write_geotiff(
        PROJECT_ROOT / "lineament_density.tif",
        lineament_density.astype(np.float32),
        geotransform,
        projection,
        gdal.GDT_Float32,
    )
    write_geotiff(
        PROJECT_ROOT / "fuzzy_lineament_density.tif",
        fuzzy_density.astype(np.float32),
        geotransform,
        projection,
        gdal.GDT_Float32,
    )
    write_geotiff(
        PROJECT_ROOT / "fuzzy_lineament_distance.tif",
        fuzzy_distance.astype(np.float32),
        geotransform,
        projection,
        gdal.GDT_Float32,
    )
    write_geotiff(
        PROJECT_ROOT / "fuzzy_structural_overlay.tif",
        fuzzy_gamma.astype(np.float32),
        geotransform,
        projection,
        gdal.GDT_Float32,
    )

    validate_gold_prospectivity(
        str(PROJECT_ROOT / "fuzzy_structural_overlay.tif"),
        str(PROJECT_ROOT / "mods_gold_occurrences.geojson"),
    )

    feature_names = [
        "lineament_density.tif",
        "lineament_distance.tif",
        "alteration_index.tif",
        "iser_index.tif",
        "ccpi_index.tif",
    ]
    feature_stack, valid_mask, feature_geotransform, feature_projection = load_feature_stack(feature_names)
    run_random_forest(
        feature_stack,
        valid_mask,
        feature_geotransform,
        feature_projection,
        PROJECT_ROOT / "mods_gold_occurrences.geojson",
        feature_names,
    )

    dataset = None


if __name__ == "__main__":
    main()
