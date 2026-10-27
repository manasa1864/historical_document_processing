
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def binarize(image):
    # Otsu thresholding; dark pixels are treated as foreground ink.
    _, result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return result == 0



def drd_score(prediction, reference):
    """
    Calculate Distance Reciprocal Distortion (DRD) for binary images.

    prediction and reference are Boolean arrays:
    True = foreground ink, False = background.
    """
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)

    if pred.shape != ref.shape:
        return np.nan

    # DRD is based on pixels that differ from the reference.
    errors = pred != ref

    if not np.any(errors):
        return 0.0

    # Normalized 5x5 reciprocal-distance weights.
    weights = np.zeros((5, 5), dtype=np.float64)

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            distance = np.sqrt(dx * dx + dy * dy)
            if distance != 0:
                weights[dy + 2, dx + 2] = 1.0 / distance

    weights /= weights.sum()

    # Pad the reference image to handle boundary pixels.
    padded_ref = np.pad(
        ref.astype(np.uint8),
        ((2, 2), (2, 2)),
        mode="constant",
        constant_values=0,
    )

    # Sum weighted neighborhood distortion at every error pixel.
    error_y, error_x = np.where(errors)
    distorted_center = pred[error_y, error_x].astype(np.uint8)
    total_distortion = 0.0

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            weight = weights[dy + 2, dx + 2]

            if weight == 0:
                continue

            neighborhood = padded_ref[
                error_y + 2 + dy,
                error_x + 2 + dx,
            ]

            total_distortion += weight * np.abs(
                neighborhood.astype(np.int16)
                - distorted_center.astype(np.int16)
            ).sum()

    # Count non-uniform 8x8 blocks in the reference.
    height, width = ref.shape
    nubn = 0

    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block = ref[y:y + 8, x:x + 8]
            if np.any(block) and not np.all(block):
                nubn += 1

    # DRD is undefined when the reference has no non-uniform blocks.
    if nubn == 0:
        return np.nan

    return float(total_distortion / nubn)


def binary_metrics(prediction, reference):
    pred = binarize(prediction)
    ref = binarize(reference)

    if pred.shape != ref.shape:
        return {
            "precision": np.nan,
            "recall": np.nan,
            "fm": np.nan,
            "drd": np.nan,
            "metric_note": "Image dimensions differ; align images first",
        }

    tp = np.logical_and(pred, ref).sum()
    fp = np.logical_and(pred, ~ref).sum()
    fn = np.logical_and(~pred, ref).sum()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    fm = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "fm": float(fm),
        "drd": drd_score(pred, ref),
        "metric_note": "",
    }





def brisque_score(image):
    model_path = (
        Path(__file__).parent
        / "models"
        / "brisque_model_live.yml"
    )
    range_path = (
        Path(__file__).parent
        / "models"
        / "brisque_range_live.yml"
    )

    if not hasattr(cv2, "quality"):
        raise RuntimeError("OpenCV quality module is unavailable")

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing BRISQUE model: {model_path}")

    if not range_path.is_file():
        raise FileNotFoundError(f"Missing BRISQUE range: {range_path}")

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    scores = cv2.quality.QualityBRISQUE_compute(
        color_image,
        str(model_path),
        str(range_path),
    )

    return float(scores[0])



def find_images(folder):
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    return {
        path.stem.lower(): path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def evaluate(restored_dir, reference_dir=None, output_csv="evaluation_results.csv"):
    restored_images = find_images(restored_dir)
    references = find_images(reference_dir) if reference_dir else {}

    if not restored_images:
        raise ValueError(f"No images found in {restored_dir}")

    rows = []

    for name, restored_path in sorted(restored_images.items()):
        try:
            restored = load_image(restored_path)
            reference_path = references.get(name)
            reference = load_image(reference_path) if reference_path else None

            row = {
                "image": restored_path.name,
                "restored_path": str(restored_path),
                "reference_path": str(reference_path) if reference_path else "",
                "precision": np.nan,
                "recall": np.nan,
                "fm": np.nan,
                "drd": np.nan,
                "psnr": np.nan,
                "ssim": np.nan,
                "brisque": brisque_score(restored),
                "status": "OK",
                "note": "",
            }

            
            if reference is not None:
                if restored.shape != reference.shape:
                    reference_for_metrics = cv2.resize(
                        reference,
                        (restored.shape[1], restored.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                    row["note"] = (
                        "Reference resized in memory to restored-image "
                        "dimensions; verify alignment and reference provenance"
                    )
                else:
                    reference_for_metrics = reference
                    row["note"] = ""

                row.update(binary_metrics(restored, reference_for_metrics))

                row["psnr"] = peak_signal_noise_ratio(
                    reference_for_metrics,
                    restored,
                    data_range=255,
                )

                row["ssim"] = structural_similarity(
                    reference_for_metrics,
                    restored,
                    data_range=255,
                )

                row["note"] = "; ".join(
                    part for part in [
                        row.get("note", ""),
                        row.pop("metric_note", ""),
                    ]
                    if part
                )
            else:
                row["note"] = "No reference image with matching filename stem"


            rows.append(row)

        except Exception as exc:
            rows.append({
                "image": restored_path.name,
                "restored_path": str(restored_path),
                "status": "ERROR",
                "note": str(exc),
            })

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)

    print(f"Evaluated {len(rows)} images")
    print(f"Results saved to: {output.resolve()}")
    print("Metrics without required reference data will be blank.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate restored historical document images"
    )
    parser.add_argument("--restored", required=True,
                        help="Folder containing restored images")
    parser.add_argument("--reference", default=None,
                        help="Folder containing matching reference images")
    parser.add_argument("--output", default="evaluation/results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    evaluate(args.restored, args.reference, args.output)
