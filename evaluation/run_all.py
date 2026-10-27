
from pathlib import Path

import pandas as pd

from evaluate_images import evaluate as evaluate_images
from evaluate_ocr import evaluate_ocr


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "evaluation"

# Five dataset pairs in the existing repository.
DATASETS = {
    "main": ("dataset", "dataset_restored"),
    "dibco_2017": ("dibco_2017", "dibco_2017_restored"),
    "dibco_2018": ("dibco_2018", "dibco_2018_restored"),
    "kaggle": ("kaggle_input", "kaggle_restored"),
    "persian": ("persian", "persian_restored"),
}


def main():
    image_summaries = []
    ocr_summaries = []

    for name, (reference_folder, restored_folder) in DATASETS.items():
        reference_dir = ROOT / reference_folder
        restored_dir = ROOT / restored_folder

        if not reference_dir.is_dir() or not restored_dir.is_dir():
            print(f"[SKIP] {name}: input or restored folder is missing")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {name}")
        print(f"Reference/input: {reference_dir}")
        print(f"Restored:        {restored_dir}")

        image_csv = OUTPUT / f"{name}_image_metrics.csv"
        ocr_csv = OUTPUT / f"{name}_ocr_results.csv"

        # Reference-based and no-reference image metrics.
        evaluate_images(
            restored_dir=str(restored_dir),
            reference_dir=str(reference_dir),
            output_csv=str(image_csv),
        )

        # OCR extraction. CER/WER require a reference transcription CSV.
        evaluate_ocr(
            image_dir=str(restored_dir),
            output_csv=str(ocr_csv),
            reference_csv=None,
            language="eng",
        )

        image_df = pd.read_csv(image_csv)
        image_df.insert(0, "dataset", name)
        image_summaries.append(image_df)

        ocr_df = pd.read_csv(ocr_csv)
        ocr_df.insert(0, "dataset", name)
        ocr_summaries.append(ocr_df)

    if image_summaries:
        combined_images = pd.concat(image_summaries, ignore_index=True)
        combined_images.to_csv(
            OUTPUT / "all_datasets_image_metrics.csv", index=False
        )

    if ocr_summaries:
        combined_ocr = pd.concat(ocr_summaries, ignore_index=True)
        combined_ocr.to_csv(
            OUTPUT / "all_datasets_ocr_results.csv", index=False
        )

    print(f"\nBatch evaluation finished. Results are in: {OUTPUT}")
    print("Review each CSV for missing references and metric errors.")


if __name__ == "__main__":
    main()
