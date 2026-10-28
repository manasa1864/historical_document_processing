"""Plot-based no-reference comparison of original and restored historical documents.

Creates per-dataset charts using matplotlib and seaborn:
- Metric mean comparison: original vs restored for BRISQUE, background std, ink/background contrast,
  estimated ink coverage, stroke-continuity proxy.
- Distribution plots for each metric.
- Representative image panels: original, restored, and absolute pixel-difference visualization.
- Grayscale intensity distributions for original vs restored.

These are diagnostic comparisons only. Difference images and heuristic metrics do not prove
restoration accuracy in the absence of ground-truth images.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METRICS = [
    ("brisque", "BRISQUE score (lower is generally better)"),
    ("background_std", "Background intensity standard deviation (lower = more uniform)"),
    ("ink_background_contrast", "Estimated ink-background contrast (higher usually = stronger contrast)"),
    ("estimated_ink_coverage_pct", "Estimated ink coverage (%)"),
    ("stroke_continuity_proxy", "Stroke continuity proxy (heuristic)"),
]


def load_gray(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"Could not read image: {path}")
    return im


def get_brisque(gray: np.ndarray, model_path: Path, range_path: Path) -> float:
    if not hasattr(cv2, "quality") or not hasattr(cv2.quality, "QualityBRISQUE_compute"):
        return float("nan")
    if not model_path.is_file() or not range_path.is_file():
        return float("nan")
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    score = float(cv2.quality.QualityBRISQUE_compute(bgr, str(model_path), str(range_path))[0])
    return score if math.isfinite(score) else float("nan")


def masks(gray: np.ndarray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = binary > 0
    bg = ~ink
    return ink, bg


def compute_metrics(gray: np.ndarray, model_path: Path, range_path: Path) -> dict:
    ink, bg = masks(gray)
    ink_pixels = gray[ink]
    bg_pixels = gray[bg]
    if bg_pixels.size >= 20:
        bright = bg_pixels[bg_pixels >= np.percentile(bg_pixels, 50)]
        bg_std = float(np.std(bright.astype(np.float32))) if bright.size >= 20 else float("nan")
    else:
        bg_std = float("nan")
    contrast = (
        float(np.median(bg_pixels) - np.median(ink_pixels))
        if ink_pixels.size >= 10 and bg_pixels.size >= 10 else float("nan")
    )
    coverage = float(np.mean(ink) * 100.0)
    if np.any(ink):
        n = cv2.filter2D(ink.astype(np.uint8), -1, np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT)
        continuity = float(np.mean(n[ink] > 1))
    else:
        continuity = float("nan")
    return {
        "brisque": get_brisque(gray, model_path, range_path),
        "background_std": bg_std,
        "ink_background_contrast": contrast,
        "estimated_ink_coverage_pct": coverage,
        "stroke_continuity_proxy": continuity,
    }


def find_images(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        return {}
    return {
        p.stem.lower(): p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def resize_for_display(im: np.ndarray, max_dim: int = 1000) -> np.ndarray:
    h, w = im.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1:
        im = cv2.resize(im, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA)
    return im


def save_pair_panel(original: np.ndarray, restored: np.ndarray, out: Path, title: str):
    # Align dimensions for visualization only; no saved source images are modified.
    if original.shape != restored.shape:
        restored_aligned = cv2.resize(restored, (original.shape[1], original.shape[0]),
                                      interpolation=cv2.INTER_AREA)
    else:
        restored_aligned = restored
    original_small = resize_for_display(original)
    restored_small = cv2.resize(restored_aligned, (original_small.shape[1], original_small.shape[0]),
                                interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(original_small, restored_small)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(original_small, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[1].imshow(restored_small, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Restored")
    axes[2].imshow(diff, cmap="inferno", vmin=0, vmax=255)
    axes[2].set_title("Absolute pixel difference (visual aid)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_intensity_plot(original: np.ndarray, restored: np.ndarray, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(original.ravel()[::max(1, original.size // 200000)],
                 bins=64, stat="density", element="step", fill=False,
                 label="Original", ax=ax)
    sns.histplot(restored.ravel()[::max(1, restored.size // 200000)],
                 bins=64, stat="density", element="step", fill=False,
                 label="Restored", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Grayscale intensity (0=black, 255=white)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metric_plots(df: pd.DataFrame, out_dir: Path, dataset: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Long-form data for paired metric comparison.
    long_rows = []
    for _, row in df.iterrows():
        for metric, _ in METRICS:
            for stage in ("original", "restored"):
                value = row.get(f"{metric}_{stage}", np.nan)
                if pd.notna(value) and np.isfinite(float(value)):
                    long_rows.append({
                        "filename": row["filename"], "metric": metric,
                        "stage": stage.capitalize(), "value": float(value)
                    })
    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return

    # One plot per metric; different units are not mixed on a single axis.
    for metric, label in METRICS:
        subset = long_df[long_df["metric"] == metric]
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.boxplot(data=subset, x="stage", y="value", ax=axes[0])
        sns.stripplot(data=subset, x="stage", y="value", ax=axes[0],
                      color="black", alpha=0.25, size=2)
        axes[0].set_title("Original vs restored")
        axes[0].set_xlabel("")
        axes[0].set_ylabel(label)
        sns.histplot(data=subset, x="value", hue="stage", element="step",
                     fill=False, common_norm=False, ax=axes[1])
        axes[1].set_title("Metric distribution")
        axes[1].set_xlabel(label)
        axes[1].set_ylabel("Image count")
        fig.suptitle(f"{dataset}: {label}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Summary bar chart with means per stage, one facet per metric, each own axis.
    means = long_df.groupby(["metric", "stage"], as_index=False)["value"].mean()
    for metric, label in METRICS:
        sub = means[means["metric"] == metric]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=sub, x="stage", y="value", ax=ax, errorbar=None)
        ax.set_title(f"{dataset}: mean {label}")
        ax.set_xlabel("")
        ax.set_ylabel(label)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_mean_bar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="evaluation/visual_analysis")
    parser.add_argument("--model", default="evaluation/models/brisque_model_live.yml")
    parser.add_argument("--range", dest="range_file", default="evaluation/models/brisque_range_live.yml")
    parser.add_argument("--samples-per-dataset", type=int, default=8,
                        help="Number of representative original/restored image panels per dataset")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output = (root / args.output_dir).resolve()
    model = (root / args.model).resolve()
    range_file = (root / args.range_file).resolve()
    output.mkdir(parents=True, exist_ok=True)

    datasets = {
        "main": ("dataset", "dataset_restored"),
        "dibco_2017": ("dibco_2017", "dibco_2017_restored"),
        "dibco_2018": ("dibco_2018", "dibco_2018_restored"),
        "kaggle": ("kaggle_input", "kaggle_restored"),
        "persian": ("persian", "persian_restored"),
    }

    all_rows = []
    for dataset, (orig_rel, rest_rel) in datasets.items():
        originals = find_images(root / orig_rel)
        restored = find_images(root / rest_rel)
        if not restored:
            print(f"[SKIP] {dataset}: no restored images in {root / rest_rel}")
            continue
        matched = sorted(set(originals) & set(restored))
        print(f"[RUN] {dataset}: {len(restored)} restored, {len(matched)} matching originals")
        dataset_rows = []
        for key in sorted(restored):
            rp = restored[key]
            try:
                rimg = load_gray(rp)
                rm = compute_metrics(rimg, model, range_file)
                row = {"dataset": dataset, "filename": rp.name, "restored_path": str(rp)}
                row.update({f"{k}_restored": v for k, v in rm.items()})
                op = originals.get(key)
                row["original_path"] = str(op) if op else ""
                if op:
                    oimg = load_gray(op)
                    om = compute_metrics(oimg, model, range_file)
                    row.update({f"{k}_original": v for k, v in om.items()})
                    row["visual_pair_available"] = True
                else:
                    row.update({f"{k}_original": np.nan for k, _ in METRICS})
                    row["visual_pair_available"] = False
                dataset_rows.append(row)
            except Exception as exc:
                dataset_rows.append({
                    "dataset": dataset, "filename": rp.name, "restored_path": str(rp),
                    "status": "ERROR", "note": str(exc)
                })

        df = pd.DataFrame(dataset_rows)
        dataset_dir = output / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(dataset_dir / f"{dataset}_original_restored_metrics.csv",
                  index=False, encoding="utf-8-sig")
        successful = df[df.get("visual_pair_available", False) == True] if "visual_pair_available" in df else df.iloc[0:0]
        if not successful.empty:
            save_metric_plots(successful, dataset_dir / "charts", dataset)
            panel_dir = dataset_dir / "image_panels"
            for _, row in successful.head(max(0, args.samples_per_dataset)).iterrows():
                try:
                    oi = load_gray(Path(row["original_path"]))
                    ri = load_gray(Path(row["restored_path"]))
                    save_pair_panel(oi, ri, panel_dir / f"{safe_filename(Path(row['filename']).stem)}_panel.png",
                                    f"{dataset} — {row['filename']}")
                    save_intensity_plot(oi, ri, panel_dir / f"{safe_filename(Path(row['filename']).stem)}_intensity.png",
                                        f"{dataset} — grayscale intensity distribution")
                except Exception as exc:
                    print(f"[WARN] Could not plot {row['filename']}: {exc}")
        all_rows.extend(dataset_rows)

    if not all_rows:
        raise SystemExit("No restored images found. Check dataset folder names.")
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(output / "all_datasets_original_restored_metrics.csv",
                  index=False, encoding="utf-8-sig")

    # Overall dataset comparison charts: metric means across datasets, original vs restored.
    summary = []
    for dataset, group in all_df.groupby("dataset"):
        r = {"dataset": dataset, "images": len(group)}
        for metric, _ in METRICS:
            for stage in ("original", "restored"):
                col = f"{metric}_{stage}"
                if col in group:
                    r[f"{metric}_{stage}_mean"] = pd.to_numeric(group[col], errors="coerce").mean()
        summary.append(r)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output / "all_datasets_summary.csv", index=False, encoding="utf-8-sig")

    for metric, label in METRICS:
        columns = [f"{metric}_original_mean", f"{metric}_restored_mean"]
        if not all(c in summary_df.columns for c in columns):
            continue
        plot_df = summary_df.melt(id_vars="dataset", value_vars=columns,
                                  var_name="stage", value_name="mean_value")
        plot_df["stage"] = plot_df["stage"].map({
            columns[0]: "Original", columns[1]: "Restored"
        })
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=plot_df, x="dataset", y="mean_value", hue="stage", ax=ax, errorbar=None)
        ax.set_title(f"All datasets: original vs restored — {label}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output / f"all_datasets_{metric}_comparison.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    print("\nFinished visual analysis.")
    print(f"Outputs: {output}")
    print("See each dataset folder for charts, image panels, and CSV results.")
    print("Caution: difference panels are visual aids, not ground-truth error maps.")
    print("Ink coverage and stroke continuity are heuristics, not proof of ink preservation.")


if __name__ == "__main__":
    main()
