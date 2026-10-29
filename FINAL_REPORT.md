# Historical Document Restoration: Final Evaluation Report

All results compare the **restored** image with the **original** scan. Where a ground-truth (hand-made answer) image exists, the restored image is also compared with that.

---

## 1. Summary

| Question | Answer |
|---|---|
| Does restoration make pages cleaner? | **Yes, on every dataset and almost every image.** Background noise goes to zero and contrast rises 2 to 4 times (section 5). |
| Is it closer to the ground truth than the original scan? | **Yes on average, on both DIBCO sets.** F-measure 0.52 → 0.80 (DIBCO 2018) and 0.78 → 0.82 (DIBCO 2017) against a plain threshold of the original (section 4). |
| Is it always better? | **No.** On DIBCO 2017 it wins on only about half of the individual images. The gain is largest on pages with bleed-through and zero or negative on clean pages. |
| Does it keep all the writing? | **Not always.** Very faint ink is sometimes lost, and dark page edges are sometimes kept (section 7). |
| Is BRISQUE a valid measure here? | **No.** It scores the ground truth worst (section 6). It is not used to judge the result. |

**Method in one line:** classical image processing only (no machine learning). Each page's thresholds are measured from the page itself. The output keeps the exact size of the input (no cropping or resizing).

---

## 2. What was evaluated

| Dataset | Images | Ground truth? | Restored folder used |
|---|---|---|---|
| DIBCO 2018 | 10 | Yes | `dibco_2018_restored/` |
| DIBCO 2017 | 20 | Yes | `dibco_2017_restored/` |
| Letters | 150 | No | Fresh run of the current pipeline (see the note below) |
| Persian Heritage | 15 | No | `persian_restored/` |
| Kaggle *Denoising Dirty Documents*, test split | 72 | No (kept hidden by the competition) | `dataset_restored_kaggle/` |

Not evaluated: the Kaggle **train** split (144 images). Its answer key (`train_cleaned.zip`) is on this machine but was not used.

**Note on the Letters folder.** The `dataset_restored/` folder in the repository was produced by an **earlier version** of the pipeline (committed 15 Oct). That version shrinks large images and crops 3% off every edge, so its images are a different size from the originals. For this report the Letters results come from a **fresh run of the current pipeline**, saved outside the repository. To make the repository match, re-run the Letters cells of `MAIN.ipynb`.

---

## 3. How the comparisons were made

### 3.1 With a ground truth (DIBCO 2017 and 2018)

The restored image is a grayscale picture, so it was turned into black and white with **Otsu's threshold** before being compared with the ground truth. The same step was applied to the original scan to give a **baseline**: "what you get from just thresholding the original".

| Measure | Meaning | Better |
|---|---|---|
| **F-measure** | Balance of precision and recall (1.0 is perfect) | Higher |
| **Precision** | Of the pixels kept as ink, how many really are ink | Higher |
| **Recall** | Of the real ink pixels, how many were kept | Higher |
| **PSNR** (dB) | How closely the pixels match the ground truth | Higher |
| **DRD** | Distortion that weights errors by how visible they are (errors next to text count more) | Lower (0 is perfect) |
| **Weighted recall** (2018 only) | Recall that gives more weight to the centre-line of strokes, using the supplied `RWeights` files | Higher |

These measures were calculated with my own code from the standard definitions, not the official DIBCO tool, so small differences from published tables are possible. The pseudo-precision measure was not computed, because its official formula could not be checked.

### 3.2 Without a ground truth

Only measures of cleanliness can be calculated. They use the same definitions as the project's existing evaluation script:

| Measure | Meaning | Better |
|---|---|---|
| **Background std** | How much the paper background varies | Lower |
| **Ink-to-paper contrast** | Brightness gap between paper and ink | Higher |
| **% pure-white pixels** | Share of the page that is clean white | Higher |
| **Ink coverage %** | Share of the page that is ink | Information only |

---

## 4. Results against the ground truth

### 4.1 DIBCO 2018 (10 images)

| Measure | Ground truth | **Restored** | Baseline: threshold of original |
|---|---|---|---|
| F-measure | 1.000 | **0.797** | 0.515 |
| Precision | 1.000 | **0.775** | 0.425 |
| Recall | 1.000 | **0.843** | 0.788 |
| Weighted recall | 1.000 | 0.876 | **0.903** |
| PSNR (dB) | not applicable | **15.99** | 9.79 |
| DRD (lower is better) | 0 | **10.54** | 58.77 |

The restored images beat the baseline on F-measure for **9 of 10** images (image 10 is the exception) and on DRD for **10 of 10**. The baseline has slightly higher weighted recall because it keeps more of everything, including a lot of junk (its precision is 0.43).

Per image, F-measure (restored versus baseline):

| Image | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Restored | 0.930 | 0.706 | 0.942 | 0.752 | 0.514 | 0.851 | 0.852 | 0.857 | 0.907 | 0.655 |
| Baseline | 0.428 | 0.158 | 0.831 | 0.240 | 0.267 | 0.660 | 0.564 | 0.812 | 0.445 | 0.740 |

### 4.2 DIBCO 2017 (20 images)

| Measure | Ground truth | **Restored** | Baseline: threshold of original |
|---|---|---|---|
| F-measure | 1.000 | **0.817** | 0.781 |
| Precision | 1.000 | **0.751** | 0.719 |
| Recall | 1.000 | 0.930 | 0.927 |
| PSNR (dB) | not applicable | **14.60** | 13.98 |
| DRD (lower is better) | 0 | **9.00** | 14.87 |

(No weight files were supplied for 2017, so weighted recall is not reported.)

The average is better, but the picture per image is mixed. Restored beat the baseline on F-measure for **10 of 20** images, and on PSNR and DRD for **9 of 20**.

| Image | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Restored | 0.712 | 0.906 | 0.773 | 0.755 | 0.737 | 0.904 | 0.904 | 0.846 | 0.926 | 0.916 |
| Baseline | 0.531 | 0.925 | 0.503 | 0.377 | 0.541 | 0.880 | 0.876 | 0.880 | 0.842 | 0.874 |

| Image | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| Restored | 0.857 | 0.786 | 0.626 | 0.804 | 0.859 | 0.887 | 0.646 | 0.845 | 0.796 | 0.851 |
| Baseline | 0.935 | 0.812 | 0.673 | 0.847 | 0.884 | 0.880 | 0.842 | 0.853 | 0.823 | 0.850 |

### 4.3 What this means

- **Restoration helps most where the original is hardest to read.** On pages with strong show-through (2017 images 1, 3, 4 and 5) or stains and dark backgrounds (2018 images 1, 2, 4 and 9) a plain threshold produces a mess, because the unwanted marks become black. Restoration removes most of them. For example, 2017 image 4 goes from F 0.38 to 0.76.
- **On clean pages there is little to gain, and sometimes a loss.** Where a plain threshold already works (2017 images 2, 8, 11, 17), restoration is equal or worse. 2017 image 17 drops to F 0.65 because real ink was lost (recall 0.49).
- **The threshold choice changes the numbers.** With a fixed cut-off of 128 instead of Otsu, the F-measure is 0.843 on 2017 but 0.768 on 2018. No single choice is better on both, so Otsu was used for both.
- **Small samples.** There are 10 and 20 images. The averages are encouraging, not a final verdict.

---

## 5. Results without a ground truth

| Measure | Letters (150) | Persian (15) | Kaggle test (72) |
|---|---|---|---|
| Background std (lower is better) | 1.91 → **0.00** | 4.75 → **0.00** | 6.37 → **0.00** |
| Ink-to-paper contrast (higher) | 52.8 → **209.6** | 73.1 → **228.5** | 105.2 → **248.5** |
| % pure-white pixels (higher) | 0.0 → **97.0** | 0.0 → **88.9** | 1.9 → **83.4** |
| Ink coverage % (information only) | 11.0 → 2.7 | 11.2 → 9.9 | 25.9 → 16.7 |
| Images improved (first three rows) | 150 of 150 | 15 of 15 | 72 of 72 |

(Each cell reads "original → restored", as averages over the images in the dataset.)

**What this means.** Every page becomes cleaner and has much stronger contrast.

**What it does not mean.** These measures improve for almost any method that sets the background to white. They show the pages are **cleaner**, not that the writing is **more accurate** or fully preserved. The drop in ink coverage is mostly the removal of dirt and bleed-through, but without a ground truth it is not possible to tell whether some faint writing was removed too.

(A "stroke continuity" measure was also calculated. It is 1.00 before and after on every dataset, so it carries no information and is left out.)

### OCR as an indirect check on the writing

The repository's existing OCR evaluation (Tesseract) counts how many words can be read. It is a proxy, not an accuracy measure.

| Dataset | Words per image (original → restored) | OCR confidence (original → restored) | Images with more / same / fewer words |
|---|---|---|---|
| DIBCO 2018 | 24.3 → 34.2 | 28.3 → 28.7 | 6 / 0 / 4 |
| DIBCO 2017 | 64.5 → 62.2 | 36.4 → 37.3 | 12 / 1 / 7 |
| Kaggle (all 216) | 100.1 → 102.7 | 88.8 → 90.6 | 119 / 77 / 20 |
| Letters (150) | 152.1 → 190.1 | 34.0 → 35.4 | 111 / 1 / 38 |
| Persian | not evaluated (the Persian OCR language was not installed) | | |

**Readings.** OCR does not show a big gain: the confidence rises by about 1 point or less. On DIBCO 2017 the average word count goes **down** slightly and 7 of 20 images lose words. The Letters OCR result was run on the **earlier pipeline's** output (see section 2), so it does not describe the current pipeline. The report's own percentage column does not match the sign of the 2017 change, so the word counts above are quoted instead.

---

## 6. BRISQUE (not used)

BRISQUE scores one image at a time and needs no ground truth. It was trained on photographs. On the DIBCO 2018 images (lower is supposed to be better):

| | Original scans | Restored | Ground truth |
|---|---|---|---|
| Mean BRISQUE | 26.1 | 52.5 | 100.0 |

The ground truth, the cleanest possible result, gets the worst possible score (100), and the restored images score worse than the originals. So BRISQUE measures how "photo-like" an image is, which is the wrong question for black-and-white documents. It should not be used as evidence for or against the method. The averages match the earlier project evaluation (52.5 for DIBCO 2018).

---

## 7. Known weaknesses

Found by colouring the errors on the weakest DIBCO 2018 images (2, 5 and 10):

1. **Faint writing is dropped.** The ground truth counts faint handwriting as ink, but the pipeline treats very faint ink as bleed-through. In DIBCO 2018 image 10 almost all of the writing is lost (F 0.65), and in image 5 the lighter lower lines are lost. This is a deliberate trade-off: keeping faint ink would bring back bleed-through on other pages.
2. **Dark page edges are sometimes kept.** Irregular dark bands (a book edge or binding shadow along the top) in DIBCO 2018 images 2 and 5 are kept as ink, which lowers precision.
3. **Light grey ghost smudges can remain** on pages with heavy show-through (mostly on the Persian pages).
4. **Faint pencil notes in margins** are sometimes lost on the Letters pages.
5. **The dashed-line detector** is the most rule-of-thumb part of the method. It has been tested on the five datasets above only.

---

## 8. Conclusion

- Against the original scans, the restored images are **consistently cleaner** on every dataset tested.
- Against the **ground truth**, the restored images are **closer on average** than a plain threshold of the original, clearly so on DIBCO 2018 (F-measure 0.80 versus 0.52) and modestly so on DIBCO 2017 (0.82 versus 0.78).
- The gain comes mainly from **removing bleed-through**. On clean pages it is small or negative.
- The main losses are **faint ink** and **dark page edges**. Fixing these is the most promising way to improve the scores.
- The evidence is limited: 30 images with ground truth, one simple baseline, no comparison with other published methods. No claim is made about how it ranks against them.

---

## 9. Files

| File | Contents |
|---|---|
| `evaluation/gt_comparison_dibco2018.csv` | Per-image scores against the DIBCO 2018 ground truth |
| `evaluation/gt_comparison_dibco2017.csv` | Per-image scores against the DIBCO 2017 ground truth |
| `evaluation/noreference_comparison.csv` | No-ground-truth measures for Letters, Persian and Kaggle test |
| `evaluation/brisque_dibco2018_gt.csv` | BRISQUE for original, restored and ground truth (DIBCO 2018) |
| `ocr_results/` | The existing OCR evaluation |
| `MAIN.ipynb` | The pipeline and all dataset runs |
