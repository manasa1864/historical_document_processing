# OCR Evaluation Report: Historical Document Restoration

- Tesseract version: `5.5.3.20260724`
- OCR configuration: `--oem 3 --psm 6`
- OCR language: configured per dataset; see summary CSV and detailed CSV.
- Matching: explicit mappings, exact relative paths, then unique normalized filenames.
- Important: word count and Tesseract confidence are proxy metrics, not ground-truth accuracy.

## Dataset processing status

| dataset_pair                      | status                               |   raw_images |   restored_images |   matched_pairs |   successful_pairs |
|:----------------------------------|:-------------------------------------|-------------:|------------------:|----------------:|-------------------:|
| dataset vs dataset_restored       | EVALUATED                            |          150 |               150 |             150 |                150 |
| dibco_2017 vs dibco_2017_restored | EVALUATED                            |           20 |                20 |              20 |                 20 |
| dibco_2018 vs dibco_2018_restored | EVALUATED                            |           10 |                10 |              10 |                 10 |
| kaggle_input vs kaggle_restored   | EVALUATED                            |          216 |               216 |             216 |                216 |
| persian vs persian_restored       | SKIPPED - LANGUAGE fas NOT INSTALLED |           15 |                15 |               0 |                  0 |

## Evaluation results

| dataset_pair                      |   matched_images |   successful_pairs |   raw_mean_word_count |   restored_mean_word_count |   mean_word_count_change |   mean_word_count_change_percent |   raw_mean_ocr_confidence |   restored_mean_ocr_confidence |   images_with_more_words_after_restoration |   images_with_same_word_count |   images_with_fewer_words_after_restoration |   raw_total_ocr_seconds |   restored_total_ocr_seconds |
|:----------------------------------|-----------------:|-------------------:|----------------------:|---------------------------:|-------------------------:|---------------------------------:|--------------------------:|-------------------------------:|-------------------------------------------:|------------------------------:|--------------------------------------------:|------------------------:|-----------------------------:|
| dataset vs dataset_restored       |              150 |                150 |                152.08 |                     190.07 |                    37.99 |                            62.46 |                     34.01 |                          35.38 |                                        111 |                             1 |                                          38 |                 1011.45 |                       220.16 |
| dibco_2017 vs dibco_2017_restored |               20 |                 20 |                 64.45 |                      62.15 |                    -2.3  |                             7.52 |                     36.44 |                          37.28 |                                         12 |                             1 |                                           7 |                   34.78 |                        21.33 |
| dibco_2018 vs dibco_2018_restored |               10 |                 10 |                 24.3  |                      34.2  |                     9.9  |                            40.77 |                     28.31 |                          28.69 |                                          6 |                             0 |                                           4 |                   12.59 |                         7.98 |
| kaggle_input vs kaggle_restored   |              216 |                216 |                100.06 |                     102.72 |                     2.66 |                             2.99 |                     88.78 |                          90.57 |                                        119 |                            77 |                                          20 |                  173.05 |                       163.03 |

## Interpretation

### dataset vs dataset_restored
- Matched images: 150; successful OCR pairs: 150.
- Mean word-count change: 62.46% (change in detected quantity, not accuracy).
- Mean OCR confidence increased by 1.37 points.
- Images with more detected words: 111; same: 1; fewer: 38.

### dibco_2017 vs dibco_2017_restored
- Matched images: 20; successful OCR pairs: 20.
- Mean word-count change: 7.52% (change in detected quantity, not accuracy).
- Mean OCR confidence increased by 0.84 points.
- Images with more detected words: 12; same: 1; fewer: 7.

### dibco_2018 vs dibco_2018_restored
- Matched images: 10; successful OCR pairs: 10.
- Mean word-count change: 40.77% (change in detected quantity, not accuracy).
- Mean OCR confidence increased by 0.38 points.
- Images with more detected words: 6; same: 0; fewer: 4.

### kaggle_input vs kaggle_restored
- Matched images: 216; successful OCR pairs: 216.
- Mean word-count change: 2.99% (change in detected quantity, not accuracy).
- Mean OCR confidence increased by 1.79 points.
- Images with more detected words: 119; same: 77; fewer: 20.

## Limitations and recommendations

1. More detected words do not necessarily mean better OCR accuracy.
2. Tesseract confidence is not equivalent to correctness.
3. For stronger evaluation, compare OCR output against verified reference transcriptions using CER and WER.
4. Check `unmatched_files.csv` and `dataset_processing_status.csv` before drawing conclusions.
5. Differently named images must be paired using a verified mapping. Do not assume train_N corresponds to test_N without confirmation.
6. Persian OCR requires the Tesseract `fas` language data. Check `tesseract --list-langs`.
7. Review a representative sample of OCR text files manually.

## Generated files

- `summary_results.csv`: aggregate results for evaluated datasets.
- `dataset_processing_status.csv`: status and image counts for every configured dataset.
- `all_ocr_results_detailed.csv`: per-image OCR metrics and extracted text.
- `unmatched_files.csv`: unmatched, ambiguous, or invalid mappings.
- Per-dataset CSV files: individual evaluation results.
- Comparison graphs: word count and OCR confidence.