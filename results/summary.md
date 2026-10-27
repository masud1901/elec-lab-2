# Ablation summary

## Part 0 - Reference

| Variant | Seeds | Weighted F1 | Macro F1 | Gesture macro F1 | Accuracy | Params | Best epoch |
|---|---|---|---|---|---|---|---|
| ref | 1 | 0.893 | 0.595 | 0.574 | 0.896 | 3,965,778 | 76.0 |

Values are mean ± std over seeds on the held-out test split.

## Part A - Is the LSTM what helps?

| Variant | Seeds | Weighted F1 | Macro F1 | Gesture macro F1 | Accuracy | Params | Best epoch |
|---|---|---|---|---|---|---|---|
| A_cnn | 1 | 0.873 | 0.474 | 0.446 | 0.882 | 7,486,546 | 33.0 |
| A_gru | 1 | 0.892 | 0.624 | 0.605 | 0.892 | 2,990,418 | 62.0 |

Values are mean ± std over seeds on the held-out test split.

## Part B - Do the convolutions matter?

| Variant | Seeds | Weighted F1 | Macro F1 | Gesture macro F1 | Accuracy | Params | Best epoch |
|---|---|---|---|---|---|---|---|
| B_conv0 | 1 | 0.798 | 0.228 | 0.188 | 0.839 | 258,834 | 12.0 |
| B_conv1 | 1 | 0.881 | 0.542 | 0.518 | 0.882 | 3,904,146 | 39.0 |
| B_conv2 | 1 | 0.878 | 0.541 | 0.517 | 0.876 | 3,924,690 | 38.0 |

Values are mean ± std over seeds on the held-out test split.

## Part C - How long should the window be?

| Variant | Seeds | Weighted F1 | Macro F1 | Gesture macro F1 | Accuracy | Params | Best epoch |
|---|---|---|---|---|---|---|---|
| C_w20 | 1 | 0.892 | 0.596 | 0.575 | 0.891 | 3,965,778 | 48.0 |
| C_w42 | 1 | 0.873 | 0.536 | 0.513 | 0.872 | 3,965,778 | 28.0 |

Values are mean ± std over seeds on the held-out test split.

## Notes

- `ref` = 4 conv layers, LSTM, window 24 (0.8 s), dropout 0.5, weight decay 1e-4, augmentation.
- Test predictions are sample-wise (stride 1); train/val windows use stride 12.
- Weighted F1 includes Null; macro F1 averages all 18 classes; gesture macro F1 excludes Null.
