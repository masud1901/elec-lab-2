# Ablation summary

## Part 0 - Reference

| Variant | Seeds | Weighted F1 | Macro F1 | Gesture macro F1 | Accuracy | Params | Best epoch |
|---|---|---|---|---|---|---|---|
| ref | 1 | 0.893 | 0.595 | 0.574 | 0.896 | 3,965,778 | 76.0 |

Values are mean ± std over seeds on the held-out test split.

## Notes

- `ref` = 4 conv layers, LSTM, window 24 (0.8 s), dropout 0.5, weight decay 1e-4, augmentation.
- Test predictions are sample-wise (stride 1); train/val windows use stride 12.
- Weighted F1 includes Null; macro F1 averages all 18 classes; gesture macro F1 excludes Null.
