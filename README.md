# DeepConvLSTM on OPPORTUNITY — systematic ablation study

This repository contains the code, experiment configuration, results and report
for **Workshop 2: DeepConvLSTM**.  The task is a systematic ablation study of
DeepConvLSTM for wearable human-activity recognition on the OPPORTUNITY
dataset (113 sensor channels at 30 Hz, 18 classes: 17 kitchen gestures + Null).

The study reproduces the reference model (Part 0) and changes one design
choice at a time:

| Part | Question | Variants |
|---|---|---|
| 0 | Reproduce the reference | `ref` |
| A | Is the LSTM what helps? | `A_cnn`, `A_gru` |
| B | Do the convolutions matter? | `B_conv0`, `B_conv1`, `B_conv2` |
| C | How long should the window be? | `C_w20`, `C_w42`, `C_w60` |
| D | How much regularization is needed? | `D_no_dropout`, `D_no_wd`, `D_no_aug` |
| E | Which optimizer settings work best? | `E_lr1e-4`, `E_lr1e-2`, `E_warm1e-4`, `E_warm1e-2` |
| F | Does handling class imbalance help rare gestures? | `F_focal` |

## Data

[OPPORTUNITY Activity Recognition](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition)
(UCI Machine Learning Repository, ~300 MB).

```bash
python download_data.py
python preprocess.py --split paper
```

`preprocess.py` selects the 113 challenge sensor channels, interpolates missing
values, and min-max normalises each channel to `[0, 1]` using training-split
statistics only.  One `.npz` file per recording is written to
`data/processed/<split>/<part>/`; the raw data and processed arrays are not
committed.

Two protocols are implemented:

- **paper** (default): train on S1 ADL1–5 + Drill and S2/S3 ADL1–2 + Drill,
  validate on S2/S3 ADL3, test on S2/S3 ADL4–5.
- **course**: train on S1 Drill + ADL1–4 and S2/S3 Drill + ADL1–3, validate on
  S1 ADL5, test on S2/S3 ADL4–5 (the split used by the public reference
  implementation).  Select with `--split course`.

## Model

The reference matches the workshop's ~4M-parameter model: four 2-D
convolutions with kernel `(5, 1)` slide over time, the 113-channel axis is kept
as width, and the resulting `8 × 113 × 64` feature map is flattened to 7232
features per time step before two LSTM layers with 128 units.  See
`models.py` for the full definition and the `--model cnn`, `--rnn gru` and
`--n-conv` variants.

## Training protocol

- window 24 samples (0.8 s), train/val stride 12, **test stride 1** (sample-wise)
- label = gesture at the last sample of the window
- orthogonal initialisation, dropout 0.5, RMSProp (`lr=1e-3`, `alpha=0.9`),
  batch size 100, weight decay 1e-4, light augmentation (jitter/scale/shift)
- early stopping on validation macro F1 (patience 15, max 80 epochs)
- test evaluated once at the best validation epoch
- metrics: accuracy, macro/weighted F1 (18 classes), gesture-only macro F1,
  per-class F1, confusion matrix
- three seeds (42, 43, 44) per variant

The reference therefore includes weight decay and augmentation, so that Part D
can remove them; this is documented in the report.

## Reproduce

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

python download_data.py
python preprocess.py --split paper

# quick end-to-end check (tiny subset, 2 epochs)
SMOKE=1 scripts/run_all.sh core

# full study (resumable; already-finished runs are skipped)
scripts/run_all.sh core
scripts/run_all.sh regularization
scripts/run_all.sh optimizer
scripts/run_all.sh imbalance
scripts/run_all.sh long

python summarize.py
```

### Parallel runner

`scripts/run_matrix.py` runs the whole matrix (any seeds, any parts) with a
configurable number of concurrent trainings and a GPU-memory guard, so heavy
variants (the dense CNN baseline, the 2 s window) always run alone:

```bash
python scripts/run_matrix.py --seeds 42 43 44 --jobs 2
python scripts/run_matrix.py --seeds 42 --parts A B --list
```

### Running on Google Colab

`notebooks/run_ablation_colab.ipynb` runs the same matrix on a Colab GPU
(T4/A100), which is considerably faster than a laptop card for this
LSTM-heavy model. It clones the repository (public — no token needed), downloads
and preprocesses the dataset, runs the selected parts and seeds, rebuilds the
tables, figures and report, and finally exports the results as zip files.

```bash
# locally: rebuild the notebook (already committed) and open it in Colab
python scripts/make_colab_notebook.py

# after the Colab run: merge the downloaded archive into this checkout
scripts/import_colab_results.sh ~/Downloads/ablation_runs.zip
python summarize.py
```

Colab sessions are independent, so different team members can run different
parts (A, B, C, D, E, F) concurrently; `summarize.py` then merges everything.
A GitHub token is only required if the notebook should push the results back
to the repository (`PUSH_BACK = True`).

Individual runs:

```bash
python train.py --run-id ref_s42 --part 0 --seed 42
python train.py --run-id A_cnn_s42 --part A --model cnn --seed 42
python train.py --run-id C_w60_s42 --part C --window 60 --seed 42
python train.py --run-id E_warm1e-2_s42 --part E --lr 1e-2 --warmup-epochs 5 --seed 42
python train.py --run-id F_focal_s42 --part F --loss focal --seed 42
python evaluate.py --run runs/ref_s42 --split test
```

Each run writes `runs/<run-id>/` with `config.json`, `history.csv`, `best.pt`,
`per_class.csv`, `confusion_test.npy`, `test_predictions.npz`, `train.log` and
`run.json`.  `summarize.py` collects finished runs into `results/runs.csv`,
`results/per_class.csv`, `results/agg.csv`, `results/summary.md`,
`results/tables.tex` and `results/figures/`.

## Repository layout

```
download_data.py     download + unpack OPPORTUNITY
preprocess.py        channel selection, NaN filling, normalisation, splits
data.py              windowed datasets / data loaders / augmentation
models.py            DeepConvLSTM and ablation variants
losses.py            cross-entropy and focal loss
train.py             one training run (CLI)
evaluate.py          reload a checkpoint, evaluate any split
summarize.py         tables, aggregated metrics, figures
utils.py             metrics, seeding, logging
scripts/run_all.sh   full experiment matrix
tests/               unit tests
results/             committed CSV tables and figures
report/              report source and compiled PDF
```

## Hardware / runtime

Experiments were run on a single NVIDIA GeForce GTX 1650 Ti (4 GB).  At window
24 a reference epoch takes about 15 s; long-window runs (1.4 s / 2.0 s) take
about 3–5× longer per epoch.  Exact runtimes are recorded in each `run.json`.

## Reference

Ordóñez, F. J., & Roggen, D. (2016). Deep Convolutional and LSTM Recurrent
Neural Networks for Multimodal Wearable Activity Recognition. *Sensors*, 16(1),
115. https://doi.org/10.3390/s16010115

Chavarriaga, R., et al. (2013). The Opportunity challenge: A benchmark database
for on-body sensor-based activity recognition. *Pattern Recognition Letters*,
34(15), 2033–2042.
