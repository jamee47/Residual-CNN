# Residual-CNN — Changes Walkthrough

All changes address the issues identified in the code review. Changes are grouped by file, ordered by priority.

---

## `model.py` — Architecture Fixes

### A2 · Mixed-dilation per residual block

**Problem:** Both `Conv1D` layers inside every residual block used the *same* dilation rate. A stack of `[d=2, d=2]` creates a "gridding" artefact — certain pilot positions are never covered by the receptive field.

**Fix:** `conv1` is now always undilated (`dilation_rate=1`) to preserve local context; `conv2` applies the requested dilation to widen the field.

```diff
-  # conv1 — same dilation as conv2
-  y = Conv1D(filters, kernel_size, dilation_rate=dilation_rate)(x)
+  # conv1 — always local (no holes)
+  y = Conv1D(filters, kernel_size, dilation_rate=1)(x)
   ...
-  # conv2 — same dilation (causes gridding)
-  y = Conv1D(filters, kernel_size, dilation_rate=dilation_rate)(y)
+  # conv2 — dilated to widen receptive field
+  y = Conv1D(filters, kernel_size, dilation_rate=dilation_rate)(y)
```

---

### A5 · BatchNormalization after `side_dense2`

**Problem:** The first side-info dense layer had BN; the second did not, creating an asymmetry that can cause the second layer's activations to drift.

```diff
  s = Dense(64, ...)(side_input)
  s = BatchNormalization()(s)       # had BN
  s = Dense(32, ...)(s)
+ s = BatchNormalization()(s)       # now symmetric
```

---

### A8 · `he_normal` weight initialisation

**Problem:** Default Glorot/Xavier initialisation is designed for `tanh`/sigmoid activations. For ReLU-based networks, `he_normal` is the theoretically correct choice, providing better gradient variance scaling.

**Fix:** Added `kernel_initializer="he_normal"` to every `Conv1D` and `Dense` layer in the model (stem, residual blocks, skip projections, side MLP, and fusion head).

---

## `preprocess.py` — Preprocessing Fixes

### B1 · Pilot scaler fits only on unmasked entries *(High priority)*

**Problem:** `pilot_scaler_re.fit(flat_re)` received all zeros from padded positions (mask=0), biasing both the mean and the standard deviation of the scaler.

```diff
- pilot_scaler_re.fit(flat_re)   # includes padded zeros → biased stats
- pilot_scaler_im.fit(flat_im)

+ valid = mask.astype(bool)
+ pilot_scaler_re.fit(re[valid].reshape(-1, 1))   # only real pilot positions
+ pilot_scaler_im.fit(im[valid].reshape(-1, 1))
```

The `transform` step still runs on the full array and the mask-zero-out is applied afterwards — only the *fit* step is restricted to unmasked entries.

---

### B2 · Stratified train/val/test split by MODCOD

**Problem:** Simple random splitting on concatenated rows (which are grouped by MODCOD file) risks splits where some MODCODs are underrepresented in validation or test.

```diff
- train_test_split(idx, ...)
+ train_test_split(idx, stratify=label_idx, ...)   # equal MODCOD coverage
```

---

### B3 · `import joblib` moved to top-level

**Problem:** `import joblib` was buried inside `build_dataset()`, hiding import errors until very late in a run.

```diff
+ import joblib   # moved to top of file
  ...
- import joblib   # was inside build_dataset()
  joblib.dump(...)
```

---

### B5 · Circular encoding of `phi_rad`

**Problem:** Feeding the raw phase angle `phi_rad` (radians) to `StandardScaler` is incorrect at the ±π wrap boundary — values near +π and −π are numerically close but map to opposite extremes of the standardised range.

**Fix:** Replace `phi_rad` with `sin_phi` and `cos_phi`. Both are bounded in `[-1, 1]`, wrap-continuous, and together uniquely encode any angle.

```diff
- SIDE_FEATURES = [..., "phi_rad", ...]
+ SIDE_FEATURES = [..., "sin_phi", "cos_phi", ...]

  # In _load_single_csv():
+ df["sin_phi"] = np.sin(df["phi_rad"].values)
+ df["cos_phi"] = np.cos(df["phi_rad"].values)
```

> **Note:** This increases the side-feature dimension by 1 (one `phi_rad` → two `sin/cos` columns).
> Re-run `preprocess.py` to regenerate `meta.json` and the `.npz` files before training.

---

### B6 · Log-transform `Ap_linear` (rain attenuation)

**Problem:** Linear rain attenuation is a power ratio spanning many orders of magnitude — highly right-skewed. `StandardScaler` cannot normalise this distribution effectively.

```diff
+ df["Ap_linear"] = np.log1p(df["Ap_linear"].values.clip(0))
```

---

### B7 · Log-transform `nvar_pilot` (noise variance)

Same reasoning as B6. Noise variance follows a log-normal distribution.

```diff
+ df["nvar_pilot"] = np.log1p(df["nvar_pilot"].values.clip(0))
```

---

## `predict.py` — Inference Fixes

### D2 · Evaluation metrics when ground truth is present

**Problem:** When a CSV with known `htrue_real`/`htrue_imag` was passed to `predict.py`, the true values were written to the output CSV but no metrics were printed — a missed opportunity for quick validation.

**Fix:** After prediction, if ground-truth columns are present, compute and print:
- MAE and RMSE for `hk_real` and `hk_imag` (overall)
- Per-MODCOD RMSE breakdown

```
[EVAL] Metrics vs ground truth:
  MAE   hk_real = 0.003412   hk_imag = 0.003108
  RMSE  hk_real = 0.005721   hk_imag = 0.005340

[EVAL] Per-MODCOD RMSE:
  16APSK_2_3           RMSE = 0.004812
  32APSK_5_6           RMSE = 0.006103
  ...
```

---

### D3 · Explicit `batch_size` in `model.predict`

**Problem:** Calling `model.predict(x, verbose=0)` without a `batch_size` argument uses Keras's default (32), which for large CSVs results in very slow inference or potential OOM.

```diff
- artifacts["model"].predict(x, verbose=0)
+ artifacts["model"].predict(x, batch_size=1024, verbose=0)
```

---

### D2+B5 · Feature engineering mirrored in `predict.py`

`_prep_from_df` now applies the same `sin_phi`/`cos_phi` encoding and `log1p` transforms that `preprocess.py` applies during training, so the features seen at inference time exactly match those the scaler was fitted on.

---

## `train.py` — Training Log Enhancement

### C1 · `TrainingSessionLogger` callback

A new callback writes two files to `--plot_dir` at the end of every training run:

- **`session_log.json`** — append-only JSON list, one record per run containing: session ID, start/end times, duration, hostname, hardware (GPU names or CPU), all CLI hyperparameters, `initial_epoch`, epochs trained, best epoch + `val_loss`, early-stopping flag, and final metrics.
- **`session_summary.txt`** — human-readable version of the above, also appended across runs.

### C6 · Model summary saved to disk

`model.summary()` output is now written to `--plot_dir/model_summary.txt` at the start of each run so the architecture is not lost when the terminal closes.

---

## File Summary

| File | Changes Applied |
|------|----------------|
| `model/model.py` | A2 (mixed dilation), A5 (BN on side_dense2), A8 (he_normal init) |
| `model/preprocess.py` | B1 (unmasked pilot fit), B2 (stratified split), B3 (joblib import), B5 (sin/cos phi), B6 (log Ap), B7 (log nvar) |
| `model/predict.py` | D2 (eval metrics), D3 (batched predict), feature engineering sync |
| `model/train.py` | C1 (TrainingSessionLogger), C6 (model_summary.txt) |
| `README.md` | Updated to document all architectural, preprocessing, training logging, prediction, and output folder changes. |

> **Important:** Because `preprocess.py` now has different feature engineering
> (B5 increases side-feature dim by 1; B6/B7 change feature scale), you **must
> re-run `preprocess.py`** to regenerate `data_processed/` before training.
> Any previously saved checkpoint will be incompatible with the new input shape.
