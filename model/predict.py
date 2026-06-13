"""
predict.py
==========
Loads a trained model + scalers (from preprocess.py) and predicts
hk_real / hk_imag for new samples, handling per-MODCOD pilot length
mismatches via zero-padding/masking to MAX_NP (same as training).

Two usage modes:

1. CSV mode (same row format as the dataset_gen.m output, with its
   own Np that may differ from MAX_NP):
     python predict.py --csv new_data.csv --data_dir ./data_processed \
                        --ckpt_dir ./checkpoints --out predictions.csv

2. Programmatic mode: import `predict_from_arrays`.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from preprocess import META_COLS, SIDE_FEATURES, _detect_np, _pad_pilots


def load_artifacts(data_dir, ckpt_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)

    side_scaler = joblib.load(os.path.join(data_dir, "side_scaler.pkl"))
    pilot_scaler_re = joblib.load(os.path.join(data_dir, "pilot_scaler_re.pkl"))
    pilot_scaler_im = joblib.load(os.path.join(data_dir, "pilot_scaler_im.pkl"))
    target_scaler = joblib.load(os.path.join(data_dir, "target_scaler.pkl"))

    model_path = os.path.join(ckpt_dir, "best.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(ckpt_dir, "final_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(ckpt_dir, "latest.keras")
    model = tf.keras.models.load_model(model_path)

    return {
        "meta": meta,
        "side_scaler": side_scaler,
        "pilot_scaler_re": pilot_scaler_re,
        "pilot_scaler_im": pilot_scaler_im,
        "target_scaler": target_scaler,
        "model": model,
        "model_path": model_path,
    }


def _prep_from_df(df, artifacts):
    meta = artifacts["meta"]
    max_np = meta["max_np"]
    label_to_idx = meta["label_to_idx"]
    n_labels = meta["n_labels"]

    np_pilots = _detect_np(df.shape[1])

    pilot_re = df[[f"pilot_re_{i+1}" for i in range(np_pilots)]].values.astype(np.float32)
    pilot_im = df[[f"pilot_im_{i+1}" for i in range(np_pilots)]].values.astype(np.float32)
    pilot_mask = df[[f"pilot_mask_{i+1}" for i in range(np_pilots)]].values.astype(np.float32)

    if np_pilots > max_np:
        # Truncate (rare: only if a new MODCOD has more pilots than training set saw)
        pilot_re = pilot_re[:, :max_np]
        pilot_im = pilot_im[:, :max_np]
        pilot_mask = pilot_mask[:, :max_np]
    else:
        pilot_re = _pad_pilots(pilot_re, max_np)
        pilot_im = _pad_pilots(pilot_im, max_np)
        pilot_mask = _pad_pilots(pilot_mask, max_np)

    # Scale pilot re/im, zero out padded entries
    flat_re = pilot_re.reshape(-1, 1)
    flat_im = pilot_im.reshape(-1, 1)
    re_s = artifacts["pilot_scaler_re"].transform(flat_re).reshape(pilot_re.shape)
    im_s = artifacts["pilot_scaler_im"].transform(flat_im).reshape(pilot_im.shape)
    re_s = re_s * pilot_mask
    im_s = im_s * pilot_mask

    pilots = np.stack([re_s, im_s, pilot_mask], axis=-1).astype(np.float32)

    # --- Mirror feature engineering from preprocess.py ---
    # Circular encoding of phase angle
    df = df.copy()  # avoid mutating the caller's DataFrame
    df["sin_phi"] = np.sin(df["phi_rad"].values)
    df["cos_phi"] = np.cos(df["phi_rad"].values)
    # Log-transform skewed power/variance features
    df["Ap_linear"] = np.log1p(df["Ap_linear"].values.clip(0))
    df["nvar_pilot"] = np.log1p(df["nvar_pilot"].values.clip(0))

    # Side features
    side = df[SIDE_FEATURES].values.astype(np.float32)
    labels = df["modcod_label"].values
    label_idx = np.array([label_to_idx.get(l, -1) for l in labels])
    if (label_idx == -1).any():
        unknown = set(labels[label_idx == -1])
        raise ValueError(f"Unknown modcod_label(s) not seen during training: {unknown}")

    label_onehot = np.eye(n_labels, dtype=np.float32)[label_idx]
    side_full = np.concatenate([side, label_onehot], axis=1)
    side_s = artifacts["side_scaler"].transform(side_full).astype(np.float32)

    return {"pilots": pilots, "side": side_s}


def predict_from_arrays(x, artifacts, batch_size=1024):
    """x: dict with 'pilots' [N, MAX_NP, 3] and 'side' [N, n_side] (already scaled).
    batch_size is passed to model.predict to avoid OOM on large inputs.
    """
    preds_scaled = artifacts["model"].predict(x, batch_size=batch_size, verbose=0)
    preds = artifacts["target_scaler"].inverse_transform(preds_scaled)
    return preds  # [N, 2] -> (hk_real, hk_imag)


def predict_from_csv(csv_path, artifacts, batch_size=1024):
    df = pd.read_csv(csv_path)
    x = _prep_from_df(df, artifacts)
    preds = predict_from_arrays(x, artifacts, batch_size=batch_size)

    # Build output DataFrame
    if "sample_id" in df.columns:
        out = df[["sample_id", "modcod", "modcod_label"]].copy()
    else:
        out = pd.DataFrame({"row_idx": np.arange(len(preds))})

    out["hk_real_pred"] = preds[:, 0]
    out["hk_imag_pred"] = preds[:, 1]

    if "htrue_real" in df.columns:
        out["hk_real_true"] = df["htrue_real"].values
        out["hk_imag_true"] = df["htrue_imag"].values
        # D2: print evaluation metrics when ground truth is available
        err_real = preds[:, 0] - df["htrue_real"].values
        err_imag = preds[:, 1] - df["htrue_imag"].values
        mae_real  = np.mean(np.abs(err_real))
        mae_imag  = np.mean(np.abs(err_imag))
        rmse_real = np.sqrt(np.mean(err_real ** 2))
        rmse_imag = np.sqrt(np.mean(err_imag ** 2))
        print("\n[EVAL] Metrics vs ground truth:")
        print(f"  MAE   hk_real = {mae_real:.6f}   hk_imag = {mae_imag:.6f}")
        print(f"  RMSE  hk_real = {rmse_real:.6f}   hk_imag = {rmse_imag:.6f}")
        # Per-MODCOD breakdown if label column is present
        if "modcod_label" in df.columns:
            print("\n[EVAL] Per-MODCOD RMSE:")
            for label, grp in out.groupby("modcod_label"):
                e_r = grp["hk_real_pred"].values - grp["hk_real_true"].values
                e_i = grp["hk_imag_pred"].values - grp["hk_imag_true"].values
                rmse = np.sqrt(np.mean(e_r**2 + e_i**2))
                print(f"  {label:<20} RMSE = {rmse:.6f}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV (dataset_gen.m row format)")
    parser.add_argument("--data_dir", default="./data_processed")
    parser.add_argument("--ckpt_dir", default="./checkpoints")
    parser.add_argument("--out", default="./predictions.csv")
    args = parser.parse_args()

    artifacts = load_artifacts(args.data_dir, args.ckpt_dir)
    print(f"[INFO] Loaded model from {artifacts['model_path']}")

    result = predict_from_csv(args.csv, artifacts)
    result.to_csv(args.out, index=False)
    print(f"[INFO] Wrote {len(result)} predictions to {args.out}")
    print(result.head())
