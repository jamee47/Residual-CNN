"""
preprocess.py
==============
Loads the 8 per-MODCOD CSV files produced by dataset_gen.m, handles the
fact that each file has a different pilot length (Np), and produces a
unified, padded/masked dataset ready for the CNN-residual model.

Each CSV row layout:
    [25 metadata cols] + pilot_re_1..Np + pilot_im_1..Np + pilot_mask_1..Np

Because Np differs per MODCOD/file, we:
  1. Read each file, detect its Np from the number of columns.
  2. Find MAX_NP = max Np across all 8 files.
  3. Zero-pad pilot_re / pilot_im / pilot_mask to MAX_NP (mask marks real vs padded).
  4. Stack pilots into a [N, MAX_NP, 3] tensor: (re, im, mask) per pilot slot.
  5. Extract scalar side-info features (Es/No, modOrder, code rate, etc.) and
     one-hot encode modcod_label, also append numPilotBlks normalized.
  6. Targets: htrue_real, htrue_imag.
  7. Standardize features, save scalers + MAX_NP + label encoding for reuse
     in train/predict.

Usage:
    python preprocess.py --data_dir /path/to/csvs --out_dir ./data_processed
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Metadata columns as written by dataset_gen.m (in order)
META_COLS = [
    "sample_id", "modcod", "modcod_label", "channel_type",
    "modOrder", "codeRate_num", "codeRate_den", "plFrameSize",
    "Np_actual", "numPilotBlks", "esno_nom_dB", "rainAtt_dB",
    "p_exceedance", "snr_eff_dB", "snr_awgn_dB", "nvar_pilot",
    "pilotScale", "Ap_linear", "phi_rad", "b_gain", "b_max",
    "htrue_real", "htrue_imag", "htrue_mag", "htrue_phase_rad",
]

# Scalar side-info features fed to the dense branch of the model
SIDE_FEATURES = [
    "modOrder", "codeRate_num", "codeRate_den", "plFrameSize",
    "Np_actual", "numPilotBlks", "esno_nom_dB", "rainAtt_dB",
    "p_exceedance", "snr_eff_dB", "snr_awgn_dB", "nvar_pilot",
    "pilotScale", "Ap_linear", "phi_rad", "b_gain", "b_max",
]

TARGET_COLS = ["htrue_real", "htrue_imag"]


def _detect_np(ncols):
    """Given total column count, recover Np (pilot length)."""
    n_meta = len(META_COLS)
    remaining = ncols - n_meta
    if remaining <= 0 or remaining % 3 != 0:
        raise ValueError(f"Unexpected column count {ncols}; cannot infer Np.")
    return remaining // 3


def _load_single_csv(path):
    df = pd.read_csv(path)
    np_pilots = _detect_np(df.shape[1])

    pilot_re = df[[f"pilot_re_{i+1}" for i in range(np_pilots)]].values
    pilot_im = df[[f"pilot_im_{i+1}" for i in range(np_pilots)]].values
    pilot_mask = df[[f"pilot_mask_{i+1}" for i in range(np_pilots)]].values

    side = df[SIDE_FEATURES].values.astype(np.float32)
    targets = df[TARGET_COLS].values.astype(np.float32)
    labels = df["modcod_label"].values

    return {
        "pilot_re": pilot_re.astype(np.float32),
        "pilot_im": pilot_im.astype(np.float32),
        "pilot_mask": pilot_mask.astype(np.float32),
        "side": side,
        "targets": targets,
        "labels": labels,
        "np_pilots": np_pilots,
    }


def _pad_pilots(arr, target_len):
    """Zero-pad a [N, Np] array along axis=1 to [N, target_len]."""
    n, np_cur = arr.shape
    if np_cur == target_len:
        return arr
    pad = np.zeros((n, target_len - np_cur), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def build_dataset(data_dir, out_dir, test_size=0.15, val_size=0.15, seed=42):
    os.makedirs(out_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    csv_files = [f for f in csv_files if "ref_pilots_lookup" not in f]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    loaded = [_load_single_csv(f) for f in csv_files]
    max_np = max(d["np_pilots"] for d in loaded)
    print(f"\nDetected per-file Np values: "
          f"{[(os.path.basename(f), d['np_pilots']) for f, d in zip(csv_files, loaded)]}")
    print(f"MAX_NP (padding target) = {max_np}")

    # Build unified label set
    all_labels = sorted(set(np.concatenate([d["labels"] for d in loaded]).tolist()))
    label_to_idx = {lab: i for i, lab in enumerate(all_labels)}
    print(f"\nModcod labels found ({len(all_labels)}): {all_labels}")

    pilot_re_list, pilot_im_list, pilot_mask_list = [], [], []
    side_list, target_list, label_idx_list = [], [], []

    for d in loaded:
        pilot_re_list.append(_pad_pilots(d["pilot_re"], max_np))
        pilot_im_list.append(_pad_pilots(d["pilot_im"], max_np))
        pilot_mask_list.append(_pad_pilots(d["pilot_mask"], max_np))
        side_list.append(d["side"])
        target_list.append(d["targets"])
        label_idx_list.append(np.array([label_to_idx[l] for l in d["labels"]]))

    pilot_re = np.concatenate(pilot_re_list, axis=0)
    pilot_im = np.concatenate(pilot_im_list, axis=0)
    pilot_mask = np.concatenate(pilot_mask_list, axis=0)
    side = np.concatenate(side_list, axis=0)
    targets = np.concatenate(target_list, axis=0)
    label_idx = np.concatenate(label_idx_list, axis=0)

    n_samples = pilot_re.shape[0]
    print(f"\nTotal samples: {n_samples}")

    # One-hot encode modcod label, append to side features
    n_labels = len(all_labels)
    label_onehot = np.eye(n_labels, dtype=np.float32)[label_idx]
    side_full = np.concatenate([side, label_onehot], axis=1)

    # Stack pilots into [N, MAX_NP, 3] (re, im, mask)
    pilots = np.stack([pilot_re, pilot_im, pilot_mask], axis=-1)

    # Train / val / test split
    idx = np.arange(n_samples)
    idx_train, idx_temp = train_test_split(idx, test_size=(test_size + val_size),
                                            random_state=seed, shuffle=True)
    rel_test = test_size / (test_size + val_size)
    idx_val, idx_test = train_test_split(idx_temp, test_size=rel_test,
                                          random_state=seed, shuffle=True)

    # Standardize side features (fit on train only); pilots are already
    # normalized by pilotScale in the generator, but we still standardize
    # the re/im channels (mask left untouched) for stable training.
    side_scaler = StandardScaler()
    side_train = side_scaler.fit_transform(side_full[idx_train])
    side_val = side_scaler.transform(side_full[idx_val])
    side_test = side_scaler.transform(side_full[idx_test])

    pilot_scaler_re = StandardScaler()
    pilot_scaler_im = StandardScaler()

    def _scale_pilots(p, fit=False):
        re = p[..., 0]
        im = p[..., 1]
        mask = p[..., 2]
        flat_re = re.reshape(-1, 1)
        flat_im = im.reshape(-1, 1)
        if fit:
            pilot_scaler_re.fit(flat_re)
            pilot_scaler_im.fit(flat_im)
        re_s = pilot_scaler_re.transform(flat_re).reshape(re.shape)
        im_s = pilot_scaler_im.transform(flat_im).reshape(im.shape)
        # zero-out padded entries after scaling
        re_s = re_s * mask
        im_s = im_s * mask
        return np.stack([re_s, im_s, mask], axis=-1).astype(np.float32)

    pilots_train = _scale_pilots(pilots[idx_train], fit=True)
    pilots_val = _scale_pilots(pilots[idx_val])
    pilots_test = _scale_pilots(pilots[idx_test])

    # Targets: standardize too (helps regression convergence)
    target_scaler = StandardScaler()
    targets_train = target_scaler.fit_transform(targets[idx_train])
    targets_val = target_scaler.transform(targets[idx_val])
    targets_test = target_scaler.transform(targets[idx_test])

    # Save arrays
    np.savez_compressed(
        os.path.join(out_dir, "train.npz"),
        pilots=pilots_train.astype(np.float32),
        side=side_train.astype(np.float32),
        targets=targets_train.astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(out_dir, "val.npz"),
        pilots=pilots_val.astype(np.float32),
        side=side_val.astype(np.float32),
        targets=targets_val.astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(out_dir, "test.npz"),
        pilots=pilots_test.astype(np.float32),
        side=side_test.astype(np.float32),
        targets=targets_test.astype(np.float32),
    )

    # Save metadata needed by train.py / predict.py
    meta = {
        "max_np": int(max_np),
        "n_side_features_raw": int(side.shape[1]),
        "n_labels": int(n_labels),
        "label_to_idx": label_to_idx,
        "side_feature_names": SIDE_FEATURES,
        "pilot_channels": 3,
        "n_side_features_total": int(side_full.shape[1]),
        "target_cols": TARGET_COLS,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save scalers
    import joblib
    joblib.dump(side_scaler, os.path.join(out_dir, "side_scaler.pkl"))
    joblib.dump(pilot_scaler_re, os.path.join(out_dir, "pilot_scaler_re.pkl"))
    joblib.dump(pilot_scaler_im, os.path.join(out_dir, "pilot_scaler_im.pkl"))
    joblib.dump(target_scaler, os.path.join(out_dir, "target_scaler.pkl"))

    print("\nSaved preprocessed dataset to:", out_dir)
    print(f"  train: {pilots_train.shape[0]} samples")
    print(f"  val:   {pilots_val.shape[0]} samples")
    print(f"  test:  {pilots_test.shape[0]} samples")
    print(f"  pilot tensor shape: {pilots_train.shape[1:]}")
    print(f"  side feature dim:   {side_train.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                         help="Directory containing the 8 per-MODCOD CSV files")
    parser.add_argument("--out_dir", default="./data_processed",
                         help="Output directory for processed npz/scalers/meta")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_dataset(args.data_dir, args.out_dir,
                   test_size=args.test_size, val_size=args.val_size, seed=args.seed)
