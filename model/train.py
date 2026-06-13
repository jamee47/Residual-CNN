"""
train.py
========
Trains the CNN-residual hk_real/hk_imag estimator on the preprocessed
dataset produced by preprocess.py.

Features:
  - Uses GPU automatically if available (TensorFlow auto-detects; we
    just print the device info and enable memory growth).
  - Saves model checkpoints every epoch (latest + best) so training can
    be resumed if interrupted.
  - On startup, automatically resumes from the latest checkpoint if one
    exists in --ckpt_dir.
  - Saves training history (loss/mae/rmse curves) as PNG plots and CSV
    after every epoch.
  - Early stopping + LR reduction on plateau.

Usage:
    python train.py --data_dir ./data_processed --ckpt_dir ./checkpoints \
                     --epochs 200 --batch_size 256
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import build_model, compile_model


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        print(f"[INFO] Using GPU(s): {[g.name for g in gpus]}")
    else:
        print("[INFO] No GPU found, using CPU.")


def load_split(data_dir, split):
    d = np.load(os.path.join(data_dir, f"{split}.npz"))
    return {"pilots": d["pilots"], "side": d["side"]}, d["targets"]


def make_dataset(x, y, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


class PlotHistoryCallback(tf.keras.callbacks.Callback):
    """Saves loss/mae/rmse curves and a CSV history after every epoch."""

    def __init__(self, out_dir, history_path):
        super().__init__()
        self.out_dir = out_dir
        self.history_path = history_path
        self.history = {}
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                self.history = json.load(f)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))

        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        self._plot()

    def _plot(self):
        metrics = ["loss", "mae", "rmse"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
        for ax, m in zip(axes, metrics):
            if m in self.history:
                ax.plot(self.history[m], label=f"train_{m}")
            vm = f"val_{m}"
            if vm in self.history:
                ax.plot(self.history[vm], label=vm)
            ax.set_title(m)
            ax.set_xlabel("epoch")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "training_curves.png"), dpi=120)
        plt.close(fig)


def main(args):
    configure_gpu()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)

    max_np = meta["max_np"]
    n_side = meta["n_side_features_total"]

    x_train, y_train = load_split(args.data_dir, "train")
    x_val, y_val = load_split(args.data_dir, "val")

    print(f"[INFO] max_np={max_np}, n_side_features={n_side}")
    print(f"[INFO] train: {y_train.shape[0]} samples, val: {y_val.shape[0]} samples")

    train_ds = make_dataset(x_train, y_train, args.batch_size, shuffle=True)
    val_ds = make_dataset(x_val, y_val, args.batch_size, shuffle=False)

    latest_path = os.path.join(args.ckpt_dir, "latest.keras")
    best_path = os.path.join(args.ckpt_dir, "best.keras")
    history_path = os.path.join(args.plot_dir, "history.json")
    initial_epoch = 0

    if os.path.exists(latest_path) and not args.fresh:
        print(f"[INFO] Resuming from checkpoint: {latest_path}")
        model = tf.keras.models.load_model(latest_path)
        # Recover epoch count from history if available
        if os.path.exists(history_path):
            with open(history_path) as f:
                hist = json.load(f)
            if "loss" in hist:
                initial_epoch = len(hist["loss"])
                print(f"[INFO] Resuming at epoch {initial_epoch}")
    else:
        print("[INFO] Building new model.")
        model = build_model(max_np=max_np, n_side_features=n_side)
        model = compile_model(model, learning_rate=args.lr)

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=latest_path, save_best_only=False, save_freq="epoch"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_path, save_best_only=True, monitor="val_loss", mode="min"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, args.patience // 2),
            min_lr=1e-6, verbose=1),
        PlotHistoryCallback(args.plot_dir, history_path),
        tf.keras.callbacks.CSVLogger(
            os.path.join(args.plot_dir, "training_log.csv"), append=True),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    final_path = os.path.join(args.ckpt_dir, "final_model.keras")
    model.save(final_path)
    print(f"[INFO] Saved final model to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data_processed")
    parser.add_argument("--ckpt_dir", default="./checkpoints")
    parser.add_argument("--plot_dir", default="./plots")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore existing checkpoint and start fresh.")
    args = parser.parse_args()

    main(args)
