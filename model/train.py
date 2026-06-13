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
import sys
import json
import socket
import platform
import argparse
import datetime
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



class TrainingSessionLogger(tf.keras.callbacks.Callback):
    """
    Writes a rich session record to {plot_dir}/session_log.json
    (append-only list of runs) and a human-readable summary to
    {plot_dir}/session_summary.txt after training finishes.

    Each record captures:
      - Unique session ID (ISO timestamp)
      - Start / end time and total duration
      - All CLI hyperparameters
      - Hardware (GPU names or 'CPU')
      - initial_epoch, epochs actually trained
      - Final train + val metrics (loss, mae, rmse)
      - Best epoch index and its val_loss
      - Whether EarlyStopping fired
      - Path of the saved model that will be produced
    """

    def __init__(self, plot_dir, args, initial_epoch, ckpt_dir):
        super().__init__()
        self.plot_dir = plot_dir
        self.args = vars(args)
        self.initial_epoch = initial_epoch
        self.ckpt_dir = ckpt_dir
        self.session_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self._start_time = None
        self._epoch_logs = []          # one dict per epoch
        self._stopped_early = False

    # ------------------------------------------------------------------ #
    def on_train_begin(self, logs=None):
        self._start_time = datetime.datetime.now()
        gpus = tf.config.list_physical_devices("GPU")
        self._hw = [g.name for g in gpus] if gpus else ["CPU"]

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_logs.append({"epoch": epoch, **(logs or {})})

    def on_train_end(self, logs=None):
        end_time = datetime.datetime.now()
        duration = (end_time - self._start_time).total_seconds()

        # Detect early stopping: if we ended before --epochs
        epochs_trained = len(self._epoch_logs)
        max_epochs = self.args.get("epochs", 0)
        self._stopped_early = (epochs_trained < (max_epochs - self.initial_epoch))

        # Best epoch by val_loss
        best_epoch = None
        best_val_loss = None
        for rec in self._epoch_logs:
            if "val_loss" in rec:
                if best_val_loss is None or rec["val_loss"] < best_val_loss:
                    best_val_loss = rec["val_loss"]
                    best_epoch = rec["epoch"]

        # Final metrics (last epoch)
        final_metrics = {}
        if self._epoch_logs:
            last = self._epoch_logs[-1]
            final_metrics = {k: float(v) for k, v in last.items() if k != "epoch"}

        record = {
            "session_id": self.session_id,
            "start_time": self._start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "hardware": self._hw,
            "hyperparameters": self.args,
            "initial_epoch": self.initial_epoch,
            "epochs_trained": epochs_trained,
            "final_epoch": (self.initial_epoch + epochs_trained - 1)
                            if epochs_trained > 0 else self.initial_epoch,
            "early_stopped": self._stopped_early,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
            "final_metrics": final_metrics,
            "final_model_path": os.path.join(self.ckpt_dir, "final_model.keras"),
        }

        self._write_json_log(record)
        self._write_text_summary(record)
        print(f"[SESSION LOG] Session {self.session_id} recorded → "
              f"{os.path.join(self.plot_dir, 'session_log.json')}")

    # ------------------------------------------------------------------ #
    def _write_json_log(self, record):
        log_path = os.path.join(self.plot_dir, "session_log.json")
        sessions = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    sessions = json.load(f)
                if not isinstance(sessions, list):
                    sessions = [sessions]   # migrate old single-record files
            except (json.JSONDecodeError, ValueError):
                sessions = []              # corrupted file — start fresh
        sessions.append(record)
        with open(log_path, "w") as f:
            json.dump(sessions, f, indent=2)

    def _write_text_summary(self, record):
        summary_path = os.path.join(self.plot_dir, "session_summary.txt")
        sep = "=" * 60
        lines = [
            sep,
            f"Session ID  : {record['session_id']}",
            f"Start       : {record['start_time']}",
            f"End         : {record['end_time']}",
            f"Duration    : {record['duration_seconds']:.1f} s  "
            f"({record['duration_seconds']/60:.1f} min)",
            f"Host        : {record['hostname']}",
            f"Hardware    : {', '.join(record['hardware'])}",
            "",
            "Hyperparameters:",
        ]
        for k, v in record["hyperparameters"].items():
            lines.append(f"  {k:<15} = {v}")
        lines += [
            "",
            f"Initial epoch   : {record['initial_epoch']}",
            f"Epochs trained  : {record['epochs_trained']}",
            f"Final epoch     : {record['final_epoch']}",
            f"Early stopped   : {record['early_stopped']}",
            f"Best epoch      : {record['best_epoch']}",
            f"Best val_loss   : {record['best_val_loss']}",
            "",
            "Final metrics:",
        ]
        for k, v in record["final_metrics"].items():
            lines.append(f"  {k:<25} = {v:.6f}")
        lines += [
            "",
            f"Model saved to  : {record['final_model_path']}",
            sep,
            "",
        ]
        # Append so previous runs are preserved in the same file
        with open(summary_path, "a") as f:
            f.write("\n".join(lines) + "\n")


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

    # Save model summary to a text file so it survives across runs
    summary_path = os.path.join(args.plot_dir, "model_summary.txt")
    with open(summary_path, "w") as sf:
        model.summary(print_fn=lambda line: sf.write(line + "\n"))
    model.summary()  # also print to stdout as before

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
        TrainingSessionLogger(
            plot_dir=args.plot_dir,
            args=args,
            initial_epoch=initial_epoch,
            ckpt_dir=args.ckpt_dir,
        ),
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
