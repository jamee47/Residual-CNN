"""
model.py — Residual-CNN
=======================
CNN-Residual deep network ("Residual-CNN") for predicting channel coefficient
(hk_real, hk_imag) from per-MODCOD pilot symbol sequences plus
scalar side-info (Es/No, code rate, modOrder, etc.).

Architecture overview
----------------------
Two input branches:
  1. Pilot branch: [MAX_NP, 3] tensor (re, im, mask) per sample.
     - 1D CNN stem -> stack of residual blocks (Conv1D + BN + ReLU,
       skip connections) -> global pooling.
     - The mask channel lets the network learn to ignore zero-padded
       pilot slots for MODCODs with Np < MAX_NP.
  2. Side-info branch: scalar features (Es/No, modOrder, code rate,
     numPilotBlks, modcod one-hot, etc.) -> small MLP.

The two branch outputs are concatenated and passed through a dense
regression head producing 2 outputs: hk_real, hk_imag.

Build with `build_model(max_np, n_side_features)`.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def residual_block(x, filters, kernel_size=3, dilation_rate=1, l2=1e-5, name=None):
    """1D conv residual block with pre-activation, BN, and skip connection."""
    shortcut = x

    y = layers.Conv1D(filters, kernel_size, padding="same",
                       dilation_rate=dilation_rate,
                       kernel_regularizer=regularizers.l2(l2),
                       name=None if name is None else f"{name}_conv1")(x)
    y = layers.BatchNormalization(name=None if name is None else f"{name}_bn1")(y)
    y = layers.ReLU(name=None if name is None else f"{name}_relu1")(y)

    y = layers.Conv1D(filters, kernel_size, padding="same",
                       dilation_rate=dilation_rate,
                       kernel_regularizer=regularizers.l2(l2),
                       name=None if name is None else f"{name}_conv2")(y)
    y = layers.BatchNormalization(name=None if name is None else f"{name}_bn2")(y)

    # Match channel dims for the skip connection if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same",
                                  kernel_regularizer=regularizers.l2(l2),
                                  name=None if name is None else f"{name}_proj")(shortcut)

    out = layers.Add(name=None if name is None else f"{name}_add")([shortcut, y])
    out = layers.ReLU(name=None if name is None else f"{name}_relu2")(out)
    return out


def build_model(max_np, n_side_features,
                 cnn_filters=(64, 64, 128, 128),
                 dilations=(1, 2, 4, 8),
                 dense_units=(128, 64),
                 dropout_rate=0.2,
                 l2=1e-5,
                 name="Residual-CNN"):
    """
    Build the CNN-residual model.

    Args:
        max_np: padded pilot sequence length (MAX_NP across all MODCODs).
        n_side_features: dimensionality of the scalar side-info vector
                          (includes one-hot modcod encoding).
        cnn_filters: number of filters for each residual block.
        dilations: dilation rate per residual block (same length as cnn_filters).
        dense_units: units for the dense head after concatenation.
        dropout_rate: dropout applied in dense layers.
        l2: L2 regularization strength.

    Returns:
        tf.keras.Model with inputs {"pilots": [B, max_np, 3], "side": [B, n_side_features]}
        and output [B, 2] -> (hk_real, hk_imag)
    """
    pilot_input = layers.Input(shape=(max_np, 3), name="pilots")
    side_input = layers.Input(shape=(n_side_features,), name="side")

    # ---- Pilot CNN-residual branch ----
    x = layers.Conv1D(cnn_filters[0], 7, padding="same",
                       kernel_regularizer=regularizers.l2(l2),
                       name="stem_conv")(pilot_input)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    for i, (f, d) in enumerate(zip(cnn_filters, dilations)):
        x = residual_block(x, f, kernel_size=3, dilation_rate=d, l2=l2,
                            name=f"resblock{i+1}")

    # Global pooling: average + max, concatenated
    avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    pilot_features = layers.Concatenate(name="pilot_pool_concat")([avg_pool, max_pool])

    # ---- Side-info MLP branch ----
    s = layers.Dense(64, activation="relu",
                      kernel_regularizer=regularizers.l2(l2), name="side_dense1")(side_input)
    s = layers.BatchNormalization(name="side_bn1")(s)
    s = layers.Dense(32, activation="relu",
                      kernel_regularizer=regularizers.l2(l2), name="side_dense2")(s)

    # ---- Fusion + regression head ----
    merged = layers.Concatenate(name="fusion_concat")([pilot_features, s])

    h = merged
    for i, u in enumerate(dense_units):
        h = layers.Dense(u, activation="relu",
                          kernel_regularizer=regularizers.l2(l2),
                          name=f"head_dense{i+1}")(h)
        h = layers.Dropout(dropout_rate, name=f"head_dropout{i+1}")(h)

    output = layers.Dense(2, activation="linear", name="hk_output")(h)

    model = models.Model(inputs={"pilots": pilot_input, "side": side_input},
                          outputs=output, name=name)
    return model


def compile_model(model, learning_rate=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


if __name__ == "__main__":
    # Quick sanity check
    m = build_model(max_np=792, n_side_features=25)
    m = compile_model(m)
    m.summary()
