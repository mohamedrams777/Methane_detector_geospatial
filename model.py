"""
model.py

BATCH 5 — 2D CNN architecture definitions.

Purpose
-------
Defines the primary custom 2D CNN for 32x32x7 methane-hotspot patches, sized
for a dataset in the low thousands of samples (not millions) — deep enough to
learn spatial texture, regularized enough to avoid overfitting a modest
dataset. Also provides an optional transfer-learning variant (ResNet/
EfficientNet with an adapted first layer) for comparison only, given the
pretrained-weight mismatch with 7-channel input.

Dependencies
------------
tensorflow (includes keras)
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_custom_cnn(input_shape=(32, 32, 7), l2_reg=1e-4, dropout=0.4) -> keras.Model:
    """
    Primary model: compact custom 2D CNN.

    Block 1: 32 filters x2 -> pool -> dropout
    Block 2: 64 filters x2 -> pool -> dropout
    Block 3: 128 filters -> global average pool
    Dense(64) -> dropout -> Dense(1, sigmoid)
    """
    reg = keras.regularizers.l2(l2_reg)
    inputs = keras.Input(shape=input_shape, name="patch_input")

    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="hotspot_prob")(x)

    return keras.Model(inputs, outputs, name="methane_hotspot_cnn")


def build_transfer_model(input_shape=(32, 32, 7), backbone: str = "resnet18",
                          freeze_backbone: bool = True) -> keras.Model:
    """
    OPTIONAL, secondary comparison model. Adapts a 3-channel-pretrained
    backbone to 7-channel input by replacing the first conv layer with a
    freshly-initialized 7-channel version (pretrained weights for that layer
    cannot transfer directly). Framed as an exploratory comparison only,
    given this weight mismatch — not the primary model for this project.

    Note: keras.applications does not ship a bare ResNet18; this uses
    EfficientNetB0 as the available small backbone. Swap in a different
    keras.applications model here if a different comparison is desired.
    """
    if backbone != "efficientnet_b0":
        raise ValueError(
            "Only 'efficientnet_b0' is wired up here (keras.applications has no "
            "bare ResNet18). Pass backbone='efficientnet_b0', or extend this "
            "function with another keras.applications model."
        )

    inputs = keras.Input(shape=input_shape, name="patch_input")
    # Custom entry conv: 7 channels -> 3 channels, so the pretrained backbone's
    # first real conv layer can still be used as intended (pretrained on RGB).
    x = layers.Conv2D(3, 1, padding="same", name="channel_adapter")(inputs)

    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(32, 32, 3), pooling="avg"
    )
    base.trainable = not freeze_backbone

    x = base(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="hotspot_prob")(x)

    return keras.Model(inputs, outputs, name="methane_hotspot_transfer_cnn")


if __name__ == "__main__":
    model = build_custom_cnn()
    model.summary()
