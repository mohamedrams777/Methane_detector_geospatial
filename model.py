"""
model.py

Deep Learning Architectures for AI-Based Methane Hotspot Detection,
Cross-Sensor Super-Resolution Downscaling, and 7-Day Spatio-Temporal Forecasting.

Architectures included:
1. MultiSpectralResNet50: Custom 7-12 channel adapted ResNet-50 with Squeeze-and-Excitation (SE)
   channel attention and spatial refinement blocks.
2. Custom2DCNN: Lightweight baseline CNN for 32x32x7 patches.
3. MethaneSuperResolutionUNet: Spatial deconvolutional decoder for 10m sub-grid methane density mapping.
4. SpatioTemporalForecaster: ConvLSTM / Spatio-Temporal GRU for 7-day forward hotspot progression prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="MethaneAI")
class SqueezeAndExcitationBlock(layers.Layer):
    """
    Squeeze-and-Excitation (SE) Channel Attention Block.
    Explicitly models interdependencies between spectral bands (B2, B3, B4, B8, B11, B12, NDVI)
    to dynamically recalibrate spectral feature responses.
    """
    def __init__(self, channels: int, reduction_ratio: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.reduction_ratio = int(reduction_ratio)
        self.reduced_dim = max(1, self.channels // self.reduction_ratio)
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(self.reduced_dim, activation="relu", use_bias=False)
        self.fc2 = layers.Dense(self.channels, activation="sigmoid", use_bias=False)
        self.reshape = layers.Reshape((1, 1, self.channels))

    def call(self, inputs):
        se = self.gap(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)
        return inputs * se

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MethaneAI")
class SpatialAttentionBlock(layers.Layer):
    """
    Spatial Attention Block: Computes spatial attention mask using channel-wise
    average and max pooling across feature maps.
    """
    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)
        self.conv = layers.Conv2D(1, kernel_size=self.kernel_size, padding="same", activation="sigmoid", use_bias=False)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_att = self.conv(concat)
        return inputs * spatial_att

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


def build_multispectral_resnet50(
    input_shape=(32, 32, 7),
    num_classes: int = 1,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
) -> keras.Model:
    """
    Multi-Spectral ResNet-50 (MS-ResNet50) Architecture.
    
    1. Multispectral Stem: Adapts arbitrary input channels (e.g. 7 bands: B2, B3, B4, B8, B11, B12, NDVI)
       to 64 feature maps with 3x3 Conv, BatchNorm, and SE Channel Attention.
    2. ResNet-50 Feature Extractor: Deep residual blocks capturing high-level spatial context.
    3. Dual-Attention Refinement: Combines SE Channel Attention and Spatial Attention on top representations.
    4. Multi-Task Output:
       - Classification Head: Hotspot probability P(Hotspot = 1).
       - Regression Head: Estimated continuous local CH4 enhancement (ppb).
    """
    reg = keras.regularizers.l2(l2_reg)
    inputs = keras.Input(shape=input_shape, name="multispectral_patch")

    # --- 1. Multi-Spectral Stem ---
    x = layers.Conv2D(64, kernel_size=3, padding="same", kernel_regularizer=reg, name="ms_stem_conv1")(inputs)
    x = layers.BatchNormalization(name="ms_stem_bn1")(x)
    x = layers.ReLU(name="ms_stem_relu1")(x)
    x = SqueezeAndExcitationBlock(64, reduction_ratio=4, name="ms_stem_se")(x)

    # 64 -> 3 channels projection to interface with pre-trained ResNet-50 backbone
    x_rgb = layers.Conv2D(3, kernel_size=1, padding="same", name="channel_adapter_conv")(x)

    # --- 2. ResNet-50 Backbone ---
    base_resnet = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(32, 32, 3),
        pooling=None,
    )
    base_resnet.trainable = not freeze_backbone
    feat_maps = base_resnet(x_rgb)

    # --- 3. Attention Refinement on Residual Features ---
    feat_maps = SqueezeAndExcitationBlock(2048, reduction_ratio=16, name="top_se_attention")(feat_maps)
    feat_maps = SpatialAttentionBlock(kernel_size=3, name="top_spatial_attention")(feat_maps)

    # Global representation
    feat_vector = layers.GlobalAveragePooling2D(name="global_avg_pool")(feat_maps)
    feat_vector = layers.Dropout(dropout_rate, name="classifier_dropout")(feat_vector)
    feat_vector = layers.Dense(128, activation="relu", kernel_regularizer=reg, name="dense_latent")(feat_vector)

    # --- 4. Dual-Head Outputs ---
    # Head A: Hotspot Classification (Binary)
    hotspot_prob = layers.Dense(num_classes, activation="sigmoid", name="hotspot_prob")(feat_vector)
    
    # Head B: Methane Enhancement Regression (ppb)
    enhancement_ppb = layers.Dense(1, activation="relu", name="enhancement_ppb")(feat_vector)

    model = keras.Model(
        inputs=inputs,
        outputs={"hotspot_prob": hotspot_prob, "enhancement_ppb": enhancement_ppb},
        name="MultiSpectral_ResNet50",
    )
    return model


def build_custom_cnn(input_shape=(32, 32, 7), l2_reg=1e-4, dropout=0.4) -> keras.Model:
    """
    Primary baseline: Compact custom 2D CNN with 3 convolutional blocks.
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


def build_super_resolution_unet(input_shape=(32, 32, 7)) -> keras.Model:
    """
    Sub-grid Super-Resolution U-Net:
    Ingests 32x32x7 Sentinel-2 patch + coarse Sentinel-5P prior,
    and outputs a 32x32 pixel-level 10m-resolution methane emission intensity map.
    """
    inputs = keras.Input(shape=input_shape, name="sr_input")

    # Encoder
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D(2)(c1)  # 16x16

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D(2)(c2)  # 8x8

    # Bottleneck
    b = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    b = layers.Conv2D(128, 3, activation="relu", padding="same")(b)

    # Decoder
    u1 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(b)  # 16x16
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(u1)

    u2 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c3)  # 32x32
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(u2)

    # Output 10m pixel-level emission map
    density_map = layers.Conv2D(1, 1, activation="sigmoid", name="pixel_emission_density")(c4)

    return keras.Model(inputs, density_map, name="Methane_SuperResolution_UNet")


def build_spatiotemporal_forecaster(
    seq_length: int = 4,
    spatial_shape=(32, 32, 7),
    meteo_dim: int = 5,
    forecast_horizon: int = 7,
) -> keras.Model:
    """
    Spatio-Temporal ConvLSTM + Temporal Attention Network for 7-Day Forward Hotspot Progression.
    """
    # 1. Image Sequence Stream
    patch_seq_in = keras.Input(shape=(seq_length, *spatial_shape), name="patch_sequence_in")
    
    # Time-distributed CNN feature extractor
    cnn_encoder = keras.Sequential([
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
    ], name="time_distributed_cnn")
    
    time_feats = layers.TimeDistributed(cnn_encoder)(patch_seq_in)  # (batch, seq_len, 64)
    
    # Temporal sequence modeling via GRU
    temporal_mem = layers.GRU(64, return_sequences=False)(time_feats)  # (batch, 64)

    # 2. Meteorological Forcing Stream
    meteo_in = keras.Input(shape=(forecast_horizon, meteo_dim), name="meteo_forecast_in")
    meteo_gru = layers.GRU(32, return_sequences=False)(meteo_in)       # (batch, 32)

    # 3. Spatio-Temporal Fusion
    fused = layers.concatenate([temporal_mem, meteo_gru])
    dense_fused = layers.Dense(128, activation="relu")(fused)
    dense_fused = layers.Dropout(0.3)(dense_fused)

    # 4. Multi-Horizon Prediction (Day +1 through Day +7)
    forecast_prob = layers.Dense(forecast_horizon, activation="sigmoid", name="forecast_7day_probs")(dense_fused)

    return keras.Model(
        inputs=[patch_seq_in, meteo_in],
        outputs=forecast_prob,
        name="Methane_7Day_SpatioTemporal_Forecaster",
    )
