# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Any, Callable, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, losses, models, optimizers

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/denoising-dirty-documents/prepared/public"
OUTPUT_DATA_PATH = "output/f88466a1-e032-494a-acbe-a8ee4e4d23cf/9/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT, Any], Tuple[DT, DT]]


def train_robust_unet_tta(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains a lightweight U-Net with robust optimization (Huber Loss + Cosine Decay) 
    and applies TTA during inference.
    
    Architecture:
        - Input: GaussianNoise(0.05)
        - U-Net: 32 -> 64 -> 128 -> 256 (Bridge)
    
    Optimization:
        - Loss: Huber(delta=0.1)
        - Optimizer: Adam(lr=0.001)
        - Scheduler: CosineDecay
        - Epochs: 25
        
    Inference:
        - TTA: Average of [Original, FlipLR, FlipUD]
    """

    # -------------------------------------------------------------------------
    # 1. Data Preparation
    # -------------------------------------------------------------------------
    def to_numpy_stack(df: DT, col: str) -> np.ndarray:
        """Extracts and stacks patches from DataFrame/Series into (N, H, W, C) tensor."""
        if isinstance(df, pd.DataFrame):
            if col not in df.columns:
                # Fallback if specific column missing, though unlikely given pipeline
                return np.stack(df.values)
            data_iter = df[col].values
        elif isinstance(df, pd.Series):
            data_iter = df.values
        else:
            data_iter = df

        # Stack into a single numpy array
        return np.stack(list(data_iter))

    print("Converting data to numpy arrays...")
    X_train_np = to_numpy_stack(X_train, col='data')
    y_train_np = to_numpy_stack(y_train, col='label')
    X_val_np = to_numpy_stack(X_val, col='data')
    y_val_np = to_numpy_stack(y_val, col='label')

    # Dataset Constants
    BATCH_SIZE = 32
    EPOCHS = 25

    # Create TF Datasets
    # Note: Plan specifies GaussianNoise layer in model, so no external augmentation pipeline here
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_np, y_train_np))
    train_ds = train_ds.shuffle(buffer_size=1000, seed=42)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val_np, y_val_np))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # -------------------------------------------------------------------------
    # 2. Model Definition
    # -------------------------------------------------------------------------
    def conv_block(x, filters):
        """Standard U-Net Conv Block: [Conv-BN-ReLU] x 2"""
        x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def build_robust_unet(input_shape=(None, None, 1)):
        inputs = layers.Input(shape=input_shape)

        # Plan: Input Layer: Add GaussianNoise(0.05)
        x = layers.GaussianNoise(0.05)(inputs)

        # --- Encoder ---
        # Block 1
        c1 = conv_block(x, 32)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        # Block 2
        c2 = conv_block(p1, 64)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        # Block 3
        c3 = conv_block(p2, 128)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        # --- Bridge ---
        c4 = conv_block(p3, 256)

        # --- Decoder ---
        # Up 1
        u5 = layers.UpSampling2D((2, 2))(c4)
        u5 = layers.Concatenate()([u5, c3])
        c5 = conv_block(u5, 128)

        # Up 2
        u6 = layers.UpSampling2D((2, 2))(c5)
        u6 = layers.Concatenate()([u6, c2])
        c6 = conv_block(u6, 64)

        # Up 3
        u7 = layers.UpSampling2D((2, 2))(c6)
        u7 = layers.Concatenate()([u7, c1])
        c7 = conv_block(u7, 32)

        # Output
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)

        return models.Model(inputs=inputs, outputs=outputs)

    # -------------------------------------------------------------------------
    # 3. Optimization Setup
    # -------------------------------------------------------------------------
    try:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    except:
        strategy = tf.distribute.get_strategy()

    # Calculate decay steps for CosineDecay
    # decay_steps = (total_samples / batch_size) * epochs
    total_samples = X_train_np.shape[0]
    steps_per_epoch = total_samples // BATCH_SIZE
    decay_steps = steps_per_epoch * EPOCHS

    with strategy.scope():
        model = build_robust_unet()

        # Plan: Scheduler: CosineDecay (alpha=0.0)
        lr_schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=decay_steps,
            alpha=0.0
        )

        # Plan: Loss: Huber(delta=0.1)
        # Plan: Optimizer: Adam
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr_schedule),
            loss=losses.Huber(delta=0.1),
            metrics=['mae']
        )

    # Plan: Callbacks: ModelCheckpoint (save best val_loss)
    checkpoint_filepath = "best_model.keras"
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=0
        )
    ]

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------
    print(f"Starting training (Robust U-Net)... Samples: {total_samples}, Epochs: {EPOCHS}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )

    # Load best model for inference
    print("Loading best model from checkpoint...")
    try:
        model = models.load_model(checkpoint_filepath, custom_objects={'Huber': losses.Huber})
    except Exception as e:
        print(f"Warning: Could not load checkpoint ({e}). Using last epoch model.")

    # -------------------------------------------------------------------------
    # 5. Prediction (Validation)
    # -------------------------------------------------------------------------
    print("Predicting validation set...")
    val_preds_arr = model.predict(val_ds, verbose=1)

    val_preds_list = [val_preds_arr[i] for i in range(len(val_preds_arr))]

    if isinstance(X_val, pd.DataFrame):
        validation_predictions = pd.Series(val_preds_list, index=X_val.index)
    else:
        validation_predictions = pd.Series(val_preds_list)

    # -------------------------------------------------------------------------
    # 6. Prediction (Test with TTA)
    # -------------------------------------------------------------------------
    print("Predicting test set with TTA...")

    # Prepare test iteration
    if isinstance(X_test, pd.DataFrame):
        test_images = X_test['data'].values
        test_index = X_test.index
    else:
        test_images = X_test.values
        test_index = range(len(test_images))

    test_preds_list = []
    # Alignment for U-Net (pooling x3 => 2^3 = 8 divisor, but using 16 is safer)
    ALIGNMENT = 16

    for img in test_images:
        # img shape: (H, W, 1)
        h, w = img.shape[:2]

        # 1. Pad to multiple of ALIGNMENT
        pad_h = (ALIGNMENT - h % ALIGNMENT) % ALIGNMENT
        pad_w = (ALIGNMENT - w % ALIGNMENT) % ALIGNMENT

        if pad_h > 0 or pad_w > 0:
            img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            img_padded = img

        # 2. TTA Variants: Original, FlipLR, FlipUD
        v_orig = img_padded
        v_lr = np.fliplr(img_padded)
        v_ud = np.flipud(img_padded)

        batch = np.stack([v_orig, v_lr, v_ud], axis=0)  # (3, H, W, 1)

        # 3. Predict Batch
        preds_batch = model.predict(batch, verbose=0)

        # 4. Un-flip
        p_orig = preds_batch[0]
        p_lr = np.fliplr(preds_batch[1])
        p_ud = np.flipud(preds_batch[2])

        # 5. Average
        avg_pred = (p_orig + p_lr + p_ud) / 3.0

        # 6. Crop back to original size
        final_pred = avg_pred[:h, :w, :]
        test_preds_list.append(final_pred)

    if isinstance(X_test, pd.DataFrame):
        test_predictions = pd.Series(test_preds_list, index=test_index)
    else:
        test_predictions = pd.Series(test_preds_list)

    # Cleanup
    if os.path.exists(checkpoint_filepath):
        try:
            os.remove(checkpoint_filepath)
        except:
            pass

    return validation_predictions, test_predictions


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "robust_unet_tta": train_robust_unet_tta,
}
