"""
train_model.py
----------------
Trains a binary image classifier to distinguish:
    - Healthy pepper leaves
    - Bacterial spot pepper leaves

Approach: Transfer learning using MobileNetV2 (pretrained on ImageNet).

Dataset expected structure (pre-split Train/Val/Test):

dataset/
    Train/
        Bacterial Spot/
            img1.jpg
            ...
        Healthy/
            img1.jpg
            ...
    Val/
        Bacterial Spot/
        Healthy/
    Test/
        Bacterial Spot/
        Healthy/

Run this locally in VS Code.
After training, this script saves the model as `pepper_model.keras`,
which is loaded by app.py for inference.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------------
# 1. Config
# ------------------------------------------------------------------
DATASET_DIR = "dataset"        # contains Train/, Val/, Test/ subfolders
TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
VAL_DIR = os.path.join(DATASET_DIR, "Val")
TEST_DIR = os.path.join(DATASET_DIR, "Test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 123
MODEL_OUT = "pepper_model.keras"

# Class order is alphabetical by default from Keras:
# 0 -> Bacterial Spot
# 1 -> Healthy
CLASS_NAMES = ["Bacterial Spot", "Healthy"]

# ------------------------------------------------------------------
# 2. Load dataset (already split into Train/Val/Test folders)
# ------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

print("Detected classes:", train_ds.class_names)

# Performance: prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# ------------------------------------------------------------------
# 3. Data augmentation (applied only during training)
# ------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# ------------------------------------------------------------------
# 4. Build model (transfer learning)
# ------------------------------------------------------------------
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze base for initial training

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ------------------------------------------------------------------
# 5. Train (feature extraction phase)
# ------------------------------------------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop],
)

# ------------------------------------------------------------------
# 6. Fine-tuning phase (unfreeze top layers of base model)
# ------------------------------------------------------------------
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

fine_tune_epochs = 10
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs,
    callbacks=[early_stop],
)

# ------------------------------------------------------------------
# 7. Evaluate on the held-out test set
# ------------------------------------------------------------------
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    preds = (preds > 0.5).astype(int).flatten()
    y_pred.extend(preds)
    y_true.extend(labels.numpy().flatten().astype(int))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ------------------------------------------------------------------
# 8. Plot training curves
# ------------------------------------------------------------------
acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train Accuracy")
plt.plot(val_acc, label="Val Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.legend()
plt.title("Loss")
plt.savefig("training_curves.png")
plt.show()

# ------------------------------------------------------------------
# 9. Save model
# ------------------------------------------------------------------
model.save(MODEL_OUT)
print(f"\nModel saved to {MODEL_OUT}")