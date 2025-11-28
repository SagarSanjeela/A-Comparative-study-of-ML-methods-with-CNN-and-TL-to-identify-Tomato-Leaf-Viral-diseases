# -*- coding: utf-8 -*-
"""
Created on Mon May  5 20:55:03 2025

@author: sanje
"""

"""
Transfer Learning using MobileNetV2
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
#import seaborn as sns
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Set paths
train_dir = "C:/Users/Admin/Downloads/Paper 6/DataSet/train"
val_dir = "C:/Users/Admin/Downloads/Paper 6/DataSet/val"

# Parameters
BATCH_SIZE = 512
IMG_SIZE = (160, 160)  # MobileNetV2 default input size

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

# Preprocessing: Apply MobileNetV2 preprocessing
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Step 2: Load MobileNetV2 Base Model
base_model = MobileNetV2(input_shape=(160, 160, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False  # Freeze for feature extraction

# Step 3: Build the Transfer Learning Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Step 4: Train the Model
early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop, reduce_lr]
)

# Step 5: Plot Training History
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

plot_history(history)

# Step 6: Evaluation with Confusion Matrix
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))


print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=class_names, columns=class_names))
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()


# # Step 7: Predict a Single Image
# from tensorflow.keras.utils import load_img, img_to_array

# img_path = 'C:/Sanjeela/PhD Code/Mrs. Sanjeela sagar-760543/Dataset/Original/Septoria Leaf Spot/TSLS (61).jpg'
# img = load_img(img_path, target_size=IMG_SIZE)
# img_array = img_to_array(img)
# img_array = preprocess_input(img_array)
# img_array = np.expand_dims(img_array, axis=0)

# prediction = model.predict(img_array)
# predicted_class = class_names[np.argmax(prediction)]
# print("Predicted Class:", predicted_class)