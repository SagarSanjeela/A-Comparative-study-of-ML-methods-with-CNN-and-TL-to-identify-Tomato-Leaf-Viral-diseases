# -*- coding: utf-8 -*-
"""
Created on Sun May  4 18:34:48 2025

@author: sanjeela
"""

#Random forest

import tensorflow as tf
import numpy as np
import pandas as pd

# Load resized images (64x64)
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "yourFolderName/train",
    image_size=(256, 256),
    batch_size=128
)

raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "yourFolderName/val",
    image_size=(256, 256),
    batch_size=128
)

# Extract and flatten images and labels
def extract_features_labels(dataset):
    features = []
    labels = []
    for images, lbls in dataset:
        images = images / 255.0  # Normalize
        images_np = images.numpy().reshape(images.shape[0], -1)  # Flatten
        features.append(images_np)
        labels.append(lbls.numpy())
    return np.vstack(features), np.hstack(labels)

X_train, y_train = extract_features_labels(raw_train_ds)
X_val, y_val = extract_features_labels(raw_val_ds)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

#Step 2: Train and Evaluate Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on validation set
y_pred = rf_model.predict(X_val)

#Summary
print("\n=== Random Forest Model Summary ===")
print(f"Number of trees in the forest: {len(rf_model.estimators_)}")
print(f"Feature importances shape: {rf_model.feature_importances_.shape}")
print(f"Top 10 feature importances:\n{rf_model.feature_importances_[:10]}")

for i, tree in enumerate(rf_model.estimators_[:3]):
    print(f"\nTree {i}:")
    print(f" Depth: {tree.tree_.max_depth}")
    print(f" Number of nodes: {tree.tree_.node_count}")

print("\nModel hyperparameters:")
print(rf_model.get_params())

class_names = raw_val_ds.class_names
# Evaluation
print("Classification Report:\n", classification_report(y_val, y_pred, target_names=class_names))
cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=class_names, columns=class_names))

#Step 3: Predict a Single Image
# from PIL import Image
# import numpy as np

# Load and preprocess a single image
# img_path = 'yourFolderName/TSLS (61).jpg'
# img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
# img_array = tf.keras.utils.img_to_array(img)
# img_array = img_array / 255.0
# img_array = img_array.reshape(1, -1)  # Flatten

# # Predict
# predicted_class_index = rf_model.predict(img_array)[0]
# class_names = raw_train_ds.class_names

# print("Predicted class:", class_names[predicted_class_index])
