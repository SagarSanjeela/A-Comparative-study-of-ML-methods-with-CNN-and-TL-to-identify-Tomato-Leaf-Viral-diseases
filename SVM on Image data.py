# -*- coding: utf-8 -*-
"""
Created on Sun May  4 17:49:23 2025

@author: sanjeela
"""
import psutil

# Get system memory info
mem = psutil.virtual_memory()
print(f"Total memory: {mem.total / (1024 ** 3):.2f} GB")
print(f"Available memory: {mem.available / (1024 ** 3):.2f} GB")

#SVM
import tensorflow as tf
import numpy as np
import pandas as pd

# Load raw training and validation datasets
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "yourFolderName/train",
    image_size=(256, 256),
    batch_size=64
)

raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "yourFolderName/val",
    image_size=(256, 256),
    batch_size=64
)

# Normalize images and extract features/labels from train and val datasets
def extract_features_labels(dataset):
    features = []
    labels = []
    for images, lbls in dataset:
        images = images / 255.0  # normalize
        images_np = images.numpy()
        images_np = images_np.reshape(images_np.shape[0], -1)  # flatten to vectors
        features.append(images_np)
        labels.append(lbls.numpy())
    return np.vstack(features), np.hstack(labels)

X_train, y_train = extract_features_labels(raw_train_ds)
X_val, y_val = extract_features_labels(raw_val_ds)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

#Train an SVM Classifier

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

print("Model fit")
# Train the SVM (you can tune kernel='linear', 'rbf', etc.)
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

#Evaluate the SVM
print("working...")
# Predict
y_pred = svm_model.predict(X_val)
print(y_pred)
# Metrics
class_names = raw_val_ds.class_names
print("Classification Report:\n", classification_report(y_val, y_pred,target_names=class_names))
#print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
class_names = raw_val_ds.class_names
print(pd.DataFrame(cm, index=class_names, columns=class_names))


# Print Summary of the model
print("=== SVM Model Summary ===")
print("Support vectors per class:", svm_model.n_support_)
print("Support vectors shape:", svm_model.support_vectors_.shape)
print("Coefficients shape:", svm_model.coef_.shape)
print("Intercept:", svm_model.intercept_)
print("Model parameters:", svm_model.get_params())

# import tensorflow as tf

# # 1. Load and preprocess the image
# img_path = 'yourFolderName/TSLS (61).jpg'
# img = tf.keras.utils.load_img(img_path, target_size=(256, 256))  # resize to match training
# img_array = tf.keras.utils.img_to_array(img)
# img_array = img_array / 255.0  # normalize
# img_array = img_array.reshape(1, -1)  # flatten and add batch dimension

# # 2. Predict using the SVM model
# predicted_class_index = svm_model.predict(img_array)[0]  # SVM outputs a class index

# class_names = raw_train_ds.class_names
# print(class_names)

# # 3. Map index to class label

# print("Predicted class:", class_names[predicted_class_index])
