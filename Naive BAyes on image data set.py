# -*- coding: utf-8 -*-
"""
Created on Sun May  4 18:09:58 2025

@author: sanje
"""

#Naive Bayes
#Code to Train and Evaluate Naive Bayes Classifier
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import numpy as np

# 1. Load training and validation datasets
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Users/Admin/Downloads/Paper 6/DataSet/train",
    image_size=(256, 256),   # smaller size = fewer features (64*64*3 = 12,288)
    batch_size=64
)

raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Users/Admin/Downloads/Paper 6/DataSet/val",
    image_size=(256, 256),
    batch_size=64
)

# 2. Function to normalize and flatten images + collect labels
def extract_features_labels(dataset):
    features = []
    labels = []
    for images, lbls in dataset:
        images = images / 255.0  # Normalize pixel values
        images_np = images.numpy()
        images_np = images_np.reshape(images_np.shape[0], -1)  # Flatten (B, H*W*C)
        features.append(images_np)
        labels.append(lbls.numpy())
    return np.vstack(features), np.hstack(labels)

# 3. Extract flattened feature vectors and labels
X_train, y_train = extract_features_labels(raw_train_ds)
X_val, y_val = extract_features_labels(raw_val_ds)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")


# Step 2: Train Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Step 3: Predict on validation data
y_pred = nb_model.predict(X_val)

# Print Model Summary
print("\n=== Naive Bayes Model Summary ===")
print("Class prior probabilities:", nb_model.class_prior_)
print("Mean of each feature per class (shape):", nb_model.theta_.shape)
print("Mean of first 5 features for each class:\n", nb_model.theta_[:, :5])
print("Variance of each feature per class (shape):", nb_model.var_.shape)
print("Variance of first 5 features for each class:\n", nb_model.var_[:, :5])
print("Model parameters:\n", nb_model.get_params())

# Step 4: Evaluate
class_names = raw_val_ds.class_names
print(class_names)
print("Classification Report:\n", classification_report(y_val, y_pred, target_names=class_names))

cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=class_names, columns=class_names))

# Prediction on a Single Image
# 1. Load and preprocess the single image
#img_path = 'C:/Sanjeela/PhD Code/Mrs. Sanjeela sagar-760543/Dataset/Original/Septoria Leaf Spot/TSLS (61).jpg'
#img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
#img_array = tf.keras.utils.img_to_array(img)
#img_array = img_array / 255.0
#img_array = img_array.reshape(1, -1)  # Flatten

# 2. Predict using the Naive Bayes model
#predicted_class_index = nb_model.predict(img_array)[0]

# 3. Map index to class label
#class_names = raw_train_ds.class_names
#print("Predicted class:", class_names[predicted_class_index])