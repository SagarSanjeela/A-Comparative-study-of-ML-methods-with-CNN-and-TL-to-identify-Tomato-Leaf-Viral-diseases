# -*- coding: utf-8 -*-
"""
Created on Sun May  4 19:06:45 2025

@author: sanje
"""

#CNN

#Step 1: Load and Preprocess the Dataset
import tensorflow as tf

# Load dataset (resize to 128x128)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Sanjeela/mydataset/train",
    image_size=(128, 128),
    batch_size=32,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Sanjeela/mydataset/val",
    image_size=(128, 128),
    batch_size=32
)

# Normalize (0–1 scaling)
# Load raw datasets
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Sanjeela/mydataset/train",
    image_size=(128, 128),
    batch_size=32,
    shuffle=True
)

raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Sanjeela/mydataset/val",
    image_size=(128, 128),
    batch_size=32
)

# ✅ Save class names before mapping
class_names = raw_train_ds.class_names
num_classes = len(class_names)

# Apply normalization
#normalization_layer = tf.keras.layers.Rescaling(1./255)
#train_ds = raw_train_ds.map(lambda x, y: (normalization_layer(x), y))
#val_ds = raw_val_ds.map(lambda x, y: (normalization_layer(x), y))


#normalization_layer = tf.keras.layers.Rescaling(1./255)
#train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Class info
#class_names = train_ds.class_names
#print(class_names)
#num_classes = len(class_names)

# Step 2: Build the CNN Model
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Step 3: Train the Model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

#Step 4: Evaluate Using Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get true and predicted labels
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

#Step 5: Predict on a Single New Image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

img_path = 'C:/Sanjeela/PhD Code/Mrs. Sanjeela sagar-760543/Dataset/Original/Septoria Leaf Spot/TSLS (61).jpg'
img = load_img(img_path, target_size=(128, 128))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
print("Predicted Class:", predicted_class)