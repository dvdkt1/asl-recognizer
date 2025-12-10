import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# CONFIG
DATA_FILE = 'initial_batch.json' 
MODEL_DIR = 'public/model'

# ensures directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# preprocessing
def normalize_landmarks(flat_landmarks):
    landmarks = np.array(flat_landmarks).reshape(-1, 3)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist
    return landmarks.flatten()

# loads data
print(f"Loading {DATA_FILE}...")
try:
    with open(DATA_FILE, 'r') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: {DATA_FILE} not found!")
    exit()

X = []
y = []

print("Preprocessing samples...")
for sample in raw_data:
    features = normalize_landmarks(sample['features'])
    X.append(features)
    y.append(sample['label'])

X = np.array(X)
y = np.array(y)

# encodes labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
CLASSES = label_encoder.classes_
print(f"Classes found: {CLASSES}")

with open(f'{MODEL_DIR}/classes.json', 'w') as f:
    json.dump(CLASSES.tolist(), f)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# defines model
inputs = tf.keras.Input(shape=(63,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# trains
print("\nStarting training...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# saves
print(f"\nSaving to model_v2.h5...")
model.save('model_v2.h5')
print("Done! Now run the conversion command.")