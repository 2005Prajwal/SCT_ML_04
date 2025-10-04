import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------- CONFIG --------
dataset_path = "C:/Users/PRAJWAL B/Downloads/archive (3)/leapGestRecog"
img_width, img_height = 64, 64  # Resize all images
model_save_path = "models.h5"

# -------- LOAD DATA --------
images = []
labels = []

# Recursively go through all folders
for root, dirs, files in os.walk(dataset_path):
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if image_files:
        # Take the last folder name as label
        label_name = os.path.basename(root)
        for file in image_files:
            img_path = os.path.join(root, file)
            img = load_img(img_path, target_size=(img_width, img_height))
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label_name)

# Check if images were loaded
if len(images) == 0:
    raise ValueError(f"No images found in dataset path: {dataset_path}")

# Convert to numpy arrays
X = np.array(images, dtype='float32') / 255.0  # normalize

# Encode labels as integers
unique_labels = sorted(list(set(labels)))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
labels_encoded = np.array([label_to_index[label] for label in labels])

y = to_categorical(labels_encoded)

print(f"Loaded {len(X)} images with {len(unique_labels)} classes.")

# -------- BUILD MODEL --------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------- TRAIN MODEL --------
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

# -------- SAVE MODEL --------
model.save(model_save_path)
print(f"Model saved as '{model_save_path}'")
