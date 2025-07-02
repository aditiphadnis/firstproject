import os
import cv2
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


dataset_dir = "/Users/aditiphadnis/.cache/kagglehub/datasets/prathumarikeri/indian-sign-language-isl/versions/1"

image_paths = glob.glob(os.path.join(dataset_dir, "*", "*.jpg"))
print(f"Found {len(image_paths)} images.")
for img_path in image_paths[:5]:  # Show first 5 image paths
    print(img_path)
    
images = []
labels = []

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (64, 64))  # Resize for model input
        images.append(img)
        label = os.path.basename(os.path.dirname(img_path))
        labels.append(label)
    else:
        print(f"Failed to load image: {img_path}")

images = np.array(images)
labels = np.array(labels)

# Process the image

img = cv2.imread(image_paths[0])
cv2.imshow("Sample Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(images, labels_categorical, epochs=10, batch_size=32, validation_split=0.2)
model.save('model.h5')

# For text input
input_text = "A"  # or from speech-to-text

# Predict the sign
img = ...  # You need a way to map text to an image or sign class
pred = model.predict(np.expand_dims(img, axis=0))
sign_class = le.inverse_transform([np.argmax(pred)])

def animate_sign(sign_class):
    # Call Blender/Unity or your animation engine to play the sign
    print(f"Animating 3D avatar for sign: {sign_class}")
