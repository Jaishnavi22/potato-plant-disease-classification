import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Image parameters
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 3

# Load dataset
datagen = ImageDataGenerator(rescale=1./255)

data = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Build CNN model
model = Sequential([
    Conv2D(16, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(data, epochs=EPOCHS)

# -------------------- PREDICTION PART --------------------

# Load a test image (change path if needed)
test_image_path = "test_leaf.jpg"

img = image.load_img(test_image_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("🍂 Prediction: Diseased Potato Leaf")
else:
    print("🌱 Prediction: Healthy Potato Leaf")

