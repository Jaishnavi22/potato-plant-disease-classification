import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Image parameters
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 3

# Load images from dataset folder
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

print("🌱 Potato plant disease classification completed")
