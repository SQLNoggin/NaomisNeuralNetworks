# Copyright (c) [2023] [Naomi Arroyo]
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

print("1. Importing necessary libraries...")

# Load your custom shapes dataset
print("\n2. Loading the custom shapes dataset...")
base_folder = r"C:\yourpathgoeshere"
train_images = np.load(os.path.join(base_folder, 'train_images.npy'))
train_labels = np.load(os.path.join(base_folder, 'train_labels.npy'))
test_images = np.load(os.path.join(base_folder, 'test_images.npy'))
test_labels = np.load(os.path.join(base_folder, 'test_labels.npy'))

print(f"Shape of training images: {train_images.shape}")
print(f"Shape of training labels: {train_labels.shape}")
print(f"Shape of testing images: {test_images.shape}")
print(f"Shape of testing labels: {test_labels.shape}")

# Your images are already normalized, so you don't need the normalization step
# Create a neural network model
print("\n4. Creating the neural network model...")
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(4)  # Output layer with 4 units (one for each shape)
])

# Compile the model
print("\n5. Compiling the model...")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
print("\n6. Training the model...")
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
print("\n7. Evaluating the model on test data...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Create a new model for predictions
print("\n8. Creating a model for generating predictions...")
probability_model = tf.keras.Sequential([
    model,
    layers.Softmax()
])

# Generating a prediction for the first test image
print("\n9. Predicting label for the first test image...")
predictions = probability_model.predict(test_images)
shape_names = ["square", "circle", "triangle", "rhombus"]
predicted_label = tf.argmax(predictions[0]).numpy()
print(f'Predicted label: {shape_names[predicted_label]}, Actual label: {shape_names[test_labels[0]]}')

# Save the model after training
model_save_path = os.path.join(base_folder, 'DrawingRecog_model.h5')
model.save(model_save_path)
print(f"Model saved to {model_save_path}")