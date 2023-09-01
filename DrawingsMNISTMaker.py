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


import cv2
import numpy as np
import os
import random

# Path to the shapes directory
base_folder = r"C:\yourpathgoeshere"

shapes = ["square", "circle", "triangle", "rhombus"]
labels_dict = {"square": 0, "circle": 1, "triangle": 2, "rhombus": 3}

data = []

# Process each image and label them
for shape in shapes:
    shape_folder = os.path.join(base_folder, shape)
    for filename in os.listdir(shape_folder):
        filepath = os.path.join(shape_folder, filename)

        # Read in grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Resize to 28x28 while maintaining aspect ratio
        img_resized = cv2.resize(img, (28, 28))

        # Normalize the image
        img_normalized = img_resized / 255.0

        # Get the label for this shape
        label = labels_dict[shape]

        # Append to data
        data.append((img_normalized, label))

# Shuffle the data to ensure randomness
random.shuffle(data)

# Split data into images and labels
images = [item[0] for item in data]
labels = [item[1] for item in data]

# Convert to numpy arrays
images_np = np.array(images)
labels_np = np.array(labels)

# Split data: 80% train, 10% validation, 10% test
train_split = int(0.8 * len(images_np))
val_split = int(0.9 * len(images_np))

train_images, train_labels = images_np[:train_split], labels_np[:train_split]
val_images, val_labels = images_np[train_split:val_split], labels_np[train_split:val_split]
test_images, test_labels = images_np[val_split:], labels_np[val_split:]

# Save the processed data
np.save(os.path.join(base_folder, 'train_images.npy'), train_images)
np.save(os.path.join(base_folder, 'train_labels.npy'), train_labels)
np.save(os.path.join(base_folder, 'val_images.npy'), val_images)
np.save(os.path.join(base_folder, 'val_labels.npy'), val_labels)
np.save(os.path.join(base_folder, 'test_images.npy'), test_images)
np.save(os.path.join(base_folder, 'test_labels.npy'), test_labels)

print("Processing Complete!")