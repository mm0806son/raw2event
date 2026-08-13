"""
Extract and Save CIFAR-10 Images by Class, and saves them into separate folders by class name as PNG files.
"""
import pickle
import numpy as np
import os
from PIL import Image

# Paths
input_dir = './cifar10'  # where your batches are
output_dir = './cifar10_extracted'   # where to save sorted images

# Class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Make folders for each class
for cls in classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# Load and save function
def extract_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data']
        labels = batch[b'labels']

    for i in range(len(images)):
        img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        label = labels[i]
        class_name = classes[label]
        img = Image.fromarray(img)
        img.save(os.path.join(output_dir, class_name, f'{filename[-1]}_{i}.png'))

# Process all 5 training batches
for i in range(1, 6):
    extract_batch(os.path.join(input_dir, f'data_batch_{i}'))

# Optional: do the same for test_batch
extract_batch(os.path.join(input_dir, 'test_batch'))
