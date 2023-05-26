import cv2
import numpy as np
import os

def change_saturation(image, saturation_factor):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * saturation_factor
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# Load the input image
image_path = 'dataset/offspin/9.png'
image = cv2.imread(image_path)

# List of saturation factors for the five images
saturation_factors = [0.7, 0.4, 1.7, 2.4, 2.8]

# Output directory for the modified images
output_dir = 'dataset/offspin/changed'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate and save the modified images
for i, saturation_factor in enumerate(saturation_factors):
    modified_image = change_saturation(image, saturation_factor)
    output_path = os.path.join(output_dir, f'image_9_{i+1}.png')
    cv2.imwrite(output_path, modified_image)
    print(f'Saved modified image: {output_path}')
