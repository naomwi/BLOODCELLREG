import os
import cv2
import numpy as np
from PIL import Image

train_dir = r"c:\Users\admin\Documents\Project\bloodcellclass\data\train"

classes = os.listdir(train_dir)
print(f"Directories found in train: {classes}")

total_images = 0
resolutions = set()
brightness_list = []
contrast_list = []
class_counts = {}

for cls in classes:
    cls_dir = os.path.join(train_dir, cls)
    if not os.path.isdir(cls_dir): continue
    
    images = os.listdir(cls_dir)
    class_counts[cls] = len(images)
    total_images += len(images)
    
    # Check a subsample per class to save time (max 100 images per class)
    sample_images = images[:100]
    for img_name in sample_images:
        img_path = os.path.join(cls_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        resolutions.add((img.shape[1], img.shape[0])) # W, H
        
        # Convert to grayscale to measure brightness/contrast
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_list.append(np.mean(gray))
        contrast_list.append(np.std(gray))

print("\n--- DATASET STATISTICS ---")
print(f"Total classes: {len(classes)}")
print(f"Total images: {total_images}")
print(f"Class distribution: {class_counts}")
print(f"Unique Resolutions (W, H): {resolutions}")
if brightness_list:
    print(f"Mean Brightness: {np.mean(brightness_list):.2f} (std: {np.std(brightness_list):.2f})")
    print(f"Mean Contrast (std dev of pixels): {np.mean(contrast_list):.2f} (std: {np.std(contrast_list):.2f})")
else:
    print("Could not analyze images.")
