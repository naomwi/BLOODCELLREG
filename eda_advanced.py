import os
import cv2
import numpy as np
import json

train_dir = r"c:\Users\admin\Documents\Project\bloodcellclass\data\train"
classes = os.listdir(train_dir)

def analyze_advanced_properties():
    results = {}
    
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir): continue
        
        images = os.listdir(cls_dir)[:20]
        
        color_variances = []
        blur_metrics = []
        saturation_means = []
        center_intensities = []
        
        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            means, stds = cv2.meanStdDev(img)
            color_variances.append(stds.flatten().tolist())
            
            saturation_means.append(np.mean(hsv[:,:,1]))
            blur_metrics.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            h, w = gray.shape
            center_crop = gray[h//4:3*h//4, w//4:3*w//4]
            center_intensities.append(np.mean(center_crop))

        results[cls] = {
            "Color_Var_B_G_R": [round(x, 2) for x in np.mean(color_variances, axis=0).tolist()],
            "Mean_Saturation": round(np.mean(saturation_means), 2),
            "Mean_Blur_Var": round(np.mean(blur_metrics), 1),
            "Mean_Center_Intensity": round(np.mean(center_intensities), 1)
        }
        
    return results

if __name__ == "__main__":
    results = analyze_advanced_properties()
    with open(r"c:\Users\admin\Documents\Project\bloodcellclass\eda_advanced_output.json", "w") as f:
        json.dump(results, f, indent=4)
