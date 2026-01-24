"""
BLOOD CELL CLASSIFICATION - INFERENCE ONLY
============================================
Chạy file này sau khi đã train xong và có model weights
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths - THAY ĐỔI NẾU CẦN
    TEST_DIR = '/content/drive/MyDrive/test1'
    MODEL_DIR = '/content/drive/MyDrive/BLOOD/models_config'  # Thư mục chứa model weights
    OUTPUT_DIR = '/content/output'
    
    # Classes
    CLASSES = ['BA', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PMY', 'SNE']
    NUM_CLASSES = len(CLASSES)
    
    # Settings
    IMG_SIZE = 224
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Models đã train
    MODELS = {
        'efficientnet_b2': 'tf_efficientnet_b2.ns_jft_in1k',
        'resnet50': 'resnet50.a1_in1k'
    }
    
    # Ensemble
    USE_STACKING = False  # True nếu có validation data để fit stacking

# ============================================================================
# MODEL
# ============================================================================

class BloodCellModel(nn.Module):
    def __init__(self, model_name, num_classes=Config.NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        num_features = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.head(self.backbone(x))

# ============================================================================
# DATASET & TRANSFORMS
# ============================================================================

class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']
        return image

def get_tta_transforms():
    """3 TTA transforms"""
    return [
        A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]

# ============================================================================
# LOAD MODELS
# ============================================================================

def load_trained_models():
    """Load tất cả models đã train"""
    models = {}
    
    for name, arch in Config.MODELS.items():
        model_path = os.path.join(Config.MODEL_DIR, f'{name}_best.pth')
        
        if not os.path.exists(model_path):
            print(f"⚠️ Không tìm thấy: {model_path}")
            continue
        
        print(f"📦 Loading {name}...")
        model = BloodCellModel(arch)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True))
        model = model.to(Config.DEVICE)
        model.eval()
        models[name] = model
        print(f"✅ Loaded {name}")
    
    return models

# ============================================================================
# PREDICTION WITH TTA
# ============================================================================

@torch.no_grad()
def predict_tta(model, test_paths):
    """Predict với Test Time Augmentation"""
    model.eval()
    tta_transforms = get_tta_transforms()
    n_tta = len(tta_transforms)
    n_samples = len(test_paths)
    
    all_probs = np.zeros((n_samples, Config.NUM_CLASSES))
    
    for tta_idx, transform in enumerate(tta_transforms):
        print(f"  TTA {tta_idx+1}/{n_tta}")
        
        dataset = TestDataset(test_paths, transform=transform)
        loader = DataLoader(
            dataset, 
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
        
        batch_start = 0
        for images in tqdm(loader, desc=f'    Batch', leave=False):
            images = images.to(Config.DEVICE, non_blocking=True)
            bs = images.size(0)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs[batch_start:batch_start+bs] += probs / n_tta
            batch_start += bs
    
    return all_probs

# ============================================================================
# PREPARE TEST DATA
# ============================================================================

def prepare_test_data():
    """Load test image paths"""
    test_paths = []
    test_ids = []
    
    for img_name in sorted(os.listdir(Config.TEST_DIR)):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            test_paths.append(os.path.join(Config.TEST_DIR, img_name))
            test_ids.append(img_name)
    
    print(f"📊 Test images: {len(test_paths)}")
    return test_paths, test_ids

# ============================================================================
# ENSEMBLE
# ============================================================================

def weighted_ensemble(probs_dict, weights=None):
    """Weighted average ensemble"""
    names = list(probs_dict.keys())
    
    if weights is None:
        # Equal weights
        weights = {name: 1.0 / len(names) for name in names}
    
    print(f"📊 Ensemble weights: {weights}")
    
    final_probs = np.zeros_like(list(probs_dict.values())[0])
    for name, probs in probs_dict.items():
        final_probs += weights.get(name, 1.0/len(names)) * probs
    
    return final_probs

# ============================================================================
# MAIN INFERENCE
# ============================================================================

def main():
    print("="*60)
    print("🔬 BLOOD CELL CLASSIFICATION - INFERENCE")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Test dir: {Config.TEST_DIR}")
    print(f"Model dir: {Config.MODEL_DIR}")
    print("="*60)
    
    # Load test data
    print("\n📁 Loading test data...")
    test_paths, test_ids = prepare_test_data()
    
    if len(test_paths) == 0:
        print("❌ Không tìm thấy ảnh test!")
        return
    
    # Load models
    print("\n📦 Loading trained models...")
    models = load_trained_models()
    
    if len(models) == 0:
        print("❌ Không tìm thấy model weights!")
        return
    
    # Predict
    print("\n🎯 Generating predictions with TTA...")
    test_probs = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        test_probs[name] = predict_tta(model, test_paths)
        torch.cuda.empty_cache()
    
    # Ensemble
    print("\n📊 Creating ensemble...")
    
    # Có thể điều chỉnh weights dựa trên validation F1
    # Ví dụ nếu B2 có F1=0.96, ResNet50 có F1=0.94:
    # weights = {'efficientnet_b2': 0.52, 'resnet50': 0.48}
    
    weights = None  # None = equal weights
    final_probs = weighted_ensemble(test_probs, weights)
    
    # Generate predictions
    final_preds = np.argmax(final_probs, axis=1)
    final_classes = [Config.CLASSES[p] for p in final_preds]
    
    # Save submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'TARGET': final_classes
    })
    
    submission_path = os.path.join(Config.OUTPUT_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    # Save probabilities (optional)
    probs_df = pd.DataFrame(final_probs, columns=Config.CLASSES)
    probs_df['ID'] = test_ids
    probs_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'probabilities.csv'), index=False)
    
    print(f"\n✅ Submission saved: {submission_path}")
    print(f"\n📊 Prediction distribution:")
    print(submission['TARGET'].value_counts())
    
    print("\n" + "="*60)
    print("🏆 INFERENCE COMPLETE!")
    print("="*60)

# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    main()