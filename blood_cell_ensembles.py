"""
BLOOD CELL CLASSIFICATION - SPEED OPTIMIZED FOR T4
===================================================
Estimated time: ~2-3 hours total
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from PIL import Image

# ============================================================================
# CONFIGURATION - SPEED OPTIMIZED
# ============================================================================

class Config:
    # Paths
    TRAIN_DIR = '/content/drive/MyDrive/train'
    TEST_DIR = '/content/drive/MyDrive/test'
    OUTPUT_DIR = '/content/output'
    
    # Classes
    CLASSES = ['BA', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PMY', 'SNE']
    NUM_CLASSES = len(CLASSES)
    
    # === SPEED OPTIMIZED ===
    IMG_SIZE = 224
    BATCH_SIZE = 128  # Tăng batch size
    EPOCHS = 20  # Giảm epochs
    LEARNING_RATE = 3e-4  # Tăng LR
    WEIGHT_DECAY = 1e-4
    
    # Freeze backbone
    FREEZE_BACKBONE_EPOCHS = 3
    BACKBONE_LR_MULT = 0.1
    
    # Loss
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    USE_CLASS_WEIGHTS = True
    LABEL_SMOOTHING = 0.1
    
    # === CHỈ 2 MODELS ===
    MODELS = {
        'efficientnet_b2': 'tf_efficientnet_b2.ns_jft_in1k',
        'resnet50': 'resnet50.a1_in1k'
    }
    
    # Others
    SEED = 42
    NUM_WORKERS = 4  # Tăng workers
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PATIENCE = 7
    USE_AMP = True
    USE_STACKING = True

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# SEED
# ============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # False cho speed
    torch.backends.cudnn.benchmark = True  # True cho speed

set_seed(Config.SEED)

# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ============================================================================
# DATA AUGMENTATION - FAST VERSION
# ============================================================================

def get_train_transforms():
    """FAST augmentation - removed slow transforms"""
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        ], p=0.5),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_tta_transforms():
    """Reduced TTA - only 3 transforms"""
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
# DATASET
# ============================================================================

class BloodCellDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image

# ============================================================================
# PREPARE DATA
# ============================================================================

def calculate_class_weights(labels):
    class_counts = Counter(labels)
    total = len(labels)
    weights = []
    
    print("\n📊 Class Weights:")
    for i in range(Config.NUM_CLASSES):
        count = class_counts.get(i, 1)
        weight = total / (Config.NUM_CLASSES * count)
        weights.append(weight)
        print(f"  {Config.CLASSES[i]}: {weight:.3f} (n={count})")
    
    weights = torch.FloatTensor(weights)
    weights = weights / weights.sum() * Config.NUM_CLASSES
    return weights

def prepare_data():
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(Config.CLASSES):
        class_dir = os.path.join(Config.TRAIN_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️ {class_dir} không tồn tại!")
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)
    
    total = len(image_paths)
    print(f"📈 Tổng ảnh train: {total}")
    
    class_counts = Counter(labels)
    for idx, name in enumerate(Config.CLASSES):
        count = class_counts.get(idx, 0)
        print(f"  {name}: {count} ({count/total*100:.1f}%)")
    
    return image_paths, labels

def prepare_test_data():
    test_paths = []
    test_ids = []
    
    for img_name in sorted(os.listdir(Config.TEST_DIR)):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            test_paths.append(os.path.join(Config.TEST_DIR, img_name))
            test_ids.append(img_name)
    
    print(f"📊 Tổng ảnh test: {len(test_paths)}")
    return test_paths, test_ids

# ============================================================================
# MODEL
# ============================================================================

class BloodCellModel(nn.Module):
    def __init__(self, model_name, num_classes=Config.NUM_CLASSES, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(512, num_classes)
        )
        self._frozen = False
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._frozen = True
        print("🔒 Backbone FROZEN")
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._frozen = False
        print("🔓 Backbone UNFROZEN")
    
    def forward(self, x):
        return self.head(self.backbone(x))

# ============================================================================
# TRAINING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc='Train', leave=False)
    for images, labels in pbar:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader), f1_score(all_labels, all_preds, average='macro')

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    for images, labels in tqdm(loader, desc='Valid', leave=False):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    return (total_loss / len(loader), 
            f1_score(all_labels, all_preds, average='macro'),
            all_preds, all_labels, np.vstack(all_probs))

def train_model(model_name, train_loader, valid_loader, class_weights):
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_name}")
    print(f"{'='*60}")
    
    model = BloodCellModel(Config.MODELS[model_name]).cuda()
    
    if Config.FREEZE_BACKBONE_EPOCHS > 0:
        model.freeze_backbone()
    
    class_weights_cuda = class_weights.cuda()
    criterion = FocalLoss(
        alpha=class_weights_cuda if Config.USE_CLASS_WEIGHTS else None,
        gamma=Config.FOCAL_GAMMA,
        label_smoothing=Config.LABEL_SMOOTHING
    ) if Config.USE_FOCAL_LOSS else nn.CrossEntropyLoss(
        weight=class_weights_cuda if Config.USE_CLASS_WEIGHTS else None,
        label_smoothing=Config.LABEL_SMOOTHING
    )
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    early_stopping = EarlyStopping(patience=Config.PATIENCE)
    
    best_f1 = 0
    best_model_path = os.path.join(Config.OUTPUT_DIR, f'{model_name}_best.pth')
    history = {'train_loss': [], 'train_f1': [], 'valid_loss': [], 'valid_f1': []}
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        # Unfreeze backbone
        if epoch == Config.FREEZE_BACKBONE_EPOCHS and model._frozen:
            model.unfreeze_backbone()
            optimizer = optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': Config.LEARNING_RATE * Config.BACKBONE_LR_MULT},
                {'params': model.head.parameters(), 'lr': Config.LEARNING_RATE}
            ], weight_decay=Config.WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS - epoch)
        
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        valid_loss, valid_f1, _, _, _ = validate(model, valid_loader, criterion)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['valid_loss'].append(valid_loss)
        history['valid_f1'].append(valid_f1)
        
        print(f"  Train Loss: {train_loss:.4f} | F1: {train_f1:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f} | F1: {valid_f1:.4f}")
        
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✅ Saved! Best F1: {best_f1:.4f}")
        
        early_stopping(valid_f1)
        if early_stopping.early_stop:
            print(f"  ⚠️ Early stopping!")
            break
    
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print(f"🏆 Best F1: {best_f1:.4f}")
    
    return model, history, best_f1

# ============================================================================
# PREDICTION
# ============================================================================

@torch.no_grad()
def predict_tta(model, test_paths, batch_size=128):
    model.eval()
    tta_transforms = get_tta_transforms()
    n_tta = len(tta_transforms)
    all_probs = np.zeros((len(test_paths), Config.NUM_CLASSES))
    
    for tta_idx, transform in enumerate(tta_transforms):
        print(f"  TTA {tta_idx+1}/{n_tta}")
        
        dataset = BloodCellDataset(test_paths, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=Config.NUM_WORKERS, pin_memory=True)
        
        batch_start = 0
        for images in loader:
            images = images.cuda(non_blocking=True)
            bs = images.size(0)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs[batch_start:batch_start+bs] += probs / n_tta
            batch_start += bs
    
    return all_probs

@torch.no_grad()
def get_valid_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    
    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
        
        all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
    
    return np.vstack(all_probs), np.array(all_labels)

# ============================================================================
# STACKING
# ============================================================================

class StackingEnsemble:
    def __init__(self):
        self.meta_model = LogisticRegression(
            multi_class='multinomial', solver='lbfgs', max_iter=1000, C=1.0
        )
    
    def fit(self, probs_list, y_true):
        X = np.hstack(probs_list)
        self.meta_model.fit(X, y_true)
        y_pred = self.meta_model.predict(X)
        print(f"📊 Stacking train F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    
    def predict_proba(self, probs_list):
        return self.meta_model.predict_proba(np.hstack(probs_list))

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_history(histories, names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for hist, name in zip(histories, names):
        axes[0].plot(hist['train_loss'], '--', label=f'{name} train')
        axes[0].plot(hist['valid_loss'], label=f'{name} valid')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    for hist, name in zip(histories, names):
        axes[1].plot(hist['train_f1'], '--', label=f'{name} train')
        axes[1].plot(hist['valid_f1'], label=f'{name} valid')
    axes[1].set_title('F1-Macro')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'history.png'), dpi=150)
    plt.show()

def plot_cm(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.CLASSES, yticklabels=Config.CLASSES)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, f'{title.replace(" ", "_")}.png'), dpi=150)
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🔬 BLOOD CELL CLASSIFICATION - SPEED OPTIMIZED")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    print(f"Batch: {Config.BATCH_SIZE} | Epochs: {Config.EPOCHS}")
    print(f"Models: {list(Config.MODELS.keys())}")
    print("="*70)
    
    # Data
    print("\n📁 Loading data...")
    train_paths, train_labels = prepare_data()
    test_paths, test_ids = prepare_test_data()
    
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=Config.SEED
    )
    print(f"Train: {len(train_paths)} | Valid: {len(valid_paths)} | Test: {len(test_paths)}")
    
    class_weights = calculate_class_weights(train_labels)
    
    # DataLoaders
    train_dataset = BloodCellDataset(train_paths, train_labels, get_train_transforms())
    valid_dataset = BloodCellDataset(valid_paths, valid_labels, get_valid_transforms())
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True,
        persistent_workers=True
    )
    
    # Train
    models = {}
    histories = []
    f1_scores = {}
    valid_probs_dict = {}
    
    for name in Config.MODELS:
        model, hist, f1 = train_model(name, train_loader, valid_loader, class_weights)
        models[name] = model
        histories.append(hist)
        f1_scores[name] = f1
        
        probs, _ = get_valid_probs(model, valid_loader)
        valid_probs_dict[name] = probs
        
        torch.cuda.empty_cache()
    
    plot_history(histories, list(Config.MODELS.keys()))
    
    # Test predictions
    print("\n🎯 Test predictions with TTA...")
    test_probs_dict = {}
    for name, model in models.items():
        print(f"\n{name}:")
        test_probs_dict[name] = predict_tta(model, test_paths)
    
    # Ensemble
    if Config.USE_STACKING:
        print("\n📦 Stacking ensemble...")
        stacker = StackingEnsemble()
        stacker.fit(list(valid_probs_dict.values()), valid_labels)
        
        final_probs = stacker.predict_proba(list(test_probs_dict.values()))
        
        # Validate stacking
        valid_pred = stacker.meta_model.predict(np.hstack(list(valid_probs_dict.values())))
        print(f"📊 Stacking valid F1: {f1_score(valid_labels, valid_pred, average='macro'):.4f}")
        plot_cm(valid_labels, valid_pred, 'Stacking Validation')
    else:
        weights = np.array(list(f1_scores.values()))
        weights /= weights.sum()
        print(f"\n📊 Weights: {dict(zip(Config.MODELS.keys(), weights.round(3)))}")
        
        final_probs = sum(w * test_probs_dict[n] for w, n in zip(weights, Config.MODELS))
    
    # Save
    final_preds = np.argmax(final_probs, axis=1)
    final_classes = [Config.CLASSES[p] for p in final_preds]
    
    submission = pd.DataFrame({'ID': test_ids, 'TARGET': final_classes})
    submission.to_csv(os.path.join(Config.OUTPUT_DIR, 'submission.csv'), index=False)
    
    print(f"\n✅ Saved: {Config.OUTPUT_DIR}/submission.csv")
    print(f"\n📊 Distribution:\n{pd.Series(final_classes).value_counts()}")
    
    print("\n" + "="*70)
    print("🏆 DONE!")
    for name, f1 in f1_scores.items():
        print(f"  {name}: {f1:.4f}")
    print("="*70)

if __name__ == '__main__':
    main()