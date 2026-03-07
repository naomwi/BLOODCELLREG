"""
Run Test Inference from Checkpoints
====================================
Load trained checkpoints and generate predictions on test set.
Supports:
- Individual model evaluation
- Ensemble (soft voting / hard voting)
- TTA (Test Time Augmentation)
- Generates submission CSV

Usage:
    python run_test.py                          # Run all models, ensemble
    python run_test.py --model eva02_base       # Run single model only
    python run_test.py --no-tta                 # Disable TTA
    python run_test.py --no-ensemble            # Run each model separately (no ensemble)
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==================== CONFIG ====================
class Config:
    # Paths - sẽ được override bởi args
    TRAIN_DIR = "/workspace/data/train_combined"
    TEST_DIR = "/workspace/data/test1"
    OUTPUT_DIR = "/workspace/outputs"
    CHECKPOINT_DIR = "/workspace/checkpoints"

    # Classes - SORTED alphabetically
    CLASSES = sorted(['BA', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PMY', 'SNE'])
    NUM_CLASSES = 9

    # Ensemble Models
    ENSEMBLE_MODELS = [
        {
            'name': 'eva02_base',
            'timm_name': 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
            'img_size': 448,
            'batch_size': 12,
        },
        {
            'name': 'convnext_base',
            'timm_name': 'convnext_base.fb_in22k_ft_in1k_384',
            'img_size': 384,
            'batch_size': 16,
        },
        {
            'name': 'swin_base',
            'timm_name': 'swin_base_patch4_window12_384.ms_in22k_ft_in1k',
            'img_size': 384,
            'batch_size': 16,
        },
    ]

    N_FOLDS = 5
    SEED = 42
    NUM_WORKERS = 4
    USE_AMP = True


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ==================== DATASET ====================
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
            label = self.labels[idx]
            return image, label
        return image


# ==================== TRANSFORMS ====================
def get_val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_tta_transforms(img_size):
    """TTA transforms - deterministic rotations"""
    return [
        # Original
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # HFlip
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # VFlip
        A.Compose([
            A.Resize(img_size, img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Rotate 90
        A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=(90, 90), p=1.0, border_mode=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]


# ==================== MODEL ====================
class BloodCellModel(nn.Module):
    def __init__(self, model_name, num_classes=9, pretrained=False):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ==================== PREDICTION ====================
def predict_simple(model, image_paths, img_size, device, use_amp, batch_size=32):
    """Simple prediction without TTA"""
    model.eval()
    transform = get_val_transform(img_size)
    dataset = BloodCellDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=torch.cuda.is_available())

    all_probs = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc='Predicting', leave=False):
            images = images.to(device)
            if use_amp:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def predict_tta(model, image_paths, img_size, device, use_amp, batch_size=32):
    """TTA prediction"""
    model.eval()
    tta_transforms = get_tta_transforms(img_size)

    all_probs = []

    for t_idx, transform in enumerate(tta_transforms):
        dataset = BloodCellDataset(image_paths, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=Config.NUM_WORKERS, pin_memory=torch.cuda.is_available())

        probs = []
        with torch.no_grad():
            for images in tqdm(dataloader, desc=f'TTA {t_idx+1}/{len(tta_transforms)}', leave=False):
                images = images.to(device)
                if use_amp:
                    with autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
                probs.append(F.softmax(outputs, dim=1).cpu().numpy())

        all_probs.append(np.concatenate(probs, axis=0))

    return np.mean(all_probs, axis=0)


# ==================== DATA LOADING ====================
def load_test_data(test_dir):
    """Load test images"""
    test_images = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"Test images: {len(test_images)}")
    return test_images


# ==================== INFERENCE ====================
def run_single_model(model_config, test_images, device, use_tta=True):
    """Load all fold checkpoints for a model and predict"""

    model_name = model_config['name']
    timm_name = model_config['timm_name']
    img_size = model_config['img_size']
    batch_size = model_config['batch_size']
    use_amp = Config.USE_AMP and torch.cuda.is_available()

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"  Architecture: {timm_name}")
    print(f"  Image size: {img_size}")
    print(f"  TTA: {'ON' if use_tta else 'OFF'}")
    print(f"{'='*50}")

    fold_preds = []
    folds_loaded = 0

    for fold in range(Config.N_FOLDS):
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"{model_name}_fold{fold}.pt")

        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️ Fold {fold} checkpoint not found: {checkpoint_path}")
            continue

        print(f"\n  Loading fold {fold}...")
        model = BloodCellModel(timm_name, Config.NUM_CLASSES, pretrained=False).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        if use_tta:
            preds = predict_tta(model, test_images, img_size, device, use_amp, batch_size * 2)
        else:
            preds = predict_simple(model, test_images, img_size, device, use_amp, batch_size * 2)

        fold_preds.append(preds)
        folds_loaded += 1

        del model
        torch.cuda.empty_cache()

    if folds_loaded == 0:
        print(f"  ❌ No checkpoints found for {model_name}!")
        return None

    # Average across folds
    avg_preds = np.mean(fold_preds, axis=0)
    print(f"\n  ✓ {model_name}: {folds_loaded} folds loaded")

    # Print distribution
    pred_labels = avg_preds.argmax(axis=1)
    pred_counts = Counter(pred_labels)
    print(f"  Prediction distribution:")
    for idx in range(Config.NUM_CLASSES):
        count = pred_counts.get(idx, 0)
        pct = count / len(pred_labels) * 100
        print(f"    {Config.CLASSES[idx]}: {count} ({pct:.1f}%)")

    return avg_preds


def ensemble_models(all_preds_dict):
    """Ensemble predictions from multiple models"""

    model_names = list(all_preds_dict.keys())
    all_preds = list(all_preds_dict.values())

    print(f"\n{'='*60}")
    print(f"ENSEMBLE: {len(all_preds)} models")
    print(f"  Models: {', '.join(model_names)}")
    print(f"{'='*60}")

    # --- Soft Voting (Average Probabilities) ---
    soft_preds = np.mean(all_preds, axis=0)
    soft_labels = soft_preds.argmax(axis=1)

    print(f"\n[Soft Voting] Prediction distribution:")
    soft_counts = Counter(soft_labels)
    for idx in range(Config.NUM_CLASSES):
        count = soft_counts.get(idx, 0)
        pct = count / len(soft_labels) * 100
        print(f"  {Config.CLASSES[idx]}: {count} ({pct:.1f}%)")

    # --- Hard Voting (Majority Vote) ---
    all_labels = [p.argmax(axis=1) for p in all_preds]
    hard_labels = []
    for i in range(len(all_labels[0])):
        votes = [labels[i] for labels in all_labels]
        most_common = Counter(votes).most_common(1)[0][0]
        hard_labels.append(most_common)
    hard_labels = np.array(hard_labels)

    print(f"\n[Hard Voting] Prediction distribution:")
    hard_counts = Counter(hard_labels)
    for idx in range(Config.NUM_CLASSES):
        count = hard_counts.get(idx, 0)
        pct = count / len(hard_labels) * 100
        print(f"  {Config.CLASSES[idx]}: {count} ({pct:.1f}%)")

    # Compare soft vs hard
    agree = (soft_labels == hard_labels).mean()
    print(f"\nSoft vs Hard agreement: {agree:.1%}")

    return soft_preds, soft_labels, hard_labels


def save_submission(test_images, pred_labels, output_dir, suffix=""):
    """Save submission CSV"""
    submission = pd.DataFrame({
        'ID': [os.path.basename(p) for p in test_images],
        'TARGET': [Config.CLASSES[i] for i in pred_labels]
    })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"submission_test{suffix}_{ts}.csv"
    filepath = os.path.join(output_dir, filename)
    submission.to_csv(filepath, index=False)
    print(f"  ✓ Saved: {filepath}")
    return filepath


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description='Run test inference from checkpoints')
    parser.add_argument('--model', type=str, default=None,
                        help='Run specific model only (e.g. eva02_base, convnext_base, swin_base)')
    parser.add_argument('--no-tta', action='store_true', help='Disable TTA')
    parser.add_argument('--no-ensemble', action='store_true', help='Disable ensemble, run models separately')
    parser.add_argument('--test-dir', type=str, default=None, help='Override test data directory')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Override checkpoint directory')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    args = parser.parse_args()

    # Override configs
    if args.test_dir:
        Config.TEST_DIR = args.test_dir
    if args.checkpoint_dir:
        Config.CHECKPOINT_DIR = args.checkpoint_dir
    if args.output_dir:
        Config.OUTPUT_DIR = args.output_dir

    use_tta = not args.no_tta

    print("=" * 60)
    print("Blood Cell Classification - Test Inference")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Setup
    seed_everything(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint dir: {Config.CHECKPOINT_DIR}")
    print(f"Test dir: {Config.TEST_DIR}")
    print(f"Output dir: {Config.OUTPUT_DIR}")
    print(f"TTA: {'ON' if use_tta else 'OFF'}")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # List available checkpoints
    print(f"\nAvailable checkpoints:")
    if os.path.exists(Config.CHECKPOINT_DIR):
        ckpts = sorted([f for f in os.listdir(Config.CHECKPOINT_DIR) if f.endswith('.pt')])
        for ckpt in ckpts:
            size_mb = os.path.getsize(os.path.join(Config.CHECKPOINT_DIR, ckpt)) / 1e6
            print(f"  {ckpt} ({size_mb:.0f} MB)")
    else:
        print(f"  ❌ Checkpoint directory not found: {Config.CHECKPOINT_DIR}")
        return

    # Load test data
    print(f"\nLoading test data...")
    test_images = load_test_data(Config.TEST_DIR)

    if len(test_images) == 0:
        print("❌ No test images found!")
        return

    # Determine which models to run
    if args.model:
        models_to_run = [m for m in Config.ENSEMBLE_MODELS if m['name'] == args.model]
        if not models_to_run:
            print(f"❌ Model '{args.model}' not found. Available: {[m['name'] for m in Config.ENSEMBLE_MODELS]}")
            return
    else:
        models_to_run = Config.ENSEMBLE_MODELS

    # Override batch size if specified
    if args.batch_size:
        for m in models_to_run:
            m['batch_size'] = args.batch_size

    # Run inference
    all_preds = {}

    for model_config in models_to_run:
        preds = run_single_model(model_config, test_images, device, use_tta=use_tta)
        if preds is not None:
            all_preds[model_config['name']] = preds

            # Save individual model submission
            pred_labels = preds.argmax(axis=1)
            tta_suffix = "_tta" if use_tta else ""
            save_submission(test_images, pred_labels, Config.OUTPUT_DIR,
                            suffix=f"_{model_config['name']}{tta_suffix}")

            # Save probabilities
            np.save(
                os.path.join(Config.OUTPUT_DIR, f"test_probs_{model_config['name']}{tta_suffix}.npy"),
                preds
            )

    # Ensemble
    if len(all_preds) > 1 and not args.no_ensemble:
        soft_preds, soft_labels, hard_labels = ensemble_models(all_preds)

        tta_suffix = "_tta" if use_tta else ""

        # Save soft voting submission
        save_submission(test_images, soft_labels, Config.OUTPUT_DIR,
                        suffix=f"_ensemble_soft{tta_suffix}")

        # Save hard voting submission
        save_submission(test_images, hard_labels, Config.OUTPUT_DIR,
                        suffix=f"_ensemble_hard{tta_suffix}")

        # Save ensemble probabilities
        np.save(
            os.path.join(Config.OUTPUT_DIR, f"test_probs_ensemble{tta_suffix}.npy"),
            soft_preds
        )

    elif len(all_preds) == 1:
        print("\nOnly 1 model loaded, skipping ensemble.")

    # Summary
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"Models evaluated: {list(all_preds.keys())}")
    print(f"Test images: {len(test_images)}")
    print(f"Results saved to: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
