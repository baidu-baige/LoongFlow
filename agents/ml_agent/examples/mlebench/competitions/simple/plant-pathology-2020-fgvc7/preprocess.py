import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import Tuple, List, Any

# Task-adaptive type definitions
# X is a PyTorch Dataset that yields (image_tensor, label_tensor) or image_tensor
# y is the original target numpy array
X = Any  
y = np.ndarray

class PlantPathologyDataset(Dataset):
    """
    A custom Dataset for the Plant Pathology 2020 task.
    Handles image loading, geometric/color augmentations, and advanced 
    regularization techniques like Mixup and Cutmix.
    """
    def __init__(
        self, 
        paths: List[str], 
        labels: np.ndarray = None, 
        transform: T.Compose = None,
        mixup_prob: float = 0.0,
        cutmix_prob: float = 0.0
    ):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        
    def __len__(self) -> int:
        return len(self.paths)

    def _mixup(self, img1: torch.Tensor, lbl1: torch.Tensor, img2: torch.Tensor, lbl2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies Mixup augmentation."""
        lam = np.random.beta(0.2, 0.2)
        img = lam * img1 + (1 - lam) * img2
        lbl = lam * lbl1 + (1 - lam) * lbl2
        return img, lbl

    def _cutmix(self, img1: torch.Tensor, lbl1: torch.Tensor, img2: torch.Tensor, lbl2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies Cutmix augmentation."""
        lam = np.random.beta(1.0, 1.0)
        _, H, W = img1.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Clone to avoid modifying the original data if cached
        img = img1.clone()
        img[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        actual_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        lbl = actual_lam * lbl1 + (1 - actual_lam) * lbl2
        return img, lbl

    def __getitem__(self, idx: int) -> Any:
        # Load image
        img_path = self.paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is None:
            return image
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Apply Mixup or Cutmix during training (if labels and probs are provided)
        # We use a combined probability as per Technical Specification (p=0.5 for either)
        total_aug_prob = self.mixup_prob + self.cutmix_prob
        if total_aug_prob > 0 and np.random.rand() < total_aug_prob:
            # Pick a random second sample
            idx2 = np.random.randint(len(self.paths))
            image2 = Image.open(self.paths[idx2]).convert("RGB")
            if self.transform:
                image2 = self.transform(image2)
            label2 = torch.tensor(self.labels[idx2], dtype=torch.float32)
            
            # Decide between Mixup and Cutmix based on relative probabilities
            if np.random.rand() < (self.mixup_prob / total_aug_prob):
                image, label = self._mixup(image, label, image2, label2)
            else:
                image, label = self._cutmix(image, label, image2, label2)
        
        return image, label

def preprocess(
    X_train: List[str],
    y_train: np.ndarray,
    X_val: List[str],
    y_val: np.ndarray,
    X_test: List[str]
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw image paths and labels into model-ready PyTorch Datasets.
    Implements heavy geometric and color augmentations as per technical specification.
    """
    print("Stage 3: Preprocessing and defining augmentation strategy...")
    
    # ImageNet normalization constants (standard for pre-trained models)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    # Training Augmentation Pipeline
    # Focus: Invariance to lighting, orientation, and scale.
    train_transform = T.Compose([
        T.RandomResizedCrop(
            size=512, 
            scale=(0.8, 1.0), 
            ratio=(0.75, 1.33)
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1
        ),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    # Validation and Test Transformation Pipeline
    # Focus: Consistency and resizing.
    val_test_transform = T.Compose([
        T.Resize(512),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    # Create Dataset objects
    # Note: Mixup/Cutmix are applied with a total probability of 0.5 (0.25 each)
    X_train_processed = PlantPathologyDataset(
        paths=X_train,
        labels=y_train,
        transform=train_transform,
        mixup_prob=0.25,
        cutmix_prob=0.25
    )
    
    X_val_processed = PlantPathologyDataset(
        paths=X_val,
        labels=y_val,
        transform=val_test_transform
    )
    
    X_test_processed = PlantPathologyDataset(
        paths=X_test,
        labels=None,
        transform=val_test_transform
    )
    
    # Sanity Checks
    assert len(X_train_processed) == len(y_train), "Train feature/target mismatch"
    assert len(X_val_processed) == len(y_val), "Val feature/target mismatch"
    assert len(X_test_processed) == len(X_test), "Test completeness check failed"
    
    print(f"Preprocessing complete. Training samples: {len(X_train_processed)}, Validation samples: {len(X_val_processed)}")
    print(f"Augmentations configured: RandomResizedCrop, Flips, Rotation, ColorJitter, Mixup/Cutmix (p=0.5)")

    return X_train_processed, y_train, X_val_processed, y_val, X_test_processed