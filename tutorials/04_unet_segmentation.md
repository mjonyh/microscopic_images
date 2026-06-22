# Tutorial 4: U-Net Segmentation of Phase-Contrast Microscopy Images

## Overview

This tutorial covers the U-Net architecture used for cell segmentation in phase-contrast microscopy images. U-Net is the standard deep learning architecture for biomedical image segmentation, providing precise localization through its encoder-decoder structure with skip connections.

## Architecture

```
Input (1×704×544)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ ENCODER                                              │
│  Level 1: Conv(1→32) → Conv(32→32) → MaxPool        │
│  Level 2: Conv(32→64) → Conv(64→64) → MaxPool       │
│  Level 3: Conv(64→128) → Conv(128→128) → MaxPool    │
│  Level 4: Conv(128→256) → Conv(256→256) → MaxPool   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ BOTTLE NECK                                           │
│  Conv(256→512) → Conv(512→512)                       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ DECODER                                              │
│  Level 4: UpConv(512→256) → Concat → Conv(512→256)  │
│  Level 3: UpConv(256→128) → Concat → Conv(256→128)  │
│  Level 2: UpConv(128→64) → Concat → Conv(128→64)    │
│  Level 1: UpConv(64→32) → Concat → Conv(64→32)      │
└─────────────────────────────────────────────────────┘
    │
    ▼
Output Conv(32→1) → Sigmoid → Segmentation Mask (1×704×544)
```

## Implementation

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.downs.append(self._double_conv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._double_conv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self._double_conv(feature * 2, feature))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([skip, x], dim=1)  # Skip connection
            x = self.ups[idx + 1](x)  # Double conv
        
        return torch.sigmoid(self.final_conv(x))
```

## Loss Function

Combined Binary Cross-Entropy and Dice loss:

```python
class BCEDiceLoss(nn.Module):
    def __init__(self, lambda_dice=1.0):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.bce = nn.BCELoss()
    
    def dice_loss(self, pred, target, smooth=1.0):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return bce + self.lambda_dice * dice
```

## Data Preprocessing

```python
from PIL import Image
import numpy as np

class LIVECellDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
    
    def __getitem__(self, idx):
        # Load image and mask
        img = np.array(Image.open(self.image_paths[idx]).convert('L'), dtype=np.float32)
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'), dtype=np.float32)
        
        # Pad to power-of-2 compatible size (704×520 → 704×544)
        img = np.pad(img, ((0, 0), (0, 24)), mode='reflect')
        mask = np.pad(mask, ((0, 0), (0, 24)), mode='reflect')
        
        # Normalize
        img = img / 255.0
        mask = (mask > 127).astype(np.float32)
        
        # Data augmentation
        if self.augment:
            img, mask = self._augment(img, mask)
        
        # Convert to tensors
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        return img, mask
    
    def _augment(self, img, mask):
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        
        # Random vertical flip
        if np.random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        
        # Random rotation (±15°)
        angle = np.random.uniform(-15, 15)
        from scipy.ndimage import rotate
        img = rotate(img, angle, reshape=False, mode='reflect')
        mask = rotate(mask, angle, reshape=False, mode='constant')
        
        # Random Gaussian noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img, mask
    
    def __len__(self):
        return len(self.image_paths)
```

## Training Protocol

```python
def train_unet(model, train_loader, val_loader, epochs=100, lr=1e-4, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = BCEDiceLoss(lambda_dice=1.0)
    
    best_val_dice = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_dice += dice_coefficient(outputs, masks).item()
        
        val_dice /= len(val_loader)
        scheduler.step()
        
        # Early stopping
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, Val Dice={val_dice:.4f}")
    
    return model
```

## 5-Fold Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

def cross_validate(image_paths, mask_paths, n_folds=5):
    # Stratify by cell line
    cell_lines = [get_cell_line(p) for p in image_paths]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, cell_lines)):
        print(f"\n=== Fold {fold+1}/{n_folds} ===")
        
        train_dataset = LIVECellDataset(
            [image_paths[i] for i in train_idx],
            [mask_paths[i] for i in train_idx],
            augment=True
        )
        val_dataset = LIVECellDataset(
            [image_paths[i] for i in val_idx],
            [mask_paths[i] for i in val_idx],
            augment=False
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        model = UNet(in_channels=1, out_channels=1).to(device)
        model = train_unet(model, train_loader, val_loader)
        
        # Evaluate
        dice = evaluate(model, val_loader)
        fold_results.append(dice)
        print(f"Fold {fold+1} Dice: {dice:.4f}")
    
    print(f"\nMean Dice: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    return fold_results
```

## Results

| Cell Line | Dice (Raw) | Dice (Filtered) | Improvement |
|-----------|-----------|-----------------|-------------|
| A172 | 0.72 | 0.78 | +0.06 |
| BT474 | 0.65 | 0.69 | +0.04 |
| BV2 | 0.58 | 0.63 | +0.05 |
| Huh7 | 0.68 | 0.72 | +0.04 |
| MCF7 | 0.71 | 0.76 | +0.05 |
| SHSY5Y | 0.78 | 0.81 | +0.03 |
| SKOV3 | 0.74 | 0.79 | +0.05 |
| SkBr3 | 0.75 | 0.78 | +0.03 |
| **Mean** | **0.70** | **0.75** | **+0.05** |

## Key Implementation Details

1. **Padding**: Images are padded from 704×520 to 704×544 for power-of-2 compatibility with the U-Net architecture.

2. **Skip connections**: The encoder features are concatenated with decoder features, providing both high-level semantics and low-level spatial details.

3. **Combined loss**: BCE+Dice loss provides both pixel-level accuracy (BCE) and region-level overlap (Dice).

4. **Early stopping**: Training stops if validation Dice does not improve for 20 consecutive epochs.

5. **Data augmentation**: Random flips, rotations, and noise improve generalization.

## Source Code

- Model: `src/ws3_unet.py`
- Training: `src/phase3_segmentation.py`
- Dataset: `src/common.py`
