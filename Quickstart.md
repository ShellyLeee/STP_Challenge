# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/UNet_RNA_Protein.git
cd UNet_RNA_Protein

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e .
```

## Data Preparation

Place your data files in the project root:
- `train_rna.h5ad` - RNA expression data
- `train_pro.h5ad` - Protein expression data

## Training

### 1. Configure training (optional)

Edit `configs/config.yaml` to adjust hyperparameters:

```yaml
preprocessing:
  k_pca: 128  # Reduce to 64 for faster training, increase to 256 for better quality

training:
  patch_size: 32  # Patch size
  stride: 16      # Overlap
  batch_size: 8   # Adjust based on GPU memory
  epochs: 10      # Number of epochs
```

### 2. Start training

```bash
python scripts/train.py --config configs/config.yaml
```

This will:
- Load and preprocess data
- Create train/val split
- Train the UNet model
- Save checkpoints to `checkpoints/`
- Save best model as `checkpoints/unet_best_mse.pt`

**Training output example:**
```
Epoch 01 | time=2.5s | train_loss=1962131051.74261 | val_loss=1622183242.24000
[Best↓] Saved best checkpoint: checkpoints/unet_best_mse.pt
```

## Evaluation

### Evaluate on validation set

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/unet_best_mse.pt \
    --output_dir results/
```

This will:
- Perform tiled inference on full image
- Compute Spearman and Pearson correlations
- Generate visualization plots
- Save results to `results/`

**Expected output:**
```
[Final Evaluation]
Mean Spearman across 44 proteins: 0.2446
Mean Pearson  across 44 proteins: 0.1897

Top 10 proteins by Spearman correlation:
  TOX                   Spearman= 0.460  Pearson= 0.464
  PDGFR                 Spearman= 0.457  Pearson= 0.477
  Granzyme B            Spearman= 0.454  Pearson= 0.415
```

## Prediction

### Generate predictions for new data

```bash
python scripts/predict.py \
    --checkpoint checkpoints/unet_best_mse.pt \
    --output predictions.npy \
    --save_spots
```

This will save:
- `predictions.npy` - Full spatial predictions (419, 419, 44)
- `predictions_spots.npy` - Spot-level predictions (n_spots, 44)

## Project Structure

```
UNet_RNA_Protein/
├── configs/          # Configuration files
│   └── config.yaml
├── data/             # Dataset classes
│   ├── __init__.py
│   └── dataset.py
├── models/           # Model architectures
│   ├── __init__.py
│   ├── unet.py
│   └── losses.py
├── utils/            # Utility functions
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── metrics.py
│   └── visualization.py
├── scripts/          # Training/evaluation scripts
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── checkpoints/      # Saved models
└── notebooks/        # Jupyter notebooks
```

## Tips for Better Results

### 1. Adjust K_PCA based on GPU memory

```yaml
# Low memory (~4GB GPU)
preprocessing:
  k_pca: 64

# Medium memory (~8GB GPU)  
preprocessing:
  k_pca: 128

# High memory (~16GB+ GPU)
preprocessing:
  k_pca: 256
```

### 2. Tune patch size and stride

```yaml
# Faster training, less context
training:
  patch_size: 32
  stride: 16

# Better quality, more context
training:
  patch_size: 64
  stride: 32
```

### 3. Enable preprocessing (optional)

```yaml
preprocessing:
  normalize_rna: true   # Apply normalization to RNA
  zscore_protein: true  # Apply z-score to protein
```

### 4. Try different loss functions

```yaml
loss:
  type: "huber"  # More robust to outliers
  huber_delta: 1.0
```

## Troubleshooting

### Out of Memory Error

1. Reduce `batch_size` in config
2. Reduce `k_pca` dimension
3. Reduce `patch_size`
4. Set `num_workers: 0` in config

### Slow Training

1. Increase `batch_size` if memory allows
2. Reduce `num_workers` if CPU is bottleneck
3. Use smaller `k_pca` for faster preprocessing

### Poor Results

1. Increase `epochs` (try 20-50)
2. Increase `k_pca` to capture more information
3. Try different `learning_rate` (0.001 - 0.005)
4. Enable RNA/protein preprocessing

## Next Steps

- Explore predictions in `notebooks/exploration.ipynb`
- Fine-tune hyperparameters in `configs/config.yaml`
- Visualize results using evaluation outputs
- Experiment with different model architectures

## Support

For issues or questions:
- Open an issue on GitHub
- Check the main README.md
- Contact: your.email@example.com