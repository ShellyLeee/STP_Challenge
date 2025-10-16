# UNet: RNA → Protein (Spatial Regression)

A deep learning project for predicting spatial protein expression from RNA data using U-Net architecture.

## Overview

This project implements a U-Net model to predict protein spatial distribution from RNA expression data. The model performs:
- Grid rasterization from `array_row/array_col` (0-418) to 419×419
- RNA preprocessing with normalization and dimensionality reduction
- Protein expression prediction with spatial context
- 90/10 spatial split for training and validation

## Features

- **Data Processing**: Automatic preprocessing pipeline with TruncatedSVD
- **Model Architecture**: U-Net with configurable depth and channels
- **Training**: Masked loss for tissue-only regions
- **Evaluation**: Comprehensive metrics including Spearman and Pearson correlations
- **Visualization**: Per-protein correlation analysis and spatial predictions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/UNet_RNA_Protein.git
cd UNet_RNA_Protein

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- scanpy
- scikit-learn
- numpy
- matplotlib
- tqdm

## Quick Start

### 1. Configuration

Edit `configs/config.yaml` to set your parameters:

```yaml
data:
  rna_h5ad: "train_rna.h5ad"
  pro_h5ad: "train_pro.h5ad"
  
model:
  k_pca: 128
  base_channels: 48
  
training:
  batch_size: 8
  epochs: 10
  learning_rate: 0.002
```

### 2. Training

```bash
python scripts/train.py --config configs/config.yaml
```

### 3. Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/unet_best_mse.pt
```

### 4. Prediction

```bash
python scripts/predict.py --checkpoint checkpoints/unet_best_mse.pt --output predictions.npy
```

## Project Structure

```
UNet_RNA_Protein/
├── configs/          # Configuration files
├── data/             # Dataset classes and loaders
├── models/           # Model architectures and losses
├── utils/            # Utility functions
├── scripts/          # Training and evaluation scripts
├── notebooks/        # Jupyter notebooks for exploration
└── checkpoints/      # Saved model weights
```

## Model Architecture

The U-Net consists of:
- **Encoder**: 4 downsampling blocks (MaxPool + DoubleConv)
- **Bottleneck**: Feature extraction at lowest resolution
- **Decoder**: 4 upsampling blocks (ConvTranspose + DoubleConv)
- **Skip Connections**: Concatenation from encoder to decoder

## Results

With default configuration (K_PCA=128, PATCH_SIZE=32, STRIDE=16):
- Mean Spearman correlation: ~0.24 on validation set
- Mean Pearson correlation: ~0.19 on validation set

Top performing proteins:
1. TOX: Spearman=0.460
2. PDGFR: Spearman=0.457
3. Granzyme B: Spearman=0.454

## Tips

- Start with `K_PCA=128`, `PATCH_SIZE=32`, `STRIDE=16`
- For more VRAM, try `K_PCA=256` or `PATCH_SIZE=64`
- Monitor both MSE and correlation metrics during training
- Use tiled inference with overlap for full-resolution predictions

## Citation

If you use this code, please cite:

```bibtex
@misc{unet_rna_protein,
  author = {Your Name},
  title = {UNet for RNA to Protein Spatial Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/UNet_RNA_Protein}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com]