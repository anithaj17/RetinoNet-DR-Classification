
# RetinoNet: EfficientNet and FPN-Based Framework for Diabetic Retinopathy Classification

## 📖 Description
This repository contains the core implementation of **RetinoNet**, a deep learning-based model that uses EfficientNet-B0 and a Feature Pyramid Network (FPN) for accurate stage-wise classification of diabetic retinopathy (DR). The model is designed for clinical use cases where scalability, interpretability, and image detail preservation are critical.

## 📂 Dataset Information
- **Source**: Messidor public fundus image dataset
- **Curated Version**: Custom-preprocessed dataset with 4 DR severity levels  
  🔗 [Kaggle Dataset (view-only)](https://www.kaggle.com/code/anithajaikumar/article-4-v2-final-version)

## ⚙️ Code Overview
This repository includes:
- Image preprocessing: Background masking (GrabCut), histogram equalization, denoising
- EfficientNetB0 as the backbone model
- Feature Pyramid Network for multi-scale feature extraction
- Global Average Pooling to prevent overfitting
- Dense softmax layer for 4-class classification

## 🚀 Usage Instructions
1. Clone the repo and place your dataset as described in the paper.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
