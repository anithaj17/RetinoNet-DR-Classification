# RetinoNet: Multi-Scale Deep Learning Framework for Diabetic Retinopathy Classification

## 1. Title

**RetinoNet: Multi-Scale Deep Learning Framework for Accurate Stage-wise Classification of Diabetic Retinopathy (DR)**

## 2. Description

RetinoNet is a deep learning-based framework designed to accurately classify the severity of diabetic retinopathy (DR) using fundus images. The framework leverages EfficientNet-B0 as the backbone model, integrated with a Feature Pyramid Network (FPN) to extract multi-scale features for precise classification of DR severity levels. The implementation includes data preprocessing, model training, evaluation, and inference scripts, making it suitable for clinical and research applications.

## 3. Dataset Information

- **Source:** Messidor Public Fundus Image Dataset  
- **Curated Version:** A preprocessed dataset with four DR severity levels:
  - Class 0: No DR (120 files)
  - Class 1: Mild DR (101 files)
  - Class 2: Moderate DR (120 files)
  - Class 3: Severe DR (120 files)
- **Data Structure:** Organized as `/kaggle/dataset3/DATASET3/` with subdirectories for each class
- **Access**: View the dataset on Kaggle: [Kaggle Dataset (view-only)](https://www.kaggle.com/datasets/anithajaikumar/dataset3).

## 4. Code Information

- **Preprocessing:** Background masking (GrabCut), histogram equalization, and denoising  
- **Model Architecture:** EfficientNet-B0 as backbone, FPN for multi-scale feature extraction  
- **Training Scripts:** Includes training, validation, and evaluation scripts  
- **Evaluation:** Accuracy reported as 96.77%  
- **Inference:** Model inference script to predict DR severity on new fundus images

## 5. Usage Instructions

> **Important:** This repository can be run on both Kaggle and Jupyter Notebook.

### Step 1: Kaggle Setup

- Go to Kaggle and add the dataset to the notebook by attaching the dataset located at `anithajaikumar/dataset3`.
- The dataset will be available at:  
  `/kaggle/input/d/anithajaikumar/dataset3/DATASET3/`
- Ensure that the directory structure is maintained as:

    ```
    /kaggle/input/d/anithajaikumar/dataset3/DATASET3/Class0
    /kaggle/input/d/anithajaikumar/dataset3/DATASET3/Class1
    /kaggle/input/d/anithajaikumar/dataset3/DATASET3/Class2
    /kaggle/input/d/anithajaikumar/dataset3/DATASET3/Class3
    ```

### Step 2: Run the Notebook on Kaggle

- Open the `retinonet-v2.ipynb` notebook.
- Execute the cells in order to perform data preprocessing, model training, and evaluation.

### Step 3: Evaluate Model

- Run the final cell to evaluate the model.
- The output will display accuracy and loss metrics.

### Step 4: Jupyter Notebook Setup

- Alternatively, download the repository and run it locally using Jupyter Notebook.
- Ensure that the dataset is placed in the specified path as `/DATASET3/` with the correct class folder structure.

### Step 5: Inference

- Adjust the dataset path in the notebook to run inference on new data.

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- NumPy, Pandas
- OpenCV, scikit-image
- Matplotlib

**Install all dependencies using:**
```bash
pip install tensorflow numpy pandas opencv-python scikit-image matplotlib
```
## 7. Methodology

### Data Preprocessing
-  Background masking using **GrabCut**
-  Histogram equalization for contrast enhancement
-  Denoising for noise reduction

### Model Design
-  **EfficientNet-B0** backbone
-  **FPN** for multi-scale feature extraction
-  Dense layer with **softmax** activation for multi-class classification

### Training
-  **Epochs:** 100  
-  **Batch Size:** 32  
-  **Optimizer:** Adam  
-  **Loss:** Categorical Cross-Entropy

### Evaluation
-  Accuracy and loss metrics  
-  Confusion matrix analysis

## 8. Citations

Not applicable

## 9. License

This project is licensed under the **MIT License**. See the `LICENSE.txt` file for details.
