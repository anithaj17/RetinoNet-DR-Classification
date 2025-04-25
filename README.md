# RetinoNet: EfficientNet and FPN-Based Framework for Diabetic Retinopathy Classification

## 📖 Description
**RetinoNet** is a deep learning framework designed for the accurate stage-wise classification of diabetic retinopathy (DR) using fundus images. Leveraging **EfficientNet-B0** as the backbone and a **Feature Pyramid Network (FPN)** for multi-scale feature extraction, RetinoNet ensures scalability, interpretability, and preservation of critical image details, making it suitable for clinical applications. The model is trained to classify DR into four severity levels, providing a robust tool for early diagnosis and monitoring.

This repository contains the complete implementation of RetinoNet, including preprocessing, model architecture, training, and evaluation scripts. The model achieves high accuracy on the test dataset, as demonstrated in the provided notebook.

## 📂 Dataset Information
- **Source**: Messidor public fundus image dataset.
- **Curated Version**: A custom-preprocessed dataset with images categorized into four diabetic retinopathy severity levels (Class 0, Class 1, Class 2, Class 3).
- **Location**: The dataset is stored in the `/kaggle/dataset3/DATASET3/` directory, with subdirectories for each class.
- **Access**: View the dataset on Kaggle: [Kaggle Dataset (view-only)](https://www.kaggle.com/datasets/anithajaikumar/dataset3).

## ⚙️ Code Information
This repository includes:
- Image preprocessing: Background masking (GrabCut), histogram equalization, denoising
- EfficientNetB0 as the backbone model
- Feature Pyramid Network for multi-scale feature extraction
- Global Average Pooling to prevent overfitting
- Dense softmax layer for 4-class classification

## 🚀 Usage Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd RetinoNet
   ```
2. **Prepare the Dataset**:
   - Download the curated dataset from the Kaggle link provided above.
   - Place the dataset in the `/kaggle/dataset3/DATASET3/` directory, ensuring subdirectories for `class 0`, `class 1`, `class 2`, and `class 3` are maintained.
   - Alternatively, modify the dataset path in the notebook to point to your local dataset directory.
3. **Install Dependencies**:
   Install the required Python libraries by running:
   ```bash
   pip install tensorflow numpy pandas opencv-python matplotlib scikit-image
   ```
4. **Run the Notebook**:
   - Open the `retinonet-v2.ipynb` notebook in Jupyter or Kaggle.
   - Execute the cells sequentially to preprocess the data, build the model, train it, and evaluate its performance.
5. **Evaluate the Model**:
   - The notebook includes a test evaluation step that outputs the test accuracy and loss.
   - Use the trained model for inference on new fundus images by loading the model and preprocessing the images as described.

## 📋 Requirements
The following Python libraries are required to run the code:
- `tensorflow` (for model building and training)
- `numpy` (for numerical operations)
- `pandas` (for data handling, if applicable)
- `opencv-python` (for image preprocessing, e.g., GrabCut, histogram equalization)
- `matplotlib` (for visualization, if used)
- `scikit-image` (for additional image processing, if used)

Install these dependencies using:
```bash
pip install tensorflow numpy pandas opencv-python matplotlib scikit-image
```

Additionally, a GPU (e.g., Tesla P100-PCIE-16GB, as used in the notebook) is recommended for faster training, though the code can run on CPU with reduced performance.

## 🛠️ Methodology
The development of RetinoNet involved the following steps:
1. **Data Preprocessing**:
   - Images were preprocessed using GrabCut for background masking, histogram equalization for contrast enhancement, and denoising to improve quality.
   - Images were organized into four classes based on DR severity.
2. **Dataset Preparation**:
   - The dataset was split into training, validation, and test sets using TensorFlow’s `image_dataset_from_directory`.
   - Images were resized to a consistent input size (e.g., 224x224, standard for EfficientNet-B0) and batched for training.
3. **Model Design**:
   - EfficientNet-B0 was used as the backbone, with pre-trained weights (not shown but typically from ImageNet).
   - FPN was implemented by extracting features from multiple layers (`block6a`, `block4a`, `block3a`), followed by convolution, upsampling, and concatenation to create a rich feature map.
   - A classification head with global average pooling and dense layers was added for 4-class classification.
4. **Training**:
   - The model was trained for 100 epochs, with training and validation accuracy/loss monitored.
   - The model showed signs of overfitting after early epochs (e.g., high training accuracy but fluctuating validation accuracy), but later epochs achieved high validation accuracy (up to 96.74%).
5. **Evaluation**:
   - The model was evaluated on a separate test set, achieving 96.77% accuracy and a loss of 0.1978.
   - The training history (accuracy and loss curves) can be visualized using the `history` object for further analysis.

