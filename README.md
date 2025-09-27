# Gene Expression Prediction

This method is used to predict gene expression from histological cell images. It is based on Vision Transformer (ViT) models to extract features from cell images and then uses these features to predict the gene expression profiles of cells.

## Directory Structure

├── main    // Main program folder \
│   ├── preprocess_image.py    // Python script for image preprocessing \
│   ├── rescale.py    // Python script for image rescaling \
│   ├── extract_features.py    // Python script for extracting features from cell images \
│   ├── merge_features.py    // Python script for merging features \
│   ├── train_model.py    // Python script for training the prediction model \
│   ├── predict_expression.py    // Python script for predicting gene expression \
│   ├── model_utils.py    // Utility functions related to HIPT model \
│   ├── utils.py    // General utility functions \
│   ├── vision_transformer.py    // Vision Transformer implementation \
│   ├── README.md    // Documentation \
│   ├── requirements.txt    // Project dependencies \
│   └── work    // Various analysis and visualization scripts \
│       ├── Before train    // Scripts before training \
│       │   ├── Before train.py    // Analysis script before training \
│       │   ├── Before train2.py    // Analysis script before training 2 \
│       │   ├── Before train3.py    // Analysis script before training 3 \
│       │   ├── Before train4.py    // Analysis script before training 4 \
│       │   ├── Validation dataset.py    // Validation dataset analysis script \
│       │   └── Validation dataset 2.py    // Validation dataset analysis script 2 \
│       ├── IHC \
│       │   ├── IHC-A-cell segmentation.py    // IHC-A region cell segmentation script \
│       │   ├── IHC-A-expression.py    // IHC-A region gene expression script \
│       │   ├── IHC-B-cell segmentation.py    // IHC-B region cell segmentation script \
│       │   └── IHC-B-expression.py    // IHC-B region gene expression script \
│       └── ...

## Usage Instructions

### 1. Preprocess and Rescale Images
```bash
python preprocess_image.py \
    --valid_path /path/to/images/ \
    --train_path /path/to/images/ \
    --output_path /path/to/images/
```

```bash
python rescale.py \
    --prefix /path/to/images/
```

### 2. Extract Features
```bash
python extract_features.py \
    --prefix /path/to/images/ \
    --output_file features.pkl
```

### 3. Train Model
```bash
python train_model.py \
    --feature_file features.pkl \
    --h5ad_file expression_data.h5ad \
    --output_dir ./output/
```

### 4. Predict Gene Expression
```bash
python predict_expression.py \
    --feature_file features.pkl \
    --model_file ./output/model.pkl \
    --output_file prediction.h5ad
```

## Methodology

### Image Processing
To facilitate the processing of histological images with different resolutions, each image is first rescaled so that the size of each pixel is 0.5 × 0.5 μm². This ensures a 16 × 16-pixel tile corresponds to an area of 8 × 8 μm², which is about the size of a single cell.

### Feature Extraction
For each cell, a 256 × 256-pixel image tile centered on its spatial location is extracted and used as input to a pre-trained Vision Transformer (ViT) model. This model, specifically a ViT-256/16 pre-trained with the self-supervised DINO framework, processes the image by dividing it into a sequence of 16 × 16 patches (tokens).

The final hidden state of the dedicated [CLS] token serves as a holistic representation of the entire image tile, which is used as the global feature vector. To represent fine-grained cellular morphology, the local feature is derived from the patch tokens whose corresponding 16 × 16 regions overlap with the cell nucleus. These feature vectors are aggregated by average pooling to form a unified vector. The local and global features are then concatenated to form a comprehensive histology feature vector.

### Prediction Model
A dual-output neural network model is designed to simultaneously determine whether a gene is expressed and predict its expression level. Given the class imbalance problem in transcriptomics datasets, class weights are introduced into the loss function to enhance the model's capacity. The network architecture consists of 4 hidden layers with 512, 512, 1024, 1024 nodes, using leaky ReLU activation functions, Batch Normalization, and Dropout (rate=0.1).

### Evaluation Metrics
Prediction accuracy is assessed using the Root Mean Square Error (RMSE) and the Pearson Correlation Coefficient (PCC) between predicted and ground truth gene expression values.