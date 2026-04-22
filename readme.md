# Plant Disease Classification using EfficientNetB0

## Project Overview
This project implements a deep learning–based plant disease classification system using **EfficientNetB0** and TensorFlow/Keras. I designed the complete training pipeline to handle a large, multi-class plant disease dataset, perform proper train/validation/test splitting, address class imbalance, and achieve high classification accuracy using transfer learning and fine-tuning.

The goal of this work is to build a **research-grade image classification model** that can later be extended into a user-facing system providing disease confidence, explanations, and decision support for agriculture applications.

---

## Key Features
- Uses **EfficientNetB0 (ImageNet pretrained)** for strong feature extraction
- Clean and standardized class folder names
- Automatic **train / validation / test split (80 / 10 / 10)**
- Handles **class imbalance** using computed class weights
- Applies **controlled data augmentation** for robustness
- Two-phase training:
  - Frozen base model training
  - Fine-tuning of upper EfficientNet layers
- Uses **EarlyStopping** and **ReduceLROnPlateau** to prevent overfitting
- Saves the final model in both `.keras` and `.h5` formats

---

## Dataset Structure

### Data Acquisition and Compilation
A significant challenge in this project was the lack of a comprehensive, unified crop dataset. Existing datasets were often too specific, focusing exclusively on either fruits or cereals. While larger datasets like **DLCPD-25** exist, they were not immediately accessible within the project timeline.

To overcome this, I compiled a custom unified dataset by merging two prominent sources:
- **PlantVillage Dataset**: Contributing 38 classes of various fruits and vegetables.
- **Cereal Dataset**: Contributing 16 classes specifically for grains and cereal crops.

This resulting custom dataset provides a more holistic coverage of plant diseases across 54 total classes.

### Input Dataset (before split)
```
FINAL_UNIFIED_DATASET/
 ├── Apple___Apple_scab/
 ├── Apple___Black_rot/
 ├── ...
```

### Generated Dataset (after split)
```
FINAL_SPLIT_DATASET/
 ├── train/
 │   ├── class_1/
 │   ├── class_2/
 │   └── ...
 ├── val/
 │   └── ...
 └── test/
     └── ...
```

Each class is split independently to ensure no data leakage between training, validation, and testing.

---

## Training Configuration

- Image size: **224 × 224**
- Batch size: **32**
- Backbone: **EfficientNetB0**
- Optimizer:
  - Adam (1e-3) for initial training
  - Adam (1e-5) for fine-tuning
- Loss function:
  - Categorical Crossentropy with **label smoothing (0.1)**

---

## Training Strategy

### Phase 1: Feature Extraction
- EfficientNetB0 base model frozen
- Only classification head trained
- Purpose: stabilize high-level representations

### Phase 2: Fine-Tuning
- Last ~50 layers of EfficientNetB0 unfrozen
- Lower learning rate applied
- EarlyStopping and learning rate reduction enabled

This strategy improves generalization while avoiding catastrophic forgetting.

---

## Evaluation

- Model performance is evaluated on a **held-out test set**
- Final metrics include:
  - Test accuracy
  - Per-class precision, recall, and F1-score (classification report)

This ensures that reported accuracy reflects real generalization performance, not training bias.

---

## Model Saving

The trained model is saved in two formats:
- `plant_disease_effnetb0.keras` (recommended, native Keras format)
- `plant_disease_effnetb0.h5` (legacy compatibility)

These files can be directly used for inference, deployment, or further fine-tuning.

---

## Future Extensions
This training pipeline is designed to be extended with:
- Human-readable disease explanations
- Confidence-based predictions
- Image quality and visibility analysis
- Grad-CAM visualizations
- Deployment using Streamlit or Flask

---

## Disclaimer
This system is intended for **research and decision support purposes only** and should not replace expert agricultural or plant pathology advice.

---

## Author
**Karan Shakya**
Developed and trained as part of an academic deep learning project focused on plant disease detection using convolutional neural networks.

~~Karan Shakya