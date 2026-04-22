# 🌿 Plant Disease Classification using EfficientNetB0

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-EfficientNetB0-D00000?style=flat-square&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~97%25-2ea44f?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

A research-grade deep learning pipeline for multi-class plant disease detection — built end-to-end with careful attention to class imbalance, data leakage prevention, and a deliberate two-phase fine-tuning strategy.

---

## What This Project Does

This isn't a plug-and-play notebook. I designed the full pipeline from scratch: data preparation, stratified splitting, class imbalance handling, and a two-phase training strategy that actually generalizes rather than memorizing the training set.

The backbone is **EfficientNetB0** pretrained on ImageNet. Instead of training from zero (which would demand far more data and compute), I leveraged its existing feature representations and fine-tuned the upper layers on the plant disease dataset. The result consistently hits ~97% test accuracy — and it took deliberate engineering decisions to get there.

---

## Key Features

- ✅ **Stratified 80/10/10 split** — done per-class, no data leakage between subsets  
- ✅ **Class imbalance handling** — computed class weights so rare disease categories aren't ignored  
- ✅ **Two-phase training** — frozen head first, then careful fine-tuning of upper EfficientNet layers  
- ✅ **Label smoothing (0.1)** — prevents overconfident predictions on noisy labels  
- ✅ **Smart callbacks** — EarlyStopping + ReduceLROnPlateau, not hardcoded epoch counts  
- ✅ **Controlled augmentation** — rotation, flip, zoom, brightness adjustments for robustness  
- ✅ **Dual format save** — `.keras` (native) and `.h5` (legacy) for deployment flexibility  

---

## Dataset Structure

### Data Acquisition and Compilation

A significant challenge in this project was the lack of a comprehensive, unified crop dataset. Existing datasets were often too specific, focusing exclusively on either fruits or cereals. While larger datasets like **DLCPD-25** exist, they were not immediately accessible within the project timeline.

To overcome this, I compiled a custom unified dataset by merging two prominent sources:
- **PlantVillage Dataset**: Contributing 38 classes of various fruits and vegetables.
- **Cereal Dataset**: Contributing 16 classes specifically for grains and cereal crops.

This resulting custom dataset provides a more holistic coverage of plant diseases across 54 total classes.

**Input (before split)**
```
FINAL_UNIFIED_DATASET/
 ├── Apple___Apple_scab/
 ├── Apple___Black_rot/
 ├── Corn___Northern_Leaf_Blight/
 └── ... (38 disease classes)
```

**Generated (after split)**
```
FINAL_SPLIT_DATASET/
 ├── train/
 │   ├── Apple___Apple_scab/
 │   └── ...
 ├── val/
 │   └── ...
 └── test/
     └── ...
```

Each class is split **independently** — this is a stratified split, not a random global shuffle. Every disease label is proportionally represented in all three subsets.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 (RGB) |
| Batch size | 32 |
| Backbone | EfficientNetB0 (ImageNet weights) |
| Phase 1 optimizer | Adam — lr = `1e-3` |
| Phase 2 optimizer | Adam — lr = `1e-5` |
| Loss function | Categorical Crossentropy + label smoothing (0.1) |
| Callbacks | EarlyStopping, ReduceLROnPlateau |
| Augmentation | Rotation, flip, zoom, brightness |

---

## Two-Phase Training Strategy

A single training run with all layers unfrozen tends to either underfit or destroy the pretrained representations. The two-phase approach solves this deliberately.

### Phase 1 — Feature Extraction
- EfficientNetB0 base model is fully **frozen**
- Only the classification head trains
- Higher learning rate (`1e-3`) to quickly adapt the head
- Purpose: stabilize task-specific representations before touching the backbone

### Phase 2 — Fine-Tuning
- Last **~50 layers** of EfficientNetB0 unfrozen (not the full backbone)
- Lower learning rate (`1e-5`) to avoid catastrophic forgetting
- EarlyStopping monitors `val_loss` and halts when generalization stops improving
- ReduceLROnPlateau kicks in if the model plateaus

> The decision to unfreeze only the upper 50 layers is intentional. Lower EfficientNet layers capture universal features (edges, textures) that transfer well across domains — there's no benefit to retraining them on a domain-specific dataset.

---

## Core Fine-Tuning Logic

```python
# Phase 1 — train only the classification head
base_model.trainable = False
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy')
model.fit(train_gen, validation_data=val_gen,
          class_weight=class_weights, epochs=20)

# Phase 2 — unfreeze upper layers and fine-tune carefully
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy')
model.fit(train_gen, validation_data=val_gen,
          callbacks=[early_stop, reduce_lr],
          class_weight=class_weights, epochs=50)
```

---

## Evaluation

Model performance is reported on a **held-out test set** — data the model never saw during training or validation. Metrics include:

- Overall test accuracy
- Per-class precision, recall, and F1-score (full classification report)

Reporting on a proper test set matters. Training accuracy is meaningless. Val accuracy can still be optimistic if you tune hyperparameters against it. Test set gives the honest number.

---

## Model Output

```
plant_disease_effnetb0.keras   ← recommended (native Keras format)
plant_disease_effnetb0.h5      ← legacy compatibility
```

Both formats can be loaded directly for inference, further fine-tuning, or deployment.

---

## Planned Extensions

This training pipeline is the foundation. The goal is to extend it into a full decision-support tool:

- [ ] **Grad-CAM visualizations** — highlight which leaf regions drove the prediction
- [ ] **Confidence-based flagging** — surface low-confidence predictions for expert review
- [ ] **Disease explanations** — map class labels to structured agronomic information
- [ ] **Image quality checks** — detect blurry or poorly lit inputs before inference
- [ ] **Streamlit / Flask deployment** — user-facing interface for field use

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~97% |
| Evaluation Set | Held-out test split (10%) |
| Per-class metrics | Precision, Recall, F1 (see classification report) |

---

> **Disclaimer:** This system is built for research and academic decision-support purposes. It should not replace expert agricultural or plant pathology advice in real production environments.

---

**Author:** **Karan Shakya**  
B.Tech CSE (AI & ML) · NIET Greater Noida · AKTU  
GitHub: [@theconquero-r](https://github.com/theconquero-r)
