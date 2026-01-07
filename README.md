# Satellite Imagery–Based Property Valuation  
### Residual Multimodal Learning Approach

## Overview
This project predicts residential property prices using a **hierarchical multimodal learning strategy** that combines tabular housing data and satellite imagery.

Instead of treating satellite images as a primary predictor, the model uses **residual learning**, where:
- Tabular features predict the base property value
- Satellite imagery learns only the remaining unexplained variation

This design reflects the real-world role of visual context in valuation — **corrective rather than dominant**.

---

## Problem Statement
Given:
- Structured housing attributes (tabular data)
- Satellite images for each property location

Goal:
- Predict property prices for an unseen test set
- Output a `submission.csv` with : id, predicted_price



No ground-truth prices are available for the test set.

---

## Dataset Description

### Training Data
- ~16,209 residential properties
- Target variable: `price`
- One satellite image per property

### Test Data
- ~5,404 properties
- Same tabular features
- No price labels
- Satellite images downloaded using identical logic

### Satellite Imagery
- Source: ESRI World Imagery
- Downloaded using latitude & longitude
- Same resolution and preprocessing for train and test

---

## Tabular Features

### Raw Features
- bedrooms, bathrooms
- sqft_living, sqft_lot
- floors
- waterfront, view, condition, grade
- sqft_above, sqft_basement
- latitude, longitude
- sqft_living15, sqft_lot15

### Engineered Features
- house_age
- is_renovated
- total_sqft
- room_density

Dropped columns:
- `id`
- `date`

**Total tabular features:** 19

---

## Exploratory Data Analysis (EDA)

Key EDA findings:
- Strong monotonic relationships between structural features and price
- Tabular data explains the majority of price variance
- Properties with similar tabular attributes can still exhibit price differences
- Satellite images reveal environmental context (greenery, roads, water) not captured in tabular data
- Raw satellite imagery is noisy and inconsistent, motivating feature extraction over end-to-end CNN training

These observations directly motivated the residual learning framework.

---

## Image Feature Engineering

### CNN Feature Extraction
- Pretrained ResNet50
- Input size: 224 × 224
- Output: 2048-dimensional embeddings
- Used only as a feature extractor (no fine-tuning)

### Dimensionality Reduction
- PCA with 64 components
- PCA fitted on training data only
- Same transformation applied to test data

### Interpretable Visual Features
To reduce black-box behavior, additional handcrafted features were extracted:
- green_ratio (vegetation coverage)
- blue_ratio (water bodies)
- edge_density (road/building density)
- brightness (surface intensity)

**Final image feature vector size:** 68  
(64 PCA components + 4 interpretable features)

---

## Modeling Strategy: Residual Learning

### Why Residual Learning?
- Tabular features already capture most valuation drivers
- Satellite imagery is a weak standalone predictor
- Direct feature concatenation risks overfitting

Residual learning enforces a **hierarchy of information**:
- Tabular model predicts base price
- Image model learns only residual corrections

---

## Model Architecture

### Stage 1: Tabular Model
- Algorithm: XGBoost Regressor
- Input: Tabular features
- Output: Base price prediction

### Stage 2: Image Residual Model
- Algorithm: XGBoost Regressor
- Input: Image features (68D)
- Target: Residuals (actual price − tabular prediction)

### Final Prediction

Final Price = Tabular Prediction + Image Residual Prediction


---

## Validation Results

| Model | R² (Validation) |
|------|----------------|
| Tabular Only | ~0.890 |
| Tabular + Satellite (Residual) | ~0.891 |

The modest improvement confirms that satellite imagery contributes **complementary contextual information** rather than dominating the prediction.

---

## Inference on Test Data
- Identical feature engineering applied
- Same CNN and PCA transformation used (no refitting)
- Residual logic preserved
- Final predictions saved as `submission.csv`

---

## Project Structure

Satellite-project-23112026/
│
├── README.md
├── data_fetcher.py
├── preprocessing.ipynb
├── model_training.ipynb
├── inference.ipynb
│
├── models/
│ ├── tabular_xgb_fe.pkl
│ ├── image_residual_xgb_fe.pkl
│ └── image_pca.joblib
│
├── image_features/
│ ├── resnet50_features.npy
│ ├── resnet50_pca64.npy
│ ├── interpretable_features.npy
│ └── X_image_test.npy
│
└── submission.csv


(Note: Large binary files may be excluded due to GitHub size limits.)

---

## Limitations
- Satellite imagery resolution limits fine-grained visual cues
- High-rise apartments are not well represented from top-down imagery
- Satellite data may lag behind recent urban development

---

## Conclusion
This project demonstrates that satellite imagery is most effective when used as a **corrective signal** rather than a primary predictor in property valuation. The residual multimodal framework improves robustness, interpretability, and realism of predictions.


## Pipeline Overview


Tabular Features
↓
XGBoost (Base Price Model)
↓
Base Price

Satellite Images
↓
CNN (ResNet50)
↓
PCA + Visual Signals
↓
XGBoost (Residual Model)
↓
Residual Correction

Final Price = Base Price + Residual Correction

