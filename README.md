# VMD-DCFF-IF
Official implementation of VMD-DCFF-IF. This model features a dual-channel architecture using 2D-CNN and STGCN to extract spatio-temporal features from VMD-decomposed influenza time series. It achieves superior forecasting accuracy through deep multi-dimensional feature fusion using a BiGRU-BiLSTM framework.

# VMD-DCFF-IF: Influenza Forecasting via Dual-Channel Feature Fusion

This repository provides the official implementation of the paper: **"Influenza Forecasting method based on dual-channel feature fusion of VMD decomposition"**.

## 📖 Overview
Accurate influenza forecasting is critical for public health resource allocation. This project proposes **VMD-DCFF-IF**, a deep learning framework that handles the high nonlinearity and sparse sampling of influenza time series data.

### ✨ Key Features
* **VMD Decomposition**: Decomposes non-stationary influenza sequences into multiple Intrinsic Mode Functions (IMFs) to reduce nonlinear coupling.
* **Dual-Channel Architecture**: Parallel extraction of features using a 2D Attention-based CNN (temporal path) and a Spatio-Temporal Graph Convolutional Network (STGCN, spatial path).
* **Deep Feature Fusion**: Employs a **BiGRU-BiLSTM** (bi-GLSTM) model to dynamically fuse multi-dimensional features for final prediction.
* **High Accuracy**: Outperforms SOTA models like GAST, SAIFlu-Net, and MTS-LSTM on national datasets from 2013 to 2023.

---

## 📂 Project Structure
```text
VMD-DCFF-IF/
├── data/                   # Place your influenza datasets here
├── scripts/
│   ├── data_split.py       # Sequence cutting and normalization
│   ├── merge_features.py   # Merging GADF images, sequences, and graph matrices
│   └── evaluate.py         # Performance metrics (sMAPE, MASE)
├── models/
│   └── main.py             # Model definition, training, and prediction
├── requirements.txt        # Environment dependencies
└── README.md
```
---

## 🏗️ Model Architecture
The model consists of four main stages:
1. **Data Preprocessing**: Handling missing values and VMD-based decomposition.
2. **Component Reconstruction**: Encoding IMFs into 2D images via GADF and constructing correlation graphs via Spearman coefficients.
3. **Dual-Channel Extraction**:
    * **Temporal Channel**: Modified ForCNN with SE-Net attention.
    * **Spatial Channel**: STGCN focusing on inter-component correlations.
4. **Forecasting**: Sequence modeling using bidirectional units (**BiGRU** & **BiLSTM**).

---

## 🛠️ Requirements
The experimental environment used in the study is as follows:
* **OS**: Windows 11
* **CPU**: Intel i7-12700
* **GPU**: NVIDIA RTX 3060 (12GB)
* **Framework**: TensorFlow 2.9 (with Keras API)

---

## 📊 Dataset
The dataset is sourced from the Chinese National Influenza Center (CNIC), covering weekly influenza cases across China from 2013 to 2023.
* **Official Data Source**: [Chinese National Influenza Center (CNIC)](https://ivdc.chinacdc.cn/cnic/)
* **Data Splitting**: 80% Training, 20% Validation.
* **Testing**: The final 12 observations are withheld for independent evaluation.

---

## 🚀 Getting Started
1. **Environment Setup**:
```Bash
git clone [https://github.com/YourUsername/VMD-DCFF-IF.git](https://github.com/YourUsername/VMD-DCFF-IF.git)
cd VMD-DCFF-IF
pip install -r requirements.txt
```
2. **Data Preparation**:
   1. **Run data splitting**:
```Bash
python scripts/data_split.py
```
   2. **Merge multi-modal features**:
```Bash
python scripts/merge_features.py
```
3. **Training & Prediction**:
Modify the configuration in models/main.py and run:
```Bash
python models/train.py --batch_size 128 --learning_rate 1e-5 --loss MAE
(Note: Key hyperparameters are configured as per Table 2 in the paper.)
```

---

## 🛠️ Requirements

**Framework**: TensorFlow 2.9.0 (Keras API)
**Dependencies**: numpy, pandas, matplotlib, scipy, vmdpy, scikit-learn, pyts, axial-attention

---

## Python Dependencies
```bash
pip install tensorflow==2.9.0 numpy matplotlib pandas scipy vmdpy
