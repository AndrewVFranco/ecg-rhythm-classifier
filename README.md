<strong> **DO NOT DISTRIBUTE OR PUBLICLY POST SOLUTIONS TO THESE LABS. MAKE ALL FORKS OF THIS REPOSITORY WITH SOLUTION CODE PRIVATE. PLEASE REFER TO THE STUDENT CODE OF CONDUCT AND ETHICAL EXPECTATIONS FOR COLLEGE OF INFORMATION TECHNOLOGY STUDENTS FOR SPECIFICS. ** </strong>

# WESTERN GOVERNORS UNIVERSITY

## D683 – ADVANCED AI AND ML

Welcome to Advanced AI and ML!

For specific task instructions and requirements for this assessment, please refer to the course page.

# ECG Rhythm Classifier

A lightweight 1-D CNN proof-of-concept for on-device ECG rhythm classification targeting clinical telemetry environments. The model classifies single-lead (Lead II) ECG data into three rhythm categories locally, producing a rhythm interpretation and confidence score without transmitting sensitive PHI to a remote server.

---

## Rhythm Classes

| Label | Description |
|-------|-------------|
| `NORM` | Normal Sinus Rhythm |
| `AFIB` | Atrial Fibrillation |
| `1AVB` | 1st Degree Atrioventricular Block |

---

## Project Structure

```
d683-advanced-ai-and-ml/
├── model/
│   ├── notebooks/
│   │   ├── ptbxl_data_cleaning.ipynb       # Dataset filtering, lead extraction, augmentation
│   │   ├── ecg_model_training.ipynb        # Model architecture and training
│   │   └── ecg_model_optimization.ipynb    # Hyperparameter tuning and quantization
│   ├── ptb-xl/                             # PTB-XL dataset (not tracked in git)
│   ├── tuning_results/                     # Hyperparameter tuning logs
│   ├── ecg_model.keras                     # Base trained model
│   ├── final_augmented_dataset.npz         # Augmented dataset used for training
│   └── optimized_ecg_model.keras          # Quantization-optimized model (used by interface)
├── rhythm_sample/
│   ├── NSR_sample.npy                      # Normal Sinus Rhythm sample
│   ├── AFIB_sample.npy                     # Atrial Fibrillation sample
│   └── AVB_sample.npy                      # 1st Degree AVB sample
├── interface.py                            # Streamlit POC interface
├── .gitignore
└── README.md
```

---

## Dataset

**PTB-XL** — publicly available via PhysioNet

- 21,837 ten-second 12-lead ECG records from 18,885 patients sampled at 500 Hz
- This implementation uses **single-channel input (Lead II)** with input tensor shape `(5000, 1)`
- Three SCP codes selected: `NORM` (~9,514), `AFIB` (~1,514), `1AVB` (~793)
- Stratified 10-fold split: folds 1–8 for training, fold 9 for validation, fold 10 for testing
- Class imbalance addressed via four augmentation techniques:
  - Gaussian noise injection
  - Amplitude scaling
  - Time shifting
  - Baseline wander simulation

> Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data, 7*, 154. https://doi.org/10.1038/s41597-020-0495-6
>
> Wagner, P., et al. (2022). PTB-XL (version 1.0.3). *PhysioNet*. https://doi.org/10.13026/kfzx-aw45

---

## Model

- **Architecture:** 1-D Convolutional Neural Network (CNN)
- **Optimization:** Post-training quantization via TensorFlow Lite for edge device suitability
- **Metrics:** Overall accuracy 93%, Macro F1 0.81 pre quantization
- **Inference speed:** <1 second per classification


![img.png](final_confusion_matrix.png)

---

## Running the Interface

**Prerequisites**

```bash
pip install streamlit tensorflow numpy
```

**Launch**

```bash
streamlit run interface.py
```

The interface will open in your browser and allows you to:
- Select a rhythm sample (NSR, A-Fib, or 1st Degree AVB)
- View a scrolling ECG waveform visualization
- See the model's rhythm classification and confidence score in real time

> All inference is performed locally. No data is transmitted to a remote server.

---

## Development Environment

| Component | Detail                                                            |
|-----------|-------------------------------------------------------------------|
| Language | Python 3.13.x (miniConda3)                                        |
| IDE | PyCharm / Jupyter Notebook                                        |
| Primary OS | macOS 15.7.3                                                      |
| Hardware | Apple MacBook Air M4                                              |
| Key Libraries | TensorFlow/Keras, TensorFlow Lite, NumPy, Pandas, WFDB, Streamlit, Keras-Tuner, Scikit-Learn, Matplotlib |

---
