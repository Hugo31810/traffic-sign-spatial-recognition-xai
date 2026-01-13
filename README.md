# 🚦 Traffic Sign Spatial Recognition & XAI

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
> **Comparative analysis of spatial invariance in Neural Networks and Explainable AI (XAI) for Autonomous Driving perception systems.**

This project explores the limitations of **Dense Networks (MLP)** versus **Convolutional Networks (CNN)** in the context of **Spatial Invariance**. It implements an end-to-end pipeline for Traffic Sign Recognition (GTSRB) including custom dataset manipulation to test robustness, and utilizes **Grad-CAM** to audit model decision-making processes, ensuring the system identifies semantic features (pictograms) rather than contextual noise.

---

## 📸 Visualization & Interpretability

**"Peeking inside the Black Box"**: Understanding where the model looks to make a decision.

<br>Model focusing on the truck pictogram. | **Feature Extraction**<br>

<br>Layer 1 filters detecting edges/colors. | **Robustness Failure**<br>

<br>MLP failing on shifted data (Low Acc). |

---

## 💡 Project Overview

In **Autonomous Driving**, a classifier must recognize a stop sign whether it's in the center of the frame or shifted to the side. This project proves empirically why **Inductive Bias** in CNNs is non-negotiable for computer vision.

**Key Capabilities:**

* **🛡️ Robustness Auditing:** Creation of a custom "Shifted Dataset" (`ds2`) to rigorously test positional invariance.
* **📊 Architecture Benchmarking:** Direct comparison between MLP, FCNN, and CNN architectures under stress conditions.
* **🧠 Explainable AI (XAI):** Implementation of Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize saliency maps.
* **🔄 Data Augmentation:** Pipelines to improve generalization using affine transformations.

---

## 🏗️ System Architecture

The pipeline moves from raw data processing to interpretability, highlighting the "Stress Test" branch.

```mermaid
graph LR
    A[GTSRB Dataset] --> B{Data Pipeline}
    B -->|Original| C[Standard Training]
    B -->|Spatial Shift| D[Stress Test (DS2)]
    
    subgraph Model Architectures
    C --> E[MLP Baseline]
    C --> F[CNN / FCNN]
    D --> E
    D --> F
    end
    
    F --> G[Inference]
    G --> H[Performance Analysis]
    G --> I[Grad-CAM XAI]
    
    H --> J((Final Report))
    I --> J

```

---

## 🧪 Experiments & Results

The core experiment involved training models on centered data and testing them on spatially shifted data to measure **Generalization capability**.

### 1. The "Shift" Hypothesis

We created a synthetic dataset (`ds2`) where traffic signs were randomly translated within the 32x32 canvas to simulate camera misalignment or vehicle movement.

### 2. Quantitative Results

| Model Architecture | Inductive Bias | Test Accuracy (Shifted Data) | Conclusion |
| --- | --- | --- | --- |
| **MLP (Dense)** | None | **~24.2%** ❌ | Failed. Treats shifted pixels as new features. |
| **FCNN / CNN** | Spatial | **~83.5%** ✅ | Success. Kernels detect features anywhere. |

---

## 🛠️ Tech Stack

### Core Frameworks

* **Deep Learning:** PyTorch (`torch`, `torchvision`).
* **Data Processing:** NumPy, Pandas.
* **Visualization:** Matplotlib, Seaborn, IPyWidgets (for interactive sliders).
* **Interpretability:** `pytorch-grad-cam`.

---

## 🚀 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/Hugo31810/traffic-sign-spatial-recognition.git
cd traffic-sign-spatial-recognition

```

### 2. Install Dependencies

```bash
pip install torch torchvision matplotlib pandas scikit-learn pytorch-grad-cam opencv-python-headless

```

### 3. Run the Analysis

The project is structured in sequential notebooks for reproducibility:

1. **`01_Dataset_Prep.ipynb`**: Data downloading and "Shifted Dataset" creation.
2. **`02_MLP_Baseline.ipynb`**: Training the Dense Network and observing the failure case.
3. **`03_CNN_Architecture.ipynb`**: Implementing and training the robust CNN/FCNN.
4. **`04_XAI_Visualization.ipynb`**: Running Grad-CAM and filter visualization.

---


## 👨‍💻 Author

**Hugo Salvador Aizpún**

*Degree in Artificial Intelligence*
*Focus: Machine Learning II*

[GitHub Profile](https://www.google.com/search?q=https://github.com/Hugo31810)
