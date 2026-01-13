¡Entendido! Vamos a darle ese toque **profesional y visual** utilizando los *shields* (escudos/insignias) de **Shields.io** con el estilo `for-the-badge` (el rectangular grande), que es el estándar actual en GitHub para portafolios de ingeniería.

He seleccionado los logos exactos de las tecnologías que hemos usado en estos notebooks (PyTorch, Pandas, NumPy, etc.).

Copia y pega el siguiente código en tu archivo `README.md`:

---

# 🚦 Traffic Sign Spatial Recognition & XAI

> **Comparative analysis of spatial invariance in Neural Networks and Explainable AI (XAI) for Autonomous Driving perception systems.**

This project explores the limitations of **Dense Networks (MLP)** versus **Convolutional Networks (CNN)** in the context of **Spatial Invariance**. It implements an end-to-end pipeline for Traffic Sign Recognition (GTSRB) including custom dataset manipulation to test robustness, and utilizes **Grad-CAM** to audit model decision-making processes, ensuring the system identifies semantic features (pictograms) rather than contextual noise.

---

## 📸 Visualization & Interpretability

**"Peeking inside the Black Box"**: Understanding where the model looks to make a decision.

|  |  |  |
| --- | --- | --- |
| **Grad-CAM Analysis**<br>

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

## 📂 Project Structure

```text
.
├── metrics/                # CSV logs for Loss/Accuracy curves
│   ├── train_loss_cnn.csv
│   └── test_acc_mlp.csv
├── models/                 # Saved .pth state dictionaries
│   ├── cnn_best.pth
│   └── mlp_ds2.pth
├── notebooks/              # Jupyter Notebooks (Source Code)
├── src/                    # Python modules
│   ├── models.py           # Architecture definitions (CNN, MLP class)
│   └── utils.py            # Helper functions (plotting, training loops)
└── README.md

```

---

## 👨‍💻 Author

**Hugo Salvador Aizpún**

*Degree in Artificial Intelligence*
*Focus: Computer Vision & Deep Learning*

[GitHub Profile](https://www.google.com/search?q=https://github.com/Hugo31810)
