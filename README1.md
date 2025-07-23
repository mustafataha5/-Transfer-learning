# 🧠 Transfer Learning Projects with Keras

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg?style=flat&logo=keras&logoColor=white)](https://keras.io/)

Welcome to my repository on **Transfer Learning using Keras and TensorFlow**. This repository contains multiple projects demonstrating how to perform image classification using pre-trained models like **Xception** and **MobileNetV2**, along with techniques such as fine-tuning and model loading for inference.
<!-- 
## 📁 Repository Structure

.
├── 001 introduction/           # Introduction to Transfer Learning concepts (Not expanded yet)
├── 002 Binary Classifier/      # Binary image classification using Xception
├── 003 Multi-Classification/   # Multi-class image classification using MobileNetV2
├── anaconda_projects/          # (Potentially other project artifacts or temporary files)
├── .ipynb_checkpoints/         # Jupyter auto-saved checkpoints
├── .gitignore                  # Git ignore file for version control
├── README.md                   # Project overview and instructions
└── requirements.txt            # Python library dependencies

---

## 🔍 Project Highlights

### ✅ 002 Binary Classifier

**Goal**: Classify images as either *urban* or *rural*.

**Techniques**:
- Custom CNN modeling
- Transfer Learning with **Xception**
- Saving and loading `.keras` models for inference

**Files**:
- `001 DataLoad.ipynb` – Prepares and loads image data.
- `002 CNN with keras.ipynb` – Builds a custom CNN model.
- `003 Using Transfer Learning.ipynb` – Demonstrates using a pre-trained Xception model.
- `004 load the model to use.ipynb` – Loads a trained model and performs predictions.
- `my_xception_model.keras` – Saved trained model.

📂 Dataset Folder: `rural_and_urban_photos/` (Ensure this folder is present in `002 Binary Classifier/` or as specified by the notebook.)

---

### ✅ 003 Multi-Classification

**Goal**: Classify images from CIFAR-10 into 10 categories.

**Techniques**:
- Data preparation from CIFAR-10 dataset
- Transfer learning using **MobileNetV2**
- Fine-tuning the model for improved performance
- Saving and loading the model for inference

**Files**:
- `001 Dataload.ipynb` – Loads and preprocesses the CIFAR-10 dataset.
- `002 Classify_using_Mobile.ipynb` – Classifies images using MobileNetV2.
- `003 Classifier with Transfer Learning with FineTuning.ipynb` – Fine-tunes the MobileNetV2 model.
- `004 use of the loaded model.ipynb` – Loads and tests the saved model.
- `cifar10_model.keras` – Saved trained model.

---

## 🛠️ Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/Transfer-learning.git](https://github.com/yourusername/Transfer-learning.git)
    cd Transfer-learning
    ```
    *(Replace `yourusername` with your actual GitHub username if this is your own fork/repository.)*

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **On Windows (Command Prompt):**
        ```cmd
        venv\Scripts\activate.bat
        ```
    * **On Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter Notebook/Lab:**
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```

## 📦 Requirements

All required libraries are listed in `requirements.txt`. Key dependencies include:

* **TensorFlow**
* **Keras**
* **NumPy**
* **Matplotlib**
* **scikit-learn**

## 📌 Notes

* All trained models are saved in the new `.keras` format for easy reusability.
* Ensure that any required dataset folders (e.g., `rural_and_urban_photos/`) are correctly placed relative to their respective notebooks before execution. -->

# Deep Learning Image Classification Projects

This repository contains two main parts demonstrating the use of deep learning in image classification using TensorFlow and Keras:

- [✔️] **Binary Classification** (Urban vs. Rural)
- [✔️] **Multi-Classification** (CIFAR-10)

---

## 🧠 001 - Binary Classifier (Urban vs. Rural)

> A binary image classifier using CNN and transfer learning techniques with Keras. The model distinguishes between urban and rural scenes.

### 📁 Files

| File | Description |
|------|-------------|
| `001 DataLoad.ipynb` | Loading and preprocessing the urban and rural dataset |
| `002 CNN with keras .ipynb` | CNN model built from scratch |
| `003 using lear transfer.ipynb` | Transfer learning applied using a pretrained model |
| `004 load the model to use.ipynb` | Loading and using the trained model for inference |
| `accuracy_and_loss.png` | Graph showing training accuracy and loss |
| `my_xception_model.keras` | Saved trained model (Xception) |
| `urban-1.jpg` | Sample image used for prediction |
| `rural_and_urban_photos/` | Directory containing dataset images |

---

## 🧠 002 - Multi-Classification (CIFAR-10)

> A classifier trained to identify 10 different object categories from the CIFAR-10 dataset using MobileNet and fine-tuned transfer learning.

### 📁 Files

| File | Description |
|------|-------------|
| `001 Dataload.ipynb` | Load and preprocess CIFAR-10 dataset |
| `002 Classifiy_using_Mobile.ipynb` | Train classifier using MobileNet |
| `003 Classifier wuth transfer learning with FineTuning.ipynb` | Advanced training with fine-tuning |
| `004 use of the loaded model.ipynb` | Load and evaluate trained model |
| `accuracy.png` | Accuracy graph |
| `loss.png` | Loss graph |
| `cifar10_model.keras` | Saved trained model |

---

## 🚀 Technologies Used

- Python
- TensorFlow / Keras
- Matplotlib
- CNNs
- Transfer Learning (MobileNet, Xception)
- Jupyter Notebooks

---

## 📌 Notes

- All `.keras` files are trained model files that can be loaded for prediction.
- The accuracy and loss graphs provide insights into the training progress.
- Datasets used:
  - CIFAR-10 for multi-class
  - Custom rural/urban dataset for binary classification

---

## 📂 Folder Structure

```bash
├── 001 Binary Classifier/
│   ├── *.ipynb
│   ├── *.keras
│   ├── *.png
│   └── rural_and_urban_photos/
│
└── 002 Multi-Classification/
    ├── *.ipynb
    ├── *.keras
    └── *.png

## ✍️ Author

**Mustafa Taha**
* 📍 Ramallah, Palestine
* 📧 mustafa.taha.mu95@gmail.com
* 🔗 [GitHub](https://github.com/mustafataha5) | [LinkedIn](https://linkedin.com/in/mustafa-taha-3b87771b4/)
    *(Remember to replace `yourusername` and `yourlinkedinprofile` with your actual GitHub and LinkedIn profile links.)*

---

⭐️ If you found this useful, please consider giving it a star!