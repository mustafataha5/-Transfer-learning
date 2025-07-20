# 🧠 Transfer Learning Projects with Keras

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg?style=flat&logo=keras&logoColor=white)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

Welcome to my repository on **Transfer Learning using Keras and TensorFlow**. This repository contains two projects demonstrating image classification using pre-trained models like **Xception** and **MobileNetV2**, with techniques such as custom CNN modeling, transfer learning, fine-tuning, and model loading for inference.

# Deep Learning Image Classification Projects

This repository includes two main projects:

- [✔️] **Binary Classification** (Urban vs. Rural)
- [✔️] **Multi-Classification** (CIFAR-10)

---

## 🧠 001 - Binary Classifier (Urban vs. Rural)

> A binary image classifier that distinguishes between urban and rural scenes using a custom CNN and transfer learning with Xception.

### 📁 Files

| File | Description |
|------|-------------|
| `001 DataLoad.ipynb` | Loads and preprocesses the urban and rural dataset |
| `002 CNN with keras.ipynb` | Builds a custom CNN model from scratch |
| `003 Using Transfer Learning.ipynb` | Applies transfer learning using a pre-trained Xception model |
| `004 Load the Model to Use.ipynb` | Loads the trained model for inference |
| `accuracy_and_loss.png` | Graph showing training accuracy and loss |
| `my_xception_model.keras` | Saved trained Xception model |
| `urban-1.jpg` | Sample image used for prediction |
| `rural_and_urban_photos/` | Directory containing dataset images |

📂 **Dataset**: Ensure the `rural_and_urban_photos/` folder is placed in `001 Binary Classifier/` or as specified in the notebooks. The dataset should contain images labeled as `urban` or `rural`.

---

## 🧠 002 - Multi-Classification (CIFAR-10)

> A classifier trained to identify 10 object categories from the CIFAR-10 dataset using MobileNetV2 with transfer learning and fine-tuning.

### 📁 Files

| File | Description |
|------|-------------|
| `001 Dataload.ipynb` | Loads and preprocesses the CIFAR-10 dataset |
| `002 Classify_using_Mobile.ipynb` | Trains a classifier using MobileNetV2 |
| `003 Classifier with Transfer Learning with FineTuning.ipynb` | Fine-tunes the MobileNetV2 model |
| `004 Use of the Loaded Model.ipynb` | Loads and evaluates the trained model |
| `accuracy.png` | Accuracy graph for training and validation |
| `loss.png` | Loss graph for training and validation |
| `cifar10_model.keras` | Saved trained MobileNetV2 model |

📂 **Dataset**: Uses the CIFAR-10 dataset, accessible via `tensorflow.keras.datasets.cifar10`.

---

## 🚀 Technologies Used

- Python 3.x
- TensorFlow 2.x / Keras 2.x
- Matplotlib
- scikit-learn
- NumPy
- Convolutional Neural Networks (CNNs)
- Transfer Learning (MobileNetV2, Xception)
- Jupyter Notebooks

---

## 🛠️ Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/mustafataha5/Transfer-learning.git
    cd Transfer-learning
    ```

2. **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```
    - **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```
    - **On Windows (Command Prompt)**:
        ```cmd
        venv\Scripts\activate.bat
        ```
    - **On Windows (PowerShell)**:
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Launch Jupyter Notebook/Lab**:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```

---

## 📦 Requirements

Key dependencies are listed in `requirements.txt`:
- `tensorflow>=2.15.0`
- `keras>=2.15.0`
- `numpy`
- `matplotlib`
- `scikit-learn`

To install, run:
```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- All trained models are saved in the `.keras` format for easy loading and inference.
- Ensure the `rural_and_urban_photos/` dataset is correctly placed for the binary classifier project.
- The `accuracy.png`, `loss.png`, and `accuracy_and_loss.png` files visualize training progress.
- The CIFAR-10 dataset is automatically downloaded via TensorFlow/Keras.
- For best results, run notebooks in order (e.g., `001` → `002` → `003` → `004`).
- Tested on Python 3.8+ with TensorFlow 2.15.0.

---

## 📂 Folder Structure

```bash
├── 001 Binary Classifier/
│   ├── 001 DataLoad.ipynb
│   ├── 002 CNN with keras.ipynb
│   ├── 003 Using Transfer Learning.ipynb
│   ├── 004 Load the Model to Use.ipynb
│   ├── accuracy_and_loss.png
│   ├── my_xception_model.keras
│   ├── urban-1.jpg
│   ├── rural_and_urban_photos/
│   └── .ipynb_checkpoints/
├── 002 Multi-Classification/
│   ├── 001 Dataload.ipynb
│   ├── 002 Classify_using_Mobile.ipynb
│   ├── 003 Classifier with Transfer Learning with FineTuning.ipynb
│   ├── 004 Use of the Loaded Model.ipynb
│   ├── accuracy.png
│   ├── loss.png
│   ├── cifar10_model.keras
│   └── .ipynb_checkpoints/
├── .gitignore
├── README.md
├── requirements.txt
└── anaconda_projects/
```

---

## ✍️ Author

**Mustafa Taha**
- 📍 Ramallah, Palestine
- 📧 mustafa.taha.mu95@gmail.com
- 🔗 [GitHub](https://github.com/mustafataha5)
- 🔗 [LinkedIn](https://linkedin.com/in/mustafa-taha-3b87771b4/)

---

⭐️ If you found this useful, please consider giving it a star!