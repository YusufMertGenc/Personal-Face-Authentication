# Personal Face Authentication with Transfer Learning

## Overview

This project implements a personal face authentication system using a fine-tuned **pretrained ResNet18 CNN**. The model performs binary classification on face images and predicts whether the given image belongs to **ME** or **NOT ME**.

The project demonstrates how transfer learning can be used effectively on a small personal dataset. A pretrained convolutional neural network is adapted for a custom face authentication task, evaluated with classification metrics, and deployed as an interactive Gradio web application.

---

## Project Features

* Fine-tuning a pretrained ResNet18 model for binary face classification
* Custom dataset structure for positive and negative face samples
* Training, validation, and testing pipeline
* Evaluation using precision, recall, and F1 score
* Interactive face authentication demo with Gradio
* Saved best model checkpoint during training

---

## Project Structure

```text
assignment2/
├── src/
│   ├── dataset.py                # Dataset loading and preprocessing
│   ├── model.py                  # ResNet18 model construction
│   ├── train.py                  # Training loop implementation
│   ├── evaluate.py               # Evaluation metrics
│   └── app.py                    # Gradio prediction application
│
├── data/
│   ├── positive/
│   │   ├── train/                # Personal training images
│   │   ├── val/                  # Personal validation images
│   │   └── test/                 # Personal test images
│   └── negative/
│       ├── train/                # Negative training images
│       ├── val/                  # Negative validation images
│       └── test/                 # Negative test images
│
├── checkpoints/
│   └── best_model.pt             # Best trained model checkpoint
│
├── logs/
│   └── history.npy               # Training history
│
├── config.json                   # Hyperparameters and file paths
├── notebook.ipynb                # Main notebook for running the project
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Dataset

The dataset is organized as a binary classification problem:

| Split      | ME | NOT ME | Total |
| ---------- | -- | ------ | ----- |
| Train      | 20 | 20     | 40    |
| Validation | 5  | 5      | 10    |
| Test       | 5  | 5      | 10    |

The positive class contains personal face images, while the negative class contains fixed face samples from the LFW dataset.

The test set is kept separate from training and validation to evaluate the final model fairly.

---

## Model

The project uses **ResNet18**, a pretrained convolutional neural network. Instead of training a CNN from scratch, transfer learning is applied by reusing the pretrained feature extractor and modifying the final classification layer for a binary classification task.

The model predicts one of two classes:

```text
0 -> NOT ME
1 -> ME
```

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Prepare the dataset

Place the positive images into the following folders:

```text
data/positive/
├── train/
├── val/
└── test/
```

The negative images are handled through the dataset loading pipeline.

---

### 2. Run the notebook

Open and run the notebook step by step:

```text
notebook.ipynb
```

Recommended order:

```text
1. Setup
2. Dataset
3. Model
4. Training
5. Evaluation
6. Gradio App
```

---

### 3. Train the model

During training, the best performing model is saved to:

```text
checkpoints/best_model.pt
```

Training history is saved to:

```text
logs/history.npy
```

---

### 4. Evaluate the model

The trained model is evaluated on the test set using:

* Precision
* Recall
* F1 Score

These metrics help measure how well the model distinguishes between personal and non-personal face images.

---

### 5. Run the Gradio application

The Gradio app allows users to upload a face image and receive a prediction from the trained model.

The application outputs:

* Predicted class
* Confidence score

---

## Results

The final model successfully demonstrates a small-scale personal face authentication system. By using transfer learning, the model can learn from a limited number of personal face images while still benefiting from the strong feature extraction capabilities of a pretrained CNN.

---

## Notes

* The test images are not used during training or validation.
* The trained model checkpoint and dataset files are excluded from version control or submission packages.
* The project is designed as a practical example of transfer learning, binary classification, and model deployment with Gradio.

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* ResNet18
* NumPy
* Scikit-learn
* Gradio

---

## Author

Yusuf Mert Genç
