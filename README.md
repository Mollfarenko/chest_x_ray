# Pneumonia Detection from Chest X-Rays (In Progress...)

This project uses deep learning to detect pneumonia from chest X-ray images. It aims to train a convolutional neural network (CNN) capable of classifying whether a patient has pneumonia or not based on a chest scan.

---

## 🧠 Goals

- Build and train a CNN model with PyTorch
- Preprocess and visualize medical imaging data
- Evaluate model performance with accuracy, ROC curve, etc.
- Deploy or use the model to predict new cases

---

## 📁 Project Structure
```bash
chest_xray/
│
├── data/
│   ├── train/      # Dataset used for training the model
│   └── test/       # Dataset used for training/testing the model
│   └── val/        # Dataset used for validating the model

├── models/
│   └── model_architecture.py  # Model definition

├── scripts/
│   ├── train.py         # Model training loop
│   ├── evaluate.py      # Evaluation + metrics
│   └── predict.py       # Inference on new X-rays

├── utils/
│   └── dataloader.py    # Dataset class, transformations, etc.

├── reports/
│   ├── figures/         # Accuracy, confusion matrix, etc.
│   └── logs/            # TensorBoard/MLflow if you want

├── README.md
├── requirements.txt
├── environment.yml
└── .gitignore
```

