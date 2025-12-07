# Handwritten Digit Recognition (MNIST Dataset)
Deep Learning Classification using a Fully Connected Neural Network

This project implements an end-to-end deep learning pipeline for recognizing handwritten digits (0–9) using the MNIST dataset.  
It includes data preprocessing, model training, evaluation metrics, visualization, and an inference workflow using a saved TensorFlow model.

The repository contains:

- Training script: `create_model.py`  
- Testing and prediction script: `test_network.py`  
- Optional drawing interface: `number_drawing.py`  
- Saved TensorFlow model: `digit_recognition_model/`  
- Visualizations (accuracy curve, loss curve, sample predictions)

---

## Project Overview

Handwritten digit recognition is a foundational computer vision task and a standard benchmark dataset in deep learning research.

This project trains a neural network to classify each 28x28 grayscale input image into one of the ten digit classes:

0, 1, 2, 3, 4, 5, 6, 7, 8, 9

Key project goals:

- Preprocessing MNIST images  
- Building and training a neural network  
- Evaluating accuracy and loss  
- Saving and loading trained models  
- Performing inference on new digit images  

---

## Dataset Description

The project uses the MNIST dataset consisting of:

- 60,000 training images  
- 10,000 test images  
- 28x28 grayscale images  
- One digit per image (0–9)

TensorFlow downloads the dataset automatically:

```python
from tensorflow.keras.datasets import mnist
```

Preprocessing steps include:

- Flattening images to 784 features  
- Scaling pixel values to [0, 1]  
- Converting labels to one-hot encoded vectors  

---

## Model Architecture

The neural network used in this project is a fully connected feed-forward model with the following structure:

```
Input Layer: 784 values (flattened 28×28 image)
Dense Layer: 128 neurons, ReLU activation
Dense Layer: 128 neurons, ReLU activation
Output Layer: 10 neurons, Softmax activation
```

Model characteristics:

- Loss function: categorical cross-entropy  
- Optimizer: Adam  
- Metrics: accuracy  

---

## Training the Model

Training is handled inside `create_model.py`.

Main steps:

1. Load and preprocess the MNIST dataset  
2. Build the neural network  
3. Train the model with validation monitoring  
4. Save the trained model to `digit_recognition_model/`  
5. Generate training accuracy and loss plots  

Hyperparameters such as epochs, batch size, and learning rate can be modified inside the script.

---

## Saving and Loading the Model

After training, the TensorFlow model is saved automatically.

To load it:

```python
import tensorflow as tf
model = tf.keras.models.load_model("digit_recognition_model")
```

---

## Evaluation

The project evaluates the model using:

- Training and validation accuracy  
- Training and validation loss  
- Test accuracy on unseen images  
- Confusion matrix (optional)  
- Sample prediction outputs  

Typical MNIST performance using this model:

Accuracy: 96% – 98%

Actual results may vary depending on the number of epochs and hyperparameters.

---

## Inference Workflow

`test_network.py` loads the saved model and performs predictions on sample MNIST images.

Example:

```python
prediction = model.predict(image.reshape(1, 784))
print("Predicted Digit:", prediction.argmax())
```

The optional `number_drawing.py` script allows drawing a digit on-screen and passing it to the model for prediction.

---

## File Structure

```
Handwritten_Digit_Recognition/
│
├── create_model.py
├── test_network.py
├── number_drawing.py
├── digit_recognition_model/
│   ├── saved_model.pb
│   └── variables/
└── README.md
```

---

## Team Members

- Madhav Bhatia (102215172)
- Aditi (102215174)
- Shriya (102215224)
- Aryan Nagpal (102215358)

---

## References

MNIST Dataset by Yann LeCun  
http://yann.lecun.com/exdb/mnist/
