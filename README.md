# USA Housing Price Prediction using Deep Neural Network (TensorFlow & Keras)

## Project Overview
This project demonstrates the application of a **Deep Neural Network (DNN)** built using **TensorFlow and Keras** to learn patterns from the **USA Housing Dataset**. The objective is to train a neural network model capable of predicting target outcomes based on multiple housing-related features.

The project emphasizes understanding **deep learning workflows**, including data preparation, model architecture design, compilation, and training.

---

## Objectives
- Load and preprocess a real-world housing dataset
- Build a deep neural network using TensorFlow/Keras
- Train the model using modern optimization techniques
- Evaluate learning behavior using training and validation metrics
- Understand multi-layer neural network architecture

---

## Technologies Used
- Python 3
- Pandas
- TensorFlow
- Keras
- scikit-learn
- Matplotlib

---

## Dataset Description
The dataset (`USA Housing Dataset.csv`) contains numerical housing-related features.

### Input Features
- 12 numerical attributes representing housing characteristics  
  (e.g., area, income, population, rooms, etc.)

### Target Variable
- The first column of the dataset, representing the output variable  
- Reshaped to a column vector for neural network compatibility

---

## Data Preprocessing
- Loaded dataset using Pandas
- Selected feature matrix and target variable
- Converted data into NumPy arrays
- Split data into training and testing sets:
  - 70% Training
  - 30% Testing
- Applied reshaping to match neural network input requirements

---

## Model Architecture
The neural network is built using the **Sequential API** with the following layers:

- **Flatten Layer**
  - Converts input data into a 1D vector
- **Dense Layer (256 neurons, ReLU activation)**
  - Learns high-level feature representations
- **Dense Layer (128 neurons, ReLU activation)**
  - Adds depth and abstraction
- **Output Layer (10 neurons, Softmax activation)**
  - Produces class probability distribution

This architecture demonstrates a **multi-layer fully connected neural network**.

---

## Model Compilation
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Metrics:** Accuracy  

The chosen optimizer and loss function ensure efficient learning for multi-class classification tasks.

---

## Model Training
- Trained for **10 epochs**
- Batch size set to **2000**
- Validation split of **20%** applied to training data
- Training progress monitored using accuracy and loss metrics

---

## Results
- The model learns patterns from housing data effectively
- Training and validation accuracy provide insight into model performance
- Demonstrates the power of deep learning on structured datasets

---

## How to Run the Project

### Prerequisites
Install the required libraries:
- pandas
- tensorflow
- scikit-learn
- matplotlib

---

### Steps
1. Place `USA Housing Dataset.csv` in the specified directory or update the file path
2. Run the Python script
3. Observe:
   - Training and validation accuracy
   - Loss values across epochs

---

## Learning Outcomes
- Understanding deep neural network architecture
- Hands-on experience with TensorFlow and Keras
- Practical exposure to data preprocessing for deep learning
- Ability to train and validate multi-layer neural networks

---

## Future Improvements
- Normalize or standardize input features
- Add dropout layers to reduce overfitting
- Tune hyperparameters (epochs, batch size, neurons)
- Convert task to regression if predicting continuous housing prices
- Visualize loss and accuracy curves

---

## Use Case
This project is suitable for:
- Deep Learning and AI portfolios
- Academic coursework and labs
- Demonstrating TensorFlow/Keras proficiency
- Understanding neural networks on structured data

---

## Author
Soban Saeed
Developed as an educational deep learning project using TensorFlow and Keras for housing data analysis.
