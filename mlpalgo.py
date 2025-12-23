import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

data= pd.read_csv(r"D:\Internship\USA Housing Dataset.csv")
X = data.iloc[:, 1:].values

Y = data.iloc[:,0].values
Y = Y.reshape(Y.size, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt


model = Sequential([ 
    Flatten(input_shape= (12,)), 
    Dense(256, activation='relu'),   
    Dense(128, activation='relu'),  
    Dense(10, activation='softmax'),   
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10,  batch_size=2000,  validation_split=0.2)
