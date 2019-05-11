# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:16:37 2019

@author: dell
"""

import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_training_data(r):
    #plot some data
    plt.plot(r.history['loss'], label = 'loss')
    plt.plot(r.history['val_loss'], label = 'val_loss')
    plt.legend()
    plt.show()
    
    plt.plot(r.history['acc'], label = 'acc')
    plt.plot(r.history['val_acc'], label = 'val_acc')
    plt.legend()
    plt.show()    

def get_model():
    classifier = Sequential()
    classifier.add(Convolution2D(32, 3, 3, input_shape = ( 28, 28, 1), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 128, activation = 'relu'))
    classifier.add(Dense(output_dim = 10, activation = 'softmax'))

    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])    
    return classifier

#prepare dataset
enc = OneHotEncoder(handle_unknown='ignore')

#train dataset
train_dataset = pd.read_csv("../ML Dataset/fashionmnist/fashion-mnist_train.csv")
y_train = train_dataset.iloc[:, 0]
y_train = np.array(y_train)
y_train = y_train.reshape(-1,1)
y_train = enc.fit_transform(y_train)
y_train.astype(int)
x_train = train_dataset.iloc[: , 1:]
x_train = np.array(x_train)
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0

#test dataset
test_dataset = pd.read_csv("../ML Dataset/fashionmnist/fashion-mnist_test.csv")
y_test = test_dataset.iloc[:, 0]
y_test = np.array(y_test)
y_test = y_test.reshape(-1,1)
y_test = enc.fit_transform(y_test)
y_test.astype(int)
x_test = test_dataset.iloc[: , 1:]
x_test = np.array(x_test)
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

#get CNN Model
model = get_model()
print (model.summary())

r = model.fit(x_train, y_train, batch_size = 128, epochs = 10, validation_split = 0.33)
print(r.history.keys())
show_training_data(r)

model.evaluate(x_test, y_test)


