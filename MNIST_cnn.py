# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:32:06 2017

@author: soumyas
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
'''
(X_train, y_train),(X_test, y_test) = mnist.load_data()
print (X_train.shape)
print (y_train)
y_train = np_utils.to_categorical(y_train)
print (y_train.shape)
print (y_train)
'''

def show_mnistSampleData(X_train):
    plt.subplot(221)
    plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(X_train[3], cmap=plt.get_cmap('gray')) 


def get_model(num_class, input_shape):
    #create model
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (5, 5), data_format = 'channels_last', input_shape = ( 28, 28, 1)))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation = 'softmax'))
    model.add(Dense(num_class, activation = 'softmax'))
    #compile model
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model





(X_train, y_train),(X_test, y_test) = mnist.load_data()
#show dataset
#show_mnistSampleData(X_train)

batch_size = 200
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

X_train = X_train.reshape( X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape( X_test.shape[0], 1, 28, 28).astype('float32')
X_train = X_train/255
X_test = X_test/255
y_train = np_utils.to_categorical( y_train)
y_test = np_utils.to_categorical( y_test)
num_class = y_train.shape[1]


model = get_model(num_class, input_shape)
print (model.summary())
model.fit( X_train, y_train, epochs = 10, batch_size = batch_size, verbose = 2, validation_data = (X_test, y_test))
score = model.evaluate( X_test, y_test, verbose = 2)
print ("test accuracy = ", score[1])
