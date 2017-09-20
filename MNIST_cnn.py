from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
import numpy
import os
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
    model.add(Conv2D(32, kernel_size = (5, 5), padding='valid', input_shape = input_shape))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation = 'softmax'))
    model.add(Dense(num_class, activation = 'softmax'))
    #compile model
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def store_model(model):
    model_json = model.to_json()
    with open( 'model_json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weigths('D:\\Soumya\\Python Scripts\\model.h5')



(X_train, y_train),(X_test, y_test) = mnist.load_data()
#show dataset
#show_mnistSampleData(X_train)

batch_size = 200

input_shape = (28, 28, 1)
X_train = X_train.reshape( X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape( X_test.shape[0], 28, 28, 1).astype('float32')
X_train = X_train/255
X_test = X_test/255
y_train = np_utils.to_categorical( y_train)
y_test = np_utils.to_categorical( y_test)
num_class = y_train.shape[1]


model = get_model(num_class, input_shape)
print (model.summary())
model.fit( X_train, y_train, epochs = 1, batch_size = batch_size, verbose = 1, validation_data = (X_test, y_test))
score = model.evaluate( X_test, y_test, verbose = 2)
print ("test accuracy = ", score[1])
#store_model(model)
model.save_weights('D:\\Soumya\\Python Scripts\\mnist_convnet_model\\model.h5')
