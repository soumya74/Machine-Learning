from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt

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
    print (model.summary())
    return model

def train_store_model(model, X_train, y_train, X_test, y_test):
    batch_size = 200
    model.fit( X_train, y_train, epochs = 1, batch_size = batch_size, verbose = 1, validation_data = (X_test, y_test))
    score = model.evaluate( X_test, y_test, verbose = 2)
    print ("test accuracy = ", score[1])
    model.save('D:\\gitHub\\Machine-Learning\\model.h5')
    model.save_weights('D:\\gitHub\\Machine-Learning\\model_weights.h5')
    
    
def load_trainedModel(model_path, model_weightsPath):
    model1 = load_model(model_path)
    model1.load_weights(model_weightsPath)
    print (model1.summary())

(X_train, y_train),(X_test, y_test) = mnist.load_data()
show_mnistSampleData(X_train)
X_train = X_train.reshape( X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape( X_test.shape[0], 28, 28, 1).astype('float32')
X_train = X_train/255
X_test = X_test/255
y_train = np_utils.to_categorical( y_train)
y_test = np_utils.to_categorical( y_test)
    
num_class = y_train.shape[1]
input_shape = (28, 28, 1)

model_path = 'D:\\gitHub\\Machine-Learning\\model.h5'
model_weightsPath = 'D:\\gitHub\\Machine-Learning\\model_weights.h5'
load_trainedModel(model_path, model_weightsPath)

#model = get_model(num_class, input_shape)
#train_store_model(model, X_train, y_train, X_test, y_test)
