#http://www.codesofinterest.com/2017/08/bottleneck-features-multi-class-classification-keras.html
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical
import numpy as np

def _save_bottleneck_features(train_data_dir, validation_data_dir, 
                              bottleneck_feature_train_file, bottlneck_features_validation_file, 
                              batch_size):
    print ("\n_save_bottleneck_features--------------------->")
    #base_model = InceptionV3( include_top = False, weights = 'imagenet')
    base_model = VGG16( include_top = False, weights = 'imagenet')
    
    #When it is needed to extract data from a specific layer
    #model = Model(input = base_model.input, output = base_model.get_layer('fc1').output)
    
    datagen = ImageDataGenerator( rescale = 1./255)
    generator_train = datagen.flow_from_directory( train_data_dir,
                                            target_size = (img_width, img_height),
                                            batch_size = batch_size,
                                            class_mode = None,
                                            shuffle = False)
    
    train_batch_size = len(generator_train.filenames)//batch_size
    bottleneck_features_train = base_model.predict_generator(generator_train, train_batch_size, verbose = 1)
    np.save( bottleneck_feature_train_file , bottleneck_features_train)
    
    generator_validation = datagen.flow_from_directory( validation_data_dir,
                                                       target_size = (img_width, img_height),
                                                       batch_size = batch_size,
                                                       class_mode = None,
                                                       shuffle = False)    
    
    validation_batch_size = len(generator_validation.filenames)//batch_size    
    bottleneck_features_validation = base_model.predict_generator( generator_validation, validation_batch_size, verbose = 1)
    np.save( bottlneck_features_validation_file, bottleneck_features_validation)  


def _get_topLevel_model(input_shape, num_classes):
    print ("\n_get_topLevel_model--------------------->")
    top_model = Sequential()
    top_model.add( Flatten(input_shape = input_shape))
    top_model.add( Dense(256, activation = 'relu'))
    top_model.add( Dropout(0.5))
    top_model.add( Dense(num_classes, activation = 'sigmoid'))  
    
    top_model.compile( optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return top_model


def _train_topLevel_model( bottleneck_feature_train_file, bottlneck_features_validation_file, 
                          train_data_dir, validation_data_dir, img_width, 
                          img_height, batch_size, trained_top_model_file,
                          trained_top_modelWeights_file, topLevel_model_epochs):
    print ("\n_train_topLevel_model--------------------->")
    
    #this generator required for obtaining class levels etc details
    topLevel_datagen = ImageDataGenerator( rescale = 1./255)
    topLevel_train_generator = topLevel_datagen.flow_from_directory( train_data_dir,
                                                   target_size = (img_width, img_height),
                                                   batch_size = batch_size,
                                                   class_mode = 'categorical',
                                                   shuffle = False)
    topLevel_validation_generator = topLevel_datagen.flow_from_directory( validation_data_dir,
                                                   target_size = (img_width, img_height),
                                                   batch_size = batch_size,
                                                   class_mode = 'categorical',
                                                   shuffle = False)    

    num_classes = len(topLevel_train_generator.class_indices)
    print (num_classes)
    
    train_level = topLevel_train_generator.classes
    train_level = to_categorical( train_level, num_classes = num_classes)
    
    validation_level = topLevel_validation_generator.classes
    validation_level = to_categorical( validation_level, num_classes = num_classes)
    print (np.shape(validation_level))
    
    train_data = np.load( bottleneck_feature_train_file)
    validation_data = np.load( bottlneck_features_validation_file)
    print (np.shape(validation_data))
    
    topLevel_model = _get_topLevel_model(train_data.shape[1:], num_classes)
    topLevel_model.fit( train_data, train_level, batch_size = batch_size, 
                       epochs = topLevel_model_epochs,
                       validation_data = (validation_data, validation_level))
    
    topLevel_model.save( trained_top_model_file)
    topLevel_model.save_weights( trained_top_modelWeights_file)


   

#parameters
img_width = 224
img_height = 224
train_data_dir = "D:\\17flowers\\jpg"
validation_data_dir = "D:\\17flowers\\jpg"
bottleneck_feature_train_file = "D:\\gitHub\\Machine-Learning\\bottleneck_features_train.npy"
bottlneck_features_validation_file = "D:\\gitHub\\Machine-Learning\\bottleneck_features_validation.npy"
trained_top_model_file = "D:\\gitHub\\Machine-Learning\\top_model.h5"
trained_top_modelWeights_file = "D:\\gitHub\\Machine-Learning\\top_model_weights.h5"
batch_size = 10
topLevel_model_epochs = 10


#model parameter extraction
_save_bottleneck_features(train_data_dir, validation_data_dir, 
                              bottleneck_feature_train_file, bottlneck_features_validation_file, 
                              batch_size)

#train and save topLevel Model
_train_topLevel_model( bottleneck_feature_train_file, bottlneck_features_validation_file, 
                          train_data_dir, validation_data_dir, img_width, 
                          img_height, batch_size, trained_top_model_file,
                          trained_top_modelWeights_file, topLevel_model_epochs)
