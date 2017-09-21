from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import numpy as np

def _save_bottleneck_features(train_data_dir, validation_data_dir, 
                              bottleneck_feature_train_file, bottlneck_features_validation_file, 
                              batch_size, train_batch_size, validation_batch_size):
    base_model = InceptionV3( include_top = False, weights = 'imagenet')
    datagen = ImageDataGenerator( rescale = 1./255)
    generator_train = datagen.flow_from_directory( train_data_dir,
                                            target_size = (img_width, img_height),
                                            batch_size = batch_size,
                                            class_mode = None,
                                            shuffle = False)
    
    bottleneck_features_train = base_model.predict_generator(generator_train, train_batch_size, verbose = 1)
    np.save( bottleneck_feature_train_file , bottleneck_features_train)
    
    generator_validation = datagen.flow_from_directory( validation_data_dir,
                                                       target_size = (img_width, img_height),
                                                       batch_size = batch_size,
                                                       class_mode = None,
                                                       shuffle = False)
    bottleneck_features_validation = base_model.predict_generator( generator_validation, validation_batch_size, verbose = 1)
    np.save( bottlneck_features_validation_file, bottleneck_features_validation)  






#parameters
img_width = 224
img_height = 224
train_data_dir = "D:\\Soumya\\Python Scripts\\Database\\jpg\\train"
validation_data_dir = "D:\\Soumya\\Python Scripts\\Database\\jpg\\validation"
bottleneck_feature_train_file = "D:\\Soumya\\Python Scripts\\bottleneck_features_train.npy"
bottlneck_features_validation_file = "D:\\Soumya\\Python Scripts\\bottleneck_features_validation.npy"
batch_size = 10
train_batch_size = 40//10
validation_batch_size = 40//10

#model parameter extraction
_save_bottleneck_features(train_data_dir, validation_data_dir, 
                              bottleneck_feature_train_file, bottlneck_features_validation_file, 
                              batch_size, train_batch_size, validation_batch_size)
