import os
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import h5py
import json
import sys

base_model = InceptionV3( include_top = False, weights = 'imagenet')
image_size = (299, 299)
#print (base_model.summary())
#model = Model( input = base_model.input, output = base_model.get_layer('flatten').output)

data_dir = "D:\\17flowers\\jpg"
train_labels = os.listdir(data_dir)

features = []
labels = []


for label in train_labels:
    cur_path = data_dir + "\\" + str(label)
    for image_path in glob.glob(cur_path + "\\*.jpg"):
        #print (image_path)
        img = image.load_img(image_path, target_size = image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)
        feature = base_model.predict(x)
        flat = feature.flatten()
        print (np.shape(flat))
        features.append(flat)
        labels.append(label)
        print ('[INFO] Processed ' + image_path)
       
target_names = np.unique(labels)
le = LabelEncoder()
le_labels = le.fit_transform(labels)

features_path = "D:\\gitHub\\Machine-Learning\\Flower 17 Training\\features_file.h5"
labels_path = "D:\\gitHub\\Machine-Learning\\Flower 17 Training\\labels_file.h5"

file1 = h5py.File(features_path, 'w')
file2 = h5py.File(labels_path, 'w')

try:
    file1.create_dataset('dataset1', data = np.array(features, dtype=np.float64))
    file2.create_dataset('dataset1', data = np.array(le_labels, dtype=np.float64) )
        
    file1.close()
    file2.close()
except:
    print ("[ERROR]", sys.exc_info()[0])
    file1.close()
    file2.close()                