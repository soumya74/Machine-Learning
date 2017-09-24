import h5py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

print ("[LOADING FEATURES]")
features_path = "D:\\gitHub\\Machine-Learning\\features_file.h5"
labels_path = "D:\\gitHub\\Machine-Learning\\labels_file.h5"

file1 = h5py.File(features_path, 'r')
file2 = h5py.File(labels_path, 'r')

features_string = file1['dataset1']
labels_string = file2['dataset1']

features = np.array(features_string)
labels = np.array(labels_string)

file1.close()
file2.close()

print ("[TRAINING MODEL]")
seed = 7
(train_data, test_data, train_label, test_label) = train_test_split( features,
                                                                     labels,
                                                                     test_size = 0.7,
                                                                     random_state = seed)

model = LogisticRegression(random_state = seed)
model.fit( train_data, train_label)

rank_1 = 0
rank_5 = 0

print ("[CHECKING ACCURACY]")
for (label, features) in zip(test_label, test_data):
    predictions = model.predict_proba( np.atleast_2d(features))[0]
    predictions = np.argsort(predictions)[::-1][:5]
    
    if label==predictions[0]:
        rank_1 = rank_1 + 1
        
    if label in predictions:
        rank_5 = rank_5 + 1
        
rank_1 = (rank_1 / float(len(test_label)))*100
rank_5 = (rank_5 / float(len(test_label)))*100
         
print ("[RANK_1]" + str(rank_1))
print ("[RANK_5]" + str(rank_5))    