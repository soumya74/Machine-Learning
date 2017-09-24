import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

train_path = "D:\\17flowers\\train"
train_labels = os.listdir(train_path)
print (train_labels)
le = LabelEncoder()
le.fit([t1 for t1 in train_labels])

print (le.classes_)

features = []
print (type(features))

for i in range(0,2):
    a = np.array([0, 1, 2])
    features.append(a)
    
print (np.array(features))

