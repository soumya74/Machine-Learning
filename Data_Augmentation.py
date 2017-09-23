#Data Augmentation
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

datagen = ImageDataGenerator( rotation_range = 20,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              horizontal_flip = True,
                              fill_mode = 'nearest')

folder_path = "D:\\17flowers\\Data_Augmentation"

filecount = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        filepath = os.path.abspath(folder_path) + "\\" +filename
        print (filepath)
        img = load_img(filepath)
        x = img_to_array(img)
        x = x.reshape( (1,) + x.shape)
        
        i = 0
        for batch in datagen.flow( x, save_to_dir = folder_path,
                                   save_prefix = filename, 
                                   batch_size = 1,
                                   shuffle = True,
                                   save_format = "jpeg"):
            i = i + 1
            if (i>5):
                break
        

'''
img = load_img("D:\\17flowers\\Data_Augmentation")
x = img_to_array(img) #numpy array of size (3, img_height, img_width)
x = x.reshape( (1,) + x.shape)

datagen = ImageDataGenerator(
            rotation_range = 0,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            fill_mode = 'nearest')

i = 0
for batch in datagen.flow(x, save_to_dir = 'D:\\Soumya\\Python Scripts\\mnist_convnet_model',
                          save_prefix = 'car', batch_size = 1, shuffle = True, save_format = 'jpeg'):
    i = i + 1
    if( i > 20):
        break
'''