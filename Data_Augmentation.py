#Data Augmentation
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

def _data_augmentation( folder_path, image_count):
    datagen = ImageDataGenerator( rotation_range = 20,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,
                                  fill_mode = 'nearest')
    
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
                if (i>image_count):
                    break

folder_path = "D:\\17flowers\\Data_Augmentation"
_data_augmentation( folder_path, 5)