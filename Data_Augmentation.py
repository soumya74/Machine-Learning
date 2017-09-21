#Data Augmentation
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

img = load_img("D:\\Soumya\\Python Scripts\\mnist_convnet_model\\car.jpg")
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
