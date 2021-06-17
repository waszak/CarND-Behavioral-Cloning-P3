from sklearn.utils import shuffle
import tensorflow as tf
import random
from PIL import Image
import cv2
import numpy as np
from utils import generate_shadow, random_shift


#Generator opens the image
#Apply gausian blur
#If we augment data then it randomize transformation
#50% for flip
#50% for applying shadows
#
class DrivingDataset(tf.data.Dataset):
    def _generator(x, y, augment, repeat=4, batch_size=64):
        x, y = shuffle(x, y)
        batch_x = []
        batch_y = []
        count = 0
        while repeat > 0:
            if augment == False:
                repeat = 0
            else:
                repeat -= 1
            x, y = shuffle(x, y)
            for i in range(len(x)):
                img = np.asarray(Image.open(x[i].decode()))
                yi = y[i]
                img = cv2.GaussianBlur(img, (5, 5), 0)
                if augment:
                    img, yi = random_shift(img, yi)
                    if random.random() <0.5:
                        img = generate_shadow(img)
                    if random.random() <0.5:
                        yi = -yi
                        img = np.fliplr(img)
                
                batch_x.append(img)
                batch_y.append(yi)
                count += 1
                if batch_size <= count:
                    yield ( np.array(batch_x) ,  np.array(batch_y) )
                    count = 0
                    batch_x = []
                    batch_y = []
        if count > 0:
            yield ( np.array(batch_x) ,  np.array(batch_y) )
            
            
    def __new__(cls, x, y, batch_size=64, repeat=4, augment=False):
        ds = tf.data.Dataset.from_generator(
            cls._generator,
            output_types = (tf.uint8, tf.float32 ),
            output_shapes = ((None, 160, 320, 3) , (None,)),
            args=(x, y, augment, repeat, batch_size)
        )
        if augment:
            ds = ds.map(lambda x, y : data_augmentation(x, y), 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
       
        return ds.prefetch(tf.data.experimental.AUTOTUNE)


def create_dataset(images, measurements, batch_size):
    images, measurements = shuffle(images, measurements)
    train_images = images[:int(0.80 * len(images))]
    test_images = images[int(0.80 * len(images)): int(0.90 * len(images))]
    valid_images = images[int(0.90 * len(images)):]
    
    train_measurements = measurements[:int(0.80 * len(measurements))]
    test_measurements = measurements[int(0.80 * len(measurements)): int(0.90 * len(measurements))]
    valid_measurements = measurements[int(0.90 * len(measurements)):]
    
    train_dataset = DrivingDataset(train_images, train_measurements, batch_size, True)
    valid_dataset = DrivingDataset(valid_images, valid_measurements, batch_size)
    test_dataset =  DrivingDataset(test_images, test_measurements, batch_size)
    return train_dataset, valid_dataset, test_dataset

def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    return image, label

def data_augmentation(x, y):
    #seed = (1,2)
    x = tf.image.random_brightness(x, 0.10)
    x = tf.image.random_hue(x, 0.1) 
    x = tf.image.random_saturation(x, 0.7, 1.3)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    return x, y 

#not used
def dataset(x, y, batch_size, augment=False):
    ds =(
            tf.data.Dataset.from_tensor_slices((x, y))
            #.shuffle(len(images))
            #.batch(batch_size)
            .map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            #.cache()
            .shuffle(buffer_size=1000)
            .batch(batch_size)
        )
    if augment:
        ds = ds.map(lambda x, y: data_augmentation(x, y), 
                num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(2)
        
            
    return ds.prefetch(tf.data.experimental.AUTOTUNE)