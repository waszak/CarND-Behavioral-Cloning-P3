from sklearn.utils import shuffle
import tensorflow as tf
import random
from PIL import Image
import cv2
import numpy as np
from utils import generate_shadow, random_shift
import keras
import math
#Generator opens the image
#Apply gausian blur
#If we augment data then it randomize transformation
#50% for flip
#60% for applying shadows
#

class DrivingDatasetGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size=64, augment=False, repeat=1):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.augment= augment
        self.repeat = repeat
        self.idx = []
        self.augment = augment
        if not augment:
            repeat = 1
    
        self.on_epoch_end()
    
    def __del__(self):
        pass
    
    def on_epoch_end(self):
        idx = [ i for i in range(len(self.x))]
        idxes = []
        for i in range(self.repeat):
            idx = shuffle(idx)
            idxes.extend(idx)
        self.idx = idxes
    
    def __len__(self):
        return math.ceil((len(self.idx)) / self.batch_size) - 1
    
    def total_size(self):
        return len(self.idx)
   
    def __getitem__(self, idx):
        idxes = self.idx[idx * self.batch_size:(idx + 1) * self.batch_size]
        idxes = shuffle(idxes)
        batch_x = []
        batch_y = []

        
        for i in idxes:
            img = np.asarray(Image.open(self.x[i]))
            yi = self.y[i]
            img = cv2.GaussianBlur(img, (5, 5), 0)
            if self.augment:
                img, yi = random_shift(img, yi)
                if random.random() <0.6:
                    img = generate_shadow(img)
                if random.random() <0.5:
                    yi = -yi
                    img = np.fliplr(img)
            batch_x.append(img)
            batch_y.append(yi)     
        return np.array(batch_x, dtype=np.uint8) ,  np.array(batch_y, dtype=np.float32) 


def generator(x, y, batch_size, augment=False, repeat=1):
    return DrivingDatasetGenerator(x, y, batch_size, augment, repeat = repeat)
    
def create_dataset(images, measurements, batch_size, repeat_train_data=1):
    images, measurements = shuffle(images, measurements)
    train_images = images[:int(0.80 * len(images))]
    test_images = images[int(0.80 * len(images)): int(0.90 * len(images))]
    valid_images = images[int(0.90 * len(images)):]
    
    train_measurements = measurements[:int(0.80 * len(measurements))]
    test_measurements = measurements[int(0.80 * len(measurements)): int(0.90 * len(measurements))]
    valid_measurements = measurements[int(0.90 * len(measurements)):]
    
    train_dataset = generator(train_images, train_measurements, batch_size, True, repeat_train_data)
    valid_dataset = generator(valid_images, valid_measurements, batch_size)
    test_dataset =  generator(test_images, test_measurements, batch_size)
    return train_dataset, valid_dataset, test_dataset

def data_augmentation(x):
    x = tf.image.random_brightness(x, 0.10)
    hue = lambda x: tf.image.random_hue(x, 0.5)
    x = tf.map_fn(hue, x)
    sat = lambda x: tf.image.random_saturation(x, 0.7, 1.3)
    x = tf.map_fn(sat, x)
    #x = tf.image.random_hue(x, 0.1) 
    #x = tf.image.random_saturation(x, 0.7, 1.3)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    return x


