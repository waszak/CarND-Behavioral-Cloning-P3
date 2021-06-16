from sklearn.utils import shuffle
import tensorflow as tf
import random

def create_dataset(images, measurements, batch_size):
    images, measurements = shuffle(images, measurements)
    train_images = images[:int(0.70 * len(images))]
    test_images = images[int(0.70 * len(images)): int(0.85 * len(images))]
    valid_images = images[int(0.85 * len(images)):]
    
    train_measurements = measurements[:int(0.70 * len(measurements))]
    test_measurements = measurements[int(0.70 * len(measurements)): int(0.85 * len(measurements))]
    valid_measurements = measurements[int(0.85 * len(measurements)):]
    
    train_dataset = dataset(train_images, train_measurements, batch_size, True)
    valid_dataset = dataset(valid_images, valid_measurements, batch_size)
    test_dataset =  dataset(test_images, test_measurements, batch_size)
    return train_dataset, valid_dataset, test_dataset

def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    return image, label

def data_augmentation(x, y):
    #seed = (1,2)
    if tf.random.uniform([]) < 0.5:
        y = -y
        x = tf.image.flip_left_right(x)

    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_hue(x, 0.2) 
    x = tf.image.random_saturation(x, 0.2, 1.0)

    return x, y 

def dataset(x, y, batch_size, agument=False):
    ds =(
            tf.data.Dataset.from_tensor_slices((x, y))
            #.shuffle(len(images))
            #.batch(batch_size)
            .map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            #.cache()
            .shuffle(buffer_size=1000)
            .batch(batch_size)
        )
    if agument:
        ds = ds.map(lambda x, y: data_augmentation(x, y), 
                num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(2)
        
            
    return ds.prefetch(tf.data.experimental.AUTOTUNE)