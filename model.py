import pickle
import cv2
import numpy as np
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout
from keras.layers import Cropping2D, Conv2D
from keras.optimizers import Adam
from keras import backend as K
from utils import get_data
from dataset import create_dataset


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def create_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    #model.add(Lambda(lambda x: (x - K.constant([123.68, 116.779, 103.939]))/K.constant([58.393, 57.12, 57.375])))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.6))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
    
    
def main():
    images = []
    measurements = []
    batch_size = 64
    num_rows = 0
    num_rows, images, measurements = get_data('images')
    #print(len(measurements), len(images))
    #print("Total number of rows: " + str(num_rows))
    train_dataset, valid_dataset, test_dataset = create_dataset(images, measurements, batch_size)
    model = create_model()
    stopping_callback = EarlyStopping(monitor='val_loss', patience=3 ,restore_best_weights=True)
    save_file = 'model.h5'
    checkpoint_callback = ModelCheckpoint(
        filepath=save_file,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mse', metrics=['accuracy'], optimizer=opt)
    
    model.fit( train_dataset, validation_data=valid_dataset, epochs=30, verbose=2, callbacks=[checkpoint_callback, stopping_callback ],use_multiprocessing=True, workers=12) 
    
   
        
if __name__ == "__main__":
    main()
    





    

    


 







#save_model(model)



    
