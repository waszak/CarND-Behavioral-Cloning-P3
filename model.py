import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pickle
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout,BatchNormalization
from keras.layers import Cropping2D, Conv2D
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
from utils import get_data, benchmark, generate_shadow, random_shift
from dataset import create_dataset, DrivingDataset, data_augmentation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_model():
    model = Sequential()
    model.add(Lambda(lambda x:data_augmentation(x), input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    #model.add(Lambda(lambda x: (x / 127.5) - 1))
    model.add(Lambda(lambda x: (x - K.constant([123.68, 116.779, 103.939]))/K.constant([58.393, 57.12, 57.375])))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    return model

def example_augument():
    shadow = cv2.cvtColor(cv2.imread(r'examples\no_shadows.jpg'), cv2.COLOR_BGR2RGB)
    shadow = cv2.cvtColor(generate_shadow( shadow),  cv2.COLOR_RGB2BGR)
    #shadow = np.fliplr(shadow)
    #shadow = cv2.GaussianBlur(shadow, (5, 5), 0)
    shadow,y = random_shift(shadow)
    cv2.imwrite(r'examples\shadows.jpg', shadow) 
    
def main():
    example_augument()
    
    images = []
    measurements = []
    batch_size = 64
    num_rows = 0
    epochs = 50
    save_file = 'model.h5'
    data_folder = 'data'
   
    print('Process csv files')
    num_rows, images, measurements = get_data(data_folder)
    print('Number of rows: ' + str(num_rows))
    print('Prepare datasets')
    train_dataset, valid_dataset, test_dataset = create_dataset(images, measurements, batch_size)
        
    print('Create model')
   
    model = create_model()
    stopping_callback = EarlyStopping(monitor='val_loss', patience=5 ,restore_best_weights=True)
    
    checkpoint_callback = ModelCheckpoint(
        filepath=save_file,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    opt = Adam(learning_rate=0.0009)
    model.compile(loss='mse', metrics=['accuracy'], optimizer=opt)
    
    print('Train model')
    model.fit( train_dataset, validation_data=valid_dataset, epochs=epochs, verbose=2, callbacks=[checkpoint_callback, stopping_callback ],use_multiprocessing=True, workers=12) 
   
    print('Running test dataset')
    score, acc = model.evaluate(test_dataset, batch_size=batch_size,  verbose = 0)
    print('Test score:', score)
    print('Test accuracy:', acc)
        
if __name__ == "__main__":
    main()
    





    

    


 







#save_model(model)



    
