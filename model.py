import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pickle
import numpy as np
import cv2
import gdown

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout
from keras.layers import Cropping2D, Conv2D
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
from utils import get_data, benchmark, generate_shadow, random_shift
from dataset import create_dataset, DrivingDatasetGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

def create_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1))
    #model.add(Lambda(lambda x: (x - K.constant([123.68, 116.779, 103.939]))/K.constant([58.393, 57.12, 57.375])))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
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
    #example_augument()
    
    images = []
    measurements = []
    batch_size = 128
    num_rows = 0
    epochs = 15
    repeat_train_data = 4
    save_file = 'model.h5'
    download_folder ='/opt'
    data_folder = '/opt/data'
    url = 'https://drive.google.com/u/0/uc?id=1IsUPKgV2r4sni4A1DiLco9gfqDOQsCdX&export=download'
    """
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    if not os.path.exists(data_folder):
        gdown.download(url, os.path.join(download_folder, 'data.zip'), quiet=False)
        print('unpack data')
        return
    """    
    
   
    print('Process csv files')
    num_rows, images, measurements = get_data(data_folder)
    print('Number of rows: ' + str(num_rows))
    print('Prepare datasets')
    train_dataset, valid_dataset, test_dataset = create_dataset(images, measurements, batch_size, repeat_train_data)
    print('Train dataset size', train_dataset.total_size())    
    print('Create model')
   
    model = create_model()
    
    stopping_callback = EarlyStopping(monitor='val_loss', patience=3 )#,restore_best_weights=True
    
    checkpoint_callback = ModelCheckpoint(
        filepath=save_file,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
   
    
    opt = Adam()
    model.compile(loss='mse', metrics=['accuracy'], optimizer=opt)
    
    print('Train model')
    model.fit_generator(generator=train_dataset, steps_per_epoch=len(train_dataset), epochs=epochs, validation_data=valid_dataset,  callbacks=[checkpoint_callback, stopping_callback ] , validation_steps = len(valid_dataset), verbose=1,  use_multiprocessing=True,workers=6 ) #validation_data=validation_generator, use_multiprocessing=True,workers=6 
    
    #model.fit( train_dataset, validation_data=valid_dataset, epochs=epochs,  callbacks=[checkpoint_callback, stopping_callback ] , verbose=2,use_multiprocessing=True, workers=12) 
 
        
if __name__ == "__main__":
    main()
    





    

    


 







#save_model(model)



    
