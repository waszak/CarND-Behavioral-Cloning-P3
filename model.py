import csv
import pickle
import cv2
import numpy as np
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout
from keras.layers import Cropping2D, Conv2D
from keras.optimizers import Adam
from keras import backend as K

images = []
measurements = []

def get_data(direct, images, measurement):

    with open(direct+'\\driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            paths= [direct+'\\IMG\\'+x.split('\\')[-1] for x in row[0:3]]
            #print(paths)
            measurement = [float(x) for x in row[3:]][0]
            measurements.append(measurement)
            image = mpimg.imread(paths[0]) 
            blur = cv2.GaussianBlur(image, (5, 5), 0)
            images.append(blur )

            
            image_flipped = np.fliplr(image)
            blur_flipped = cv2.GaussianBlur(image_flipped, (5, 5), 0)
            measurement_flipped = -measurement
            images.append(blur_flipped )
            measurements.append(measurement_flipped)
            
get_data('images\\1',images,measurements)
get_data('images\\2',images,measurements)   
get_data('images\\3',images,measurements)
get_data('images\\4',images,measurements)        

X_train = np.array(images)
#X_train = normalizeVgg(X_train)
y_train = np.array(measurements)
#print(y_train)


model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
#model.add(Lambda(lambda x: (x - K.constant([123.68, 116.779, 103.939]))/K.constant([58.393, 57.12, 57.375])))
model.add(Conv2D(24, (5, 5), activation='relu', padding="same"))
model.add(Conv2D(36, (5, 5), activation='relu', padding="same"))
model.add(Conv2D(48, (5, 5), activation='relu', padding="same"))
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))

opt = Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=opt)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5,batch_size=32, verbose=2)
model.save('model.h5')
