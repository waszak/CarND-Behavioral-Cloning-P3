
Track 1

https://user-images.githubusercontent.com/3121149/123559293-6f2c2980-d79b-11eb-9d87-a2abaa04f095.mp4

Track 2

https://user-images.githubusercontent.com/3121149/123559333-a995c680-d79b-11eb-80cc-d40ebfcb10f3.mp4


# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/network.jpg "Model Visualization"
[image2]: ./examples/loss.png "Loss"
[image3]: ./examples/no_shadows.jpg "Normal Image"
[image4]: ./examples/shadows.jpg "Image with shadows"
[image5]: ./examples/shadows_flipped.jpg "Flipped image with shadows"
[image6]: ./examples/shift.jpg "Shifted Image with shadows"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* utils.py for method like generating shadow, getting data, random_shift
* dataset.py for generating augumented training set.
* model.h5 containing a trained convolution neural network 
* readme.md writeup
* track1.mp4 recording of the track 1
* track2.mp4 recording of the track 2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. Model architecture is create_model method.

Training is in main function. I used EarlyStopping and ModelCheckpoint for saving in case something goes wrong.
``` python
    model = create_model()
    stopping_callback = EarlyStopping(monitor='val_loss', patience=10 ,restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=save_file,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=False)
    opt = Adam()#
    model.compile(loss='mse', metrics=['accuracy'], optimizer=opt)
    
    history = model.fit_generator(generator=train_dataset, steps_per_epoch=len(train_dataset), epochs=epochs, validation_data=valid_dataset,  callbacks=[checkpoint_callback, stopping_callback ] , validation_steps = len(valid_dataset), verbose=2) #validation_data=validation_generator, use_multiprocessing=True,workers=6 

```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture was inspired by paper NVIDIA's End to End Learning for Self-Driving Cars. With addition of dropout layers and normalization.
I also used augumentation layer.

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers, batch_normalization in order to reduce overfitting (model.py lines 19-40).
There is also VggNormalization. Data augumentation also was used to reduce overfitting.


#### 3. Model parameter tuning

The model used an adam optimizer. I tried few diffrent learning rates but decided to use default one.

#### 4. Appropriate training data

I used a combination of center lane driving( on track2 also left lane), opposite direction also udacity dataset. 
In total I used 45736 training samples. Most of my dataset was generated by using keyboard in simulator.
I used random shift( translation by x, y) to reduce that issue with data augumentation. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Model architecture was inspired  by paper NVIDIA's End to End Learning for Self-Driving Cars.
I made some changes like adding leaky elu, batch normalization, dropout layer to reduce overfitting.
I split my data to training, validation, test dataset. With ratio 80/10/10, to check for overfitting.

Because my model had bias towards going straight. I used data augumentation to reduce that problem.
I used larger number of epochs, and smaller dataset. I decided to use more epochs because I had model checkpoints
I could stop at any time and test my model.


#### 2. Final Model Architecture

The final model architecture
```python
def create_model():
    model = Sequential()
    model.add(Lambda(lambda x:data_augmentation(x), input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))#input_shape=(160,320,3))
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
```
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda (Lambda)              (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d (Cropping2D)      (None, 65, 320, 3)        0
_________________________________________________________________
lambda_1 (Lambda)            (None, 65, 320, 3)        0
_________________________________________________________________
conv2d (Conv2D)              (None, 33, 160, 24)       1824
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 17, 80, 36)        21636
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 9, 40, 48)         43248
_________________________________________________________________
batch_normalization (BatchNo (None, 9, 40, 48)         192
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 9, 40, 64)         27712
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 9, 40, 64)         36928
_________________________________________________________________
flatten (Flatten)            (None, 23040)             0
_________________________________________________________________
dropout (Dropout)            (None, 23040)             0
_________________________________________________________________
batch_normalization_1 (Batch (None, 23040)             92160
_________________________________________________________________
dense (Dense)                (None, 100)               2304100
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0
_________________________________________________________________
batch_normalization_2 (Batch (None, 100)               400
_________________________________________________________________
dense_1 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_2 (Dense)              (None, 10)                510
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 11
=================================================================
Total params: 2,533,771
Trainable params: 2,487,395
Non-trainable params: 46,376
_________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

I used keras sequnce for training set. I augumented each image by adding random shadows, flipping image, random translation, gaussian blur.
Finally I also used random saturation, hue, brightness. I wanted to have model that don't memorize the track.
Code is in dataset.py
I used keras Sequence for class and Dataset with generators for tensorflow2. Second one uses dataset api combined with generators so its more powerfull.
Using sequence allowed me to use multiprocessing and train faster. 
For assignement I decided to use Sequence because it was much better than using pure generators. For tensorflow2(branch tensorflow2) version i used Dataset with generators.

```python 
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
```
Dataset version
```
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
       
```  
I used dataset augumentation like flipinging, translation, random shadows, color saturation, gausian blur, and more.

Image from dataset.

![alt text][image3]
Image with agumented shadows.

![alt text][image4]
Image with augumented shadows and flipped.

![alt text][image5]

Shifted image with shadows

![alt text][image6]


I used 30 epochs because of the time to train each epoch and it was close to optimal number of epochs. It needs about 20-30 epochs to learn second track.
Training loss was higher because of dropout layers and data augumentation. When i used less regularization it needs about 8 epochs.

![alt text][image2]

I used an adam optimizer and after trying multiple learning rates i decided default one was good enough.
