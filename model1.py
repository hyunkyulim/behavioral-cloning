###################### library ################################
import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
import scipy.misc
import csv
import cv2
import numpy as np
import sklearn
import random
#import json
#import os
#import errno
#import pandas 
from sklearn.model_selection import train_test_split
import scipy.misc

############ file address ##########################
log_file = './data/driving_log.csv'
image_file = './data/IMG/'
############# Tune Value ##########################
correction = 0.226
batch_size = 32
number_of_epochs = 2
activation_relu = 'relu'
dimetion = (160, 320)

############# read data #####################
samples =[]
with open(log_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples[0] = samples[1]        
train_samples, validation_samples = train_test_split(samples, test_size=0.1)
############# function #######################

def random_flip (image, angle):
    flip = random.randint(0,1)
    if flip== 1:
        return np.fliplr(image), -1 * angle
    return image, angle

def random_gamma(image):

    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)                    

def generate_new_image(image, angle):
    image, angle = random_flip (image, angle)
    image = random_gamma (image)
    return image, angle


def generator(samples, batch_size=32):
    """
       Actual batch_size is 64 because of data_agument (fliped images)
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                # for center image
                angle = float(batch_sample[3])
                name = image_file + batch_sample[0].split('/')[-1]
                image = cv2.imread(name)
                processed_image, processed_angle = generate_new_image(image, angle)
                images.append(processed_image)
                angles.append(processed_angle)
                    
                # for left image
                name = image_file + batch_sample[1].split('/')[-1]
                angle = float(angle + correction)
                image = cv2.imread(name)
                processed_image, processed_angle = generate_new_image(image, angle)
                images.append(processed_image)
                angles.append(processed_angle)
                
                # for right image
                name = image_file + batch_sample[2].split('/')[-1]
                angle = float(angle - correction )
                image = cv2.imread(name)
                processed_image, processed_angle = generate_new_image(image, angle)
                images.append(processed_image)
                angles.append(processed_angle)
                
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

            
############## model ###########################            
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
# starts with five convolutional and maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Flatten())
# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation(activation_relu))
model.add(Dense(100))
model.add(Activation(activation_relu))
model.add(Dense(50))
model.add(Activation(activation_relu))
model.add(Dense(10))
model.add(Activation(activation_relu))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss="mse", )

# create two generators for training and validation
train_generator = generator (train_samples)
validation_generator = generator (validation_samples)

# generate model
history = model.fit_generator(train_generator,
                              steps_per_epoch = len(train_samples),
                              epochs = number_of_epochs,
                              validation_data = validation_generator,
                              validation_steps= len(validation_samples),
                              verbose = 1) 
# save model
model.save('model1.h5')