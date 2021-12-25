# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 18:30:44 2021

@author: jrose
"""
import numpy as np
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image

def cv2_read_path_train(set_type, stages):
    ''' Processes images given the file path and naming conventions'''
    
    # list to hold image data - converted to numpy array on return
    image_array = []
    
    # list to hold labels - converted to numpy array on return
    label_array = []
    
    # Label encoding dictionary
    name_dict = {'NonDemented' : 0,
                 'VeryMildDemented' : 1,
                 'MildDemented' : 2,
                 'ModerateDemented' : 3}
    
    # Iterate through stages
    for stage in stages:
        # Iterate through every image in specified directory
        for filename in glob.glob(f'Alzheimers_Dataset/{set_type}/{stage}/*.jpg'):
            # Read image
            IMG_SIZE = 100
            img = cv2.imread(filename, 0)
            new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Add image to list, SCALED
            image_array.append(new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255)
            # Add label
            label_array.append(name_dict[stage])

    return np.array(image_array)[:,0,:,:,:], np.array(label_array)             

# Name of dir for each stage
all_stages = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# Create trainng sets
X_train, y_train = cv2_read_path_train(set_type='train', stages=all_stages)

# convert labels to binary identification
train_labels = pd.Series(y_train)
y_train = pd.get_dummies(train_labels)

# Create testing sets
X_test, y_test = cv2_read_path_train(set_type='test', stages=all_stages)

# convert labels to binary identification
test_labels = pd.Series(y_test)
y_test = pd.get_dummies(test_labels)

# Function to show image from dataset
def show_sample(data, index):
     plt.imshow(data[index], cmap=plt.cm.binary)
     plt.show()

# View some TRAINING MRI scans
print('Training Set:')
for n in range(1,21):
    show_sample(X_train, -n)
    print(f'Label - {y_train.iloc[-n]}')
    
time.sleep(3)

# View some TESTING MRI scans
print('Testing Set:')
for n in range(1,21):
    show_sample(X_test, -n)
    print(f'Label - {y_test.iloc[-n]}')
    
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

