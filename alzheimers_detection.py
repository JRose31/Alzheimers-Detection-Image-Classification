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
            img = cv2.imread(filename, cv2.COLOR_BGR2RGB)
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
def show_sample(data, index, label='Empty'):
     plt.imshow(data[index], cmap=plt.cm.plasma)
     plt.title(f"[{index}] : {label}")
     plt.show()

# View some TRAINING MRI scans
print('Training Set...')
for n in range(1,21):
    sample_label = np.argmax(y_train.iloc[n])
    show_sample(X_train, n, label=sample_label)
print('Complete...') 

time.sleep(3)

# View some TESTING MRI scans
print('Testing Set...')
for n in range(1,21):
    sample_label = np.argmax(y_test.iloc[-n])
    show_sample(X_test, -n, label=sample_label)
print('Complete...')
    
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit multiple models (mid and late models train more even distribution of mild and moderate demented)
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=64)

mid_model = Sequential()
mid_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(100, 100, 1)))
mid_model.add(MaxPooling2D(pool_size=(2, 2)))
mid_model.add(Dropout(0.25))
mid_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
mid_model.add(MaxPooling2D(pool_size=(2, 2)))
mid_model.add(Dropout(0.25))
mid_model.add(Flatten())
mid_model.add(Dense(64, activation='relu'))
mid_model.add(Dropout(0.5))
mid_model.add(Dense(32, activation='relu'))
mid_model.add(Dropout(0.5))
mid_model.add(Dense(4, activation='softmax'))

mid_model.summary()

mid_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

mid_model.fit(X_train[len(X_train)//2:], y_train[len(y_train)//2:], epochs=30, validation_data=(X_test[len(X_test)//2:], y_test[len(y_test)//2:]), batch_size=64)

late_model = Sequential()
late_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(100, 100, 1)))
late_model.add(MaxPooling2D(pool_size=(2, 2)))
late_model.add(Dropout(0.25))
late_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
late_model.add(MaxPooling2D(pool_size=(2, 2)))
late_model.add(Dropout(0.25))
late_model.add(Flatten())
late_model.add(Dense(64, activation='relu'))
late_model.add(Dropout(0.5))
late_model.add(Dense(32, activation='relu'))
late_model.add(Dropout(0.5))
late_model.add(Dense(4, activation='softmax'))

late_model.summary()

late_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

late_model.fit(X_train[len(X_train)//4:], y_train[len(y_train)//4:], epochs=30, validation_data=(X_test[len(X_test)//4:], y_test[len(y_test)//4:]), batch_size=64)

vloss, vacc = model.evaluate(X_test, y_test)
mid_vloss, mid_vacc = mid_model.evaluate(X_test, y_test)
late_vloss, late_vacc = late_model.evaluate(X_test, y_test)

print(f'Initial >>> Loss: {vloss}, Accuracy: {vacc}')
print(f'Mid >>> Loss: {mid_vloss}, Accuracy: {mid_vacc}')
print(f'Late >>> Loss: {late_vloss}, Accuracy: {late_vacc}')

def make_ts_prediction(idx):
    # map index/encoded value with stage label
    label_dict = {0 : 'NonDemented',
                  1 : 'VeryMildDemented',
                  2 : 'MildDemented',
                  3 : 'ModerateDemented'}
    
    # USE AVG OF SCORES FOR MORE ACCURATE VOTING
    init_mp = model.predict(X_test[idx:idx+1])
    mid_mp = mid_model.predict(X_test[idx:idx+1])
    late_mp = late_model.predict(X_test[idx:idx+1])
    ensemble = (init_mp + mid_mp + late_mp)/3
    
    # Get prediction as most voted
    pred_label = label_dict[np.argmax(ensemble)]
    true_label = label_dict[np.argmax(y_test[idx:idx+1])]
    
    #Show sample
    show_sample(X_test, idx, label=true_label)
    
    # Compare labels, print prediction values
    if pred_label == true_label:
        return f'Correct prediction on {true_label}'
    else:
        return f'False prediction on {true_label}'
    
    
# Make some predictions 
for idx in range(0,20):
    print(make_ts_prediction(idx))    

for idx in range(1259,1279):
    print(make_ts_prediction(idx))