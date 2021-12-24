# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 18:30:44 2021

@author: jrose
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import time

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

# Create testing sets
X_test, y_test = cv2_read_path_train(set_type='test', stages=all_stages)

# Function to show image from dataset
def show_sample(data, index):
     plt.imshow(data[index], cmap=plt.cm.binary)
     plt.show()

# View some TRAINING MRI scans
print('Training Set:')
for n in range(1,21):
    show_sample(X_train, -n)
    print(f'Label - {y_train[-n]}')
    
time.sleep(3)

# View some TESTING MRI scans
print('Testing Set:')
for n in range(1,21):
    show_sample(X_test, -n)
    print(f'Label - {y_test[-n]}')
    
