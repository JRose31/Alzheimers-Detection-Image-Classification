# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 18:30:44 2021

@author: jrose
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

def cv2_read_path(set_type, stage, img_count):
    ''' Processes images given the file path and naming conventions'''
    
    image_array = []
    
    label_array = []
    # Dictionary to map folder name (dementia stage) with image naming convention
    name_dict = {'NonDemented' : ['nonDem', 0],
                 'VeryMildDemented' : ['verymildDem', 1],
                 'MildDemented' : ['mildDem', 2],
                 'ModerateDemented' : ['moderateDem', 3]}
    
    # Using image count to get specific number of images in specified folder
    for idx in range(img_count+1):
        # Match folder name (stage) and image naming convention
        for val in name_dict.keys():
            if stage == val:
                # Construct file path
                img_name = name_dict[val][0]
                img_path = f'Alzheimers_Dataset/{set_type}/{stage}/{img_name+str(idx)}.jpg'
                # Read and show image
                IMG_SIZE = 100
                img = cv2.imread(img_path, 0)
                new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                image_array.append(new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255)
                # Add label
                label_array.append(name_dict[val][1])

    return np.array(image_array)[:,0,:,:,:], label_array             


nonDemdataX, nonDemdatay = cv2_read_path(set_type='train',stage='NonDemented', img_count=2559)
verymildDemdataX, verymildDemdatay = cv2_read_path(set_type='train',stage='VeryMildDemented', img_count=1791)
mildDemdataX, mildDemdatay = cv2_read_path(set_type='train',stage='MildDemented', img_count=716)
moderateDemdataX, moderateDemdatay = cv2_read_path(set_type='train',stage='ModerateDemented', img_count=51)

def show_sample(data, index):
    plt.imshow(data[index], cmap=plt.cm.binary)
    plt.show()
    
for n in range(20):
    show_sample(moderateDemdataX, n)
    print(f'Label: {moderateDemdatay[n]}')
