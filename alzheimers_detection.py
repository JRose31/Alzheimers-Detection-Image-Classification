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
    
    # Dictionary to map folder name (dementia stage) with image naming convention
    name_dict = {'NonDemented' : 'nonDem',
                 'VeryMildDemented' : 'verymildDem',
                 'MildDemented' : 'mildDem',
                 'ModerateDemented' : 'moderateDem'}
    
    # Using image count to get specific number of images in specified folder
    for idx in range(img_count+1):
        # Match folder name (stage) and image naming convention
        for val in name_dict.keys():
            if stage == val:
                # Construct file path
                img_name = name_dict[val]
                img_path = f'Alzheimers_Dataset/{set_type}/{stage}/{img_name+str(idx)}.jpg'
                # Read and show image
                IMG_SIZE = 100
                img = cv2.imread(img_path, 0)
                new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                image_array.append(new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255)

    return np.array(image_array)[:,0,:,:,:]             


nonDemdata = cv2_read_path(set_type='train',stage='NonDemented', img_count=2559)
verymildDemdata = cv2_read_path(set_type='train',stage='VeryMildDemented', img_count=1791)
mildDemdata = cv2_read_path(set_type='train',stage='MildDemented', img_count=716)
moderateDemdata = cv2_read_path(set_type='train',stage='ModerateDemented', img_count=51)

def show_sample(data, index):
    plt.imshow(data[index], cmap=plt.cm.binary)
    plt.show()
    
for n in range(500):
    show_sample(nonDemdata, n)
