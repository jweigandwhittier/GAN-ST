"""
Created on Thu Oct 14 13:01:04 2021

"""

import numpy as np

###Import input data, concatenate ground truth outputs###
x_train = np.load('Data''/sample_phantom_input.npy')
y_train = np.load('Data''/sample_phantom_gt_ksw.npy') #Load this if you would like to train on CEST-MRF outputs (ksw and concentration)
#y_train = np.load('Data''/sample_phantom_gt_ph.npy') #Load this if you would like to train on measured outputs (pH and concentration)

###Augment data with flips and translations###
x_train_flip_x = np.flip(x_train, axis = 1) #Flip input over x-axis
x_train_flip_y = np.flip(x_train, axis = 2) #Flip input over y-axis
x_train_translate_up = np.delete(np.pad(x_train, ((0,0), (0,0), (0,10), (0,0))), range(0,10), axis = 2) #Translate input +y 10 pixels
x_train_translate_down = np.delete(np.pad(x_train, ((0,0), (0,0), (10,0), (0,0))), range(118,128), axis = 2) #Translate input -y 10 pixels
x_train_translate_left = np.delete(np.pad(x_train, ((0,0), (0,10), (0,0), (0,0))), range(0,10), axis = 1) #Translate input -x 10 pixels
x_train_translate_right = np.delete(np.pad(x_train, ((0,0), (10,0), (0,0), (0,0))), range(118,128), axis = 1) #Translate input +x 10 pixels

y_train_flip_x = np.flip(y_train, axis = 1) #Flip outputs over x-axis
y_train_flip_y = np.flip(y_train, axis = 2) #Flip outputs over y-axis
y_train_translate_up = np.delete(np.pad(y_train, ((0,0), (0,0), (0,10), (0,0))), range(0,10), axis = 2) #Translate outputs +y 10 pixels
y_train_translate_down = np.delete(np.pad(y_train, ((0,0), (0,0), (10,0), (0,0))), range(118,128), axis = 2) #Translate outputs -y 10 pixels
y_train_translate_left = np.delete(np.pad(y_train, ((0,0), (0,10), (0,0), (0,0))), range(0,10), axis = 1) #Translate outputs -x 10 pixels
y_train_translate_right = np.delete(np.pad(y_train, ((0,0), (10,0), (0,0), (0,0))), range(118,128), axis = 1) #Translate outputs +x 10 pixels

###Concatenate augmented data###
y_train_aug = np.concatenate((y_train, y_train_flip_x, y_train_flip_y, y_train_translate_up, y_train_translate_down, y_train_translate_left, y_train_translate_right), axis = 0)
x_train_aug = np.concatenate((x_train, x_train_flip_x, x_train_flip_y, x_train_translate_up, x_train_translate_down, x_train_translate_left, x_train_translate_right), axis = 0)