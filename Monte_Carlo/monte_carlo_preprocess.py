#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:13:27 2022

@author: jpw25
"""

import numpy as np 
import scipy.io as scio
import matplotlib.pyplot as plt
from read_dicts import train_sig, train_conc, train_ksw
from Data_GAN_MRF import y_train
from skimage import feature, transform, draw

###Load images for masks###
image = y_train[:,:,:,0]
image_martinos = np.delete(image, range(197, 222, 1), axis = 0) #Seperate Martinos images
image_tub = np.delete(image, range(0, 197, 1,), axis = 0) #Seperate Tubingen images
image = image*1000 #To improve edge detection 

train_sig = np.swapaxes(train_sig, 0, 1)

###Mask data###
dict_conc = []
dict_ksw = []
dict_sig = []

for i in range(np.size(image_martinos, 0)):
    slice = image_martinos[i,:,:]
    
    conc_slice = np.zeros(slice.shape)
    ksw_slice = np.zeros(slice.shape)
    sig_slice = np.zeros((128,128,30))
    
    edges = feature.canny(slice, sigma=1)
    hough_radii = np.arange(3, 5, 1) 
    hough_res = transform.hough_circle(edges, hough_radii)
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, min_xdistance = 5, min_ydistance = 7, total_num_peaks = 6)
    
    for i in range(0, 6):
        if radii[i] < 4:
            radii[i] = 4
    
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy_1, circx_1 = draw.circle(cy[0], cx[0], radii[0])
        circy_2, circx_2 = draw.circle(cy[1], cx[1], radii[1])
        circy_3, circx_3 = draw.circle(cy[2], cx[2], radii[2])
        circy_4, circx_4 = draw.circle(cy[3], cx[3], radii[3])
        circy_5, circx_5 = draw.circle(cy[4], cx[4], radii[4])
        circy_6, circx_6 = draw.circle(cy[5], cx[5], radii[5])
        
    rand = np.random.randint(0, 167, 6)
    
    conc_slice[circy_1, circx_1] = train_conc[rand[0]]
    conc_slice[circy_2, circx_2] = train_conc[rand[1]]
    conc_slice[circy_3, circx_3] = train_conc[rand[2]]
    conc_slice[circy_4, circx_4] = train_conc[rand[3]]
    conc_slice[circy_5, circx_5] = train_conc[rand[4]]
    conc_slice[circy_6, circx_6] = train_conc[rand[5]]
    
    dict_conc.append(conc_slice)
    
    ksw_slice[circy_1, circx_1] = train_ksw[rand[0]]
    ksw_slice[circy_2, circx_2] = train_ksw[rand[1]]
    ksw_slice[circy_3, circx_3] = train_ksw[rand[2]]
    ksw_slice[circy_4, circx_4] = train_ksw[rand[3]]
    ksw_slice[circy_5, circx_5] = train_ksw[rand[4]]
    ksw_slice[circy_6, circx_6] = train_ksw[rand[5]]
    
    dict_ksw.append(ksw_slice)
    
    sig_slice[circy_1, circx_1] = train_sig[rand[0], :]
    sig_slice[circy_2, circx_2] = train_sig[rand[1], :]
    sig_slice[circy_3, circx_3] = train_sig[rand[2], :]
    sig_slice[circy_4, circx_4] = train_sig[rand[3], :]
    sig_slice[circy_5, circx_5] = train_sig[rand[4], :]
    sig_slice[circy_6, circx_6] = train_sig[rand[5], :]
    
    dict_sig.append(sig_slice)
    
for i in range(np.size(image_tub, 0)):
    slice = image_tub[i,:,:]
    
    conc_slice = np.zeros(slice.shape)
    ksw_slice = np.zeros(slice.shape)
    sig_slice = np.zeros((128,128,30))
    
    edges = feature.canny(slice, sigma=1)
    hough_radii = np.arange(3, 5, 1) 
    hough_res = transform.hough_circle(edges, hough_radii)
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, min_xdistance = 5, min_ydistance = 7, total_num_peaks = 7)
    
    for i in range(0, 7):
        if radii[i] < 4:
            radii[i] = 4
    
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy_1, circx_1 = draw.circle(cy[0], cx[0], radii[0])
        circy_2, circx_2 = draw.circle(cy[1], cx[1], radii[1])
        circy_3, circx_3 = draw.circle(cy[2], cx[2], radii[2])
        circy_4, circx_4 = draw.circle(cy[3], cx[3], radii[3])
        circy_5, circx_5 = draw.circle(cy[4], cx[4], radii[4])
        circy_6, circx_6 = draw.circle(cy[5], cx[5], radii[5])
        circy_7, circx_7 = draw.circle(cy[6], cx[6], radii[6])
        
    rand = np.random.randint(0, 167, 7)
    
    conc_slice[circy_1, circx_1] = train_conc[rand[0]]
    conc_slice[circy_2, circx_2] = train_conc[rand[1]]
    conc_slice[circy_3, circx_3] = train_conc[rand[2]]
    conc_slice[circy_4, circx_4] = train_conc[rand[3]]
    conc_slice[circy_5, circx_5] = train_conc[rand[4]]
    conc_slice[circy_6, circx_6] = train_conc[rand[5]]
    conc_slice[circy_7, circx_7] = train_conc[rand[6]]
    
    dict_conc.append(conc_slice)
    
    ksw_slice[circy_1, circx_1] = train_ksw[rand[0]]
    ksw_slice[circy_2, circx_2] = train_ksw[rand[1]]
    ksw_slice[circy_3, circx_3] = train_ksw[rand[2]]
    ksw_slice[circy_4, circx_4] = train_ksw[rand[3]]
    ksw_slice[circy_5, circx_5] = train_ksw[rand[4]]
    ksw_slice[circy_6, circx_6] = train_ksw[rand[5]]
    ksw_slice[circy_7, circx_7] = train_ksw[rand[6]]
    
    dict_ksw.append(ksw_slice)
    
    sig_slice[circy_1, circx_1] = train_sig[rand[0], :]
    sig_slice[circy_2, circx_2] = train_sig[rand[1], :]
    sig_slice[circy_3, circx_3] = train_sig[rand[2], :]
    sig_slice[circy_4, circx_4] = train_sig[rand[3], :]
    sig_slice[circy_5, circx_5] = train_sig[rand[4], :]
    sig_slice[circy_6, circx_6] = train_sig[rand[5], :]
    sig_slice[circy_7, circx_7] = train_sig[rand[6], :]
    
    dict_sig.append(sig_slice)
    
dict_conc = np.array(dict_conc)
dict_ksw = np.array(dict_ksw)
dict_sig = np.array(dict_sig)

dict_cest = np.stack((dict_conc, dict_ksw), axis = 3)

###Augment data###
dict_sig_flip_x = np.flip(dict_sig, axis = 1)
dict_sig_flip_y = np.flip(dict_sig, axis = 2)
dict_sig_translate_up = np.delete(np.pad(dict_sig, ((0,0), (0,0), (0,10), (0,0))), range(0,10), axis = 2)
dict_sig_translate_down = np.delete(np.pad(dict_sig, ((0,0), (0,0), (10,0), (0,0))), range(118,128), axis = 2)
dict_sig_translate_left = np.delete(np.pad(dict_sig, ((0,0), (0,10), (0,0), (0,0))), range(0,10), axis = 1)
dict_sig_translate_right = np.delete(np.pad(dict_sig, ((0,0), (10,0), (0,0), (0,0))), range(118,128), axis = 1)

dict_cest_flip_x = np.flip(dict_cest, axis = 1)
dict_cest_flip_y = np.flip(dict_cest, axis = 2)
dict_cest_translate_up = np.delete(np.pad(dict_cest, ((0,0), (0,0), (0,10), (0,0))), range(0,10), axis = 2)
dict_cest_translate_down = np.delete(np.pad(dict_cest, ((0,0), (0,0), (10,0), (0,0))), range(118,128), axis = 2)
dict_cest_translate_left = np.delete(np.pad(dict_cest, ((0,0), (0,10), (0,0), (0,0))), range(0,10), axis = 1)
dict_cest_translate_right = np.delete(np.pad(dict_cest, ((0,0), (10,0), (0,0), (0,0))), range(118,128), axis = 1)

###Concatenate augmented data###
dict_cest = np.concatenate((dict_cest, dict_cest_flip_x, dict_cest_flip_y, dict_cest_translate_up, dict_cest_translate_down, dict_cest_translate_left, dict_cest_translate_right), axis = 0)
dict_sig_30 = np.concatenate((dict_sig, dict_sig_flip_x, dict_sig_flip_y, dict_sig_translate_up, dict_sig_translate_down, dict_sig_translate_left, dict_sig_translate_right), axis = 0)

###Define normalization functions###
def normalize_range_y(original_array, original_min, original_max, new_min, new_max):
    a = original_min
    b = original_max
    c = new_min
    d = new_max
    return (original_array - a) / (b - a) * (d - c) + c


def un_normalize_range_y(normalized_array, original_min, original_max, new_min, new_max):
    a = original_min
    b = original_max
    c = new_min
    d = new_max
    return (normalized_array - c) / (d - c) * (b - a) + a

###Normalize outputs###
dict_conc = np.reshape(np.delete(dict_cest, 1, axis=3), (1554, 128, 128))
dict_ksw = np.reshape(np.delete(dict_cest, 0, axis=3), (1554, 128, 128))

dict_conc = normalize_range_y(dict_conc, original_min = 0, original_max = 130, new_min = 0, new_max = 1)
dict_ksw = normalize_range_y(dict_ksw, original_min = 0, original_max = 1500, new_min = 0, new_max = 1)

dict_cest = np.stack((dict_conc, dict_ksw), axis=3)

###Clip and normalize inputs###
dict_sig_30 = np.swapaxes(dict_sig_30, 0, 3)
dict_sig_30 = np.swapaxes(dict_sig_30, 1, 3)
dict_sig_30 = np.swapaxes(dict_sig_30, 2, 3)

dict_sig_9 = np.delete(dict_sig_30, [*range(9, 30, 1)], axis=0)

dict_sig_30 = dict_sig_30 / np.sqrt(np.sum(dict_sig_30 ** 2, axis=0))
dict_sig_30[np.isnan(dict_sig_30)] = 0 

dict_sig_30 = np.swapaxes(dict_sig_30, 0, 3)
dict_sig_30 = np.swapaxes(dict_sig_30, 0, 1)
dict_sig_30 = np.swapaxes(dict_sig_30, 1, 2)

dict_sig_9 = dict_sig_9 / np.sqrt(np.sum(dict_sig_9 ** 2, axis=0))
dict_sig_9[np.isnan(dict_sig_9)] = 0 

dict_sig_9 = np.swapaxes(dict_sig_9, 0, 3)
dict_sig_9 = np.swapaxes(dict_sig_9, 0, 1)
dict_sig_9 = np.swapaxes(dict_sig_9, 1, 2)

np.save('dict_cest', dict_cest)
np.save('dict_sig_30', dict_sig_30)
np.save('dict_sig_9', dict_sig_9)

        