#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:05:19 2022

@author: jpw25
"""

import numpy as np 
import scipy.io as scio
import matplotlib.pyplot as plt
from read_dicts import test_dict_conc_vary_conc, test_dict_conc_vary_ksw, test_dict_conc_vary_sig
from read_dicts import test_dict_ksw_vary_conc, test_dict_ksw_vary_ksw, test_dict_ksw_vary_sig
from Data_GAN_MRF import y_valid
from skimage import feature, transform, draw
from keras.models import load_model
from scipy.stats import pearsonr
import pingouin as pg
import pandas as pd

test_dict_conc_vary_sig = np.swapaxes(test_dict_conc_vary_sig, 0, 1)
test_dict_ksw_vary_sig = np.swapaxes(test_dict_ksw_vary_sig, 0, 1)

test_dict_conc_vary_sig = np.expand_dims(test_dict_conc_vary_sig, 2)
test_dict_ksw_vary_sig = np.expand_dims(test_dict_ksw_vary_sig, 2)

###Load models###
model_9 = load_model('Models/model_monte_carlo_9.h5')
model_30 = load_model('Models/model_monte_carlo_30.h5')

##Additive white Gaussian noise functions###
def GenerateNoise(signal, snr_db):
    n = np.zeros((len(signal)))
    sig_avg = np.mean(signal**2)
    sig_avg_db = 10 * np.log10(sig_avg)
    noise_avg_db = sig_avg_db - snr_db
    noise_avg = 10**(noise_avg_db/10)
    noise = np.random.normal(0, np.sqrt(noise_avg), size=(len(signal)))
    return noise

def NoisyArrays(array, repetition):
    noisyarray = []
    for i in range(0, np.size(array, 0)):
        signal = array[i, :, 0]
        noise = GenerateNoise(signal, 53)
        noisy_sig = signal + noise
        noisyarray.append(noisy_sig)
    noisyarray = np.array(noisyarray)
    noisyarray = np.expand_dims(noisyarray, 2)
    array = np.concatenate((array, noisyarray), axis = 2)
    return array

###Create 9 noise input signals###
for i in range(9):
    test_dict_conc_vary_sig = NoisyArrays(test_dict_conc_vary_sig, i)
    test_dict_ksw_vary_sig = NoisyArrays(test_dict_ksw_vary_sig, i)

###Create masks for image generation###
image = y_valid[:,:,:,0]
image = image*1000
image = np.delete(image, range(3, 17), axis=0)

conc_vary_conc_image = []
conc_vary_ksw_image = []
conc_vary_sig_image = []

ksw_vary_conc_image = []
ksw_vary_ksw_image = []
ksw_vary_sig_image = []

masks = []
 
for i in range(np.size(image, 0)):
    index = i
    slice = image[i,:,:]
    
    conc_vary_conc_slice = np.zeros(slice.shape)
    conc_vary_ksw_slice = np.zeros(slice.shape)
    conc_vary_sig_slice = np.zeros((128,128,30,10))
    
    ksw_vary_conc_slice = np.zeros(slice.shape)
    ksw_vary_ksw_slice = np.zeros(slice.shape)
    ksw_vary_sig_slice = np.zeros((128,128,30,10))
    
    vial_1_mask = np.ones(slice.shape)
    vial_2_mask = np.ones(slice.shape)
    vial_3_mask = np.ones(slice.shape)
    vial_4_mask = np.ones(slice.shape)
    vial_5_mask = np.ones(slice.shape)
    vial_6_mask = np.ones(slice.shape)
    
    edges = feature.canny(slice, sigma=1)
    hough_radii = np.arange(3, 5, 1) 
    hough_res = transform.hough_circle(edges, hough_radii)
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, min_xdistance = 5, min_ydistance = 7, total_num_peaks = 6)
    
    for i in range(0, 6):
        if radii[i] < 4:
            radii[i] = 4
            
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = draw.disk((center_y, center_x), radius, shape=slice.shape)
        
        for i in range(0, 6):
            if cy[i] in range(60,70) and cx[i] in range(40,60):
                circy_1, circx_1 = draw.disk((cy[i], cx[i]), radii[i])
            if cy[i] in range(45,55) and cx[i] in range(40,60):
                circy_2, circx_2 = draw.disk((cy[i], cx[i]), radii[i])
            if cy[i] in range(35,45) and cx[i] in range(60,80):
                circy_3, circx_3 = draw.disk((cy[i], cx[i]), radii[i])
            if cy[i] in range(65,75) and cx[i] in range(70,90):
                circy_4, circx_4 = draw.disk((cy[i], cx[i]), radii[i])
            if cy[i] in range(45, 55) and cx[i] in range(75,95):
                circy_5, circx_5 = draw.disk((cy[i], cx[i]), radii[i])
            if cy[i] in range(70,80) and cx[i] in range(55,75):
                circy_6, circx_6 = draw.disk((cy[i], cx[i]), radii[i]) 
            
    if index == 0:
        conc_vary_conc_slice[circy_1, circx_1] = test_dict_conc_vary_conc[0]
        conc_vary_conc_slice[circy_2, circx_2] = test_dict_conc_vary_conc[1]
        conc_vary_conc_slice[circy_3, circx_3] = test_dict_conc_vary_conc[2]
        conc_vary_conc_slice[circy_4, circx_4] = test_dict_conc_vary_conc[3]
        conc_vary_conc_slice[circy_5, circx_5] = test_dict_conc_vary_conc[4]
        conc_vary_conc_slice[circy_6, circx_6] = test_dict_conc_vary_conc[5]
        
        conc_vary_ksw_slice[circy_1, circx_1] = test_dict_conc_vary_ksw[0]
        conc_vary_ksw_slice[circy_2, circx_2] = test_dict_conc_vary_ksw[1]
        conc_vary_ksw_slice[circy_3, circx_3] = test_dict_conc_vary_ksw[2]
        conc_vary_ksw_slice[circy_4, circx_4] = test_dict_conc_vary_ksw[3]
        conc_vary_ksw_slice[circy_5, circx_5] = test_dict_conc_vary_ksw[4]
        conc_vary_ksw_slice[circy_6, circx_6] = test_dict_conc_vary_ksw[5]
        
        ksw_vary_conc_slice[circy_1, circx_1] = test_dict_ksw_vary_conc[0]
        ksw_vary_conc_slice[circy_2, circx_2] = test_dict_ksw_vary_conc[1]
        ksw_vary_conc_slice[circy_3, circx_3] = test_dict_ksw_vary_conc[2]
        ksw_vary_conc_slice[circy_4, circx_4] = test_dict_ksw_vary_conc[3]
        ksw_vary_conc_slice[circy_5, circx_5] = test_dict_ksw_vary_conc[4]
        ksw_vary_conc_slice[circy_6, circx_6] = test_dict_ksw_vary_conc[5]
        
        ksw_vary_ksw_slice[circy_1, circx_1] = test_dict_ksw_vary_ksw[0]
        ksw_vary_ksw_slice[circy_2, circx_2] = test_dict_ksw_vary_ksw[1]
        ksw_vary_ksw_slice[circy_3, circx_3] = test_dict_ksw_vary_ksw[2]
        ksw_vary_ksw_slice[circy_4, circx_4] = test_dict_ksw_vary_ksw[3]
        ksw_vary_ksw_slice[circy_5, circx_5] = test_dict_ksw_vary_ksw[4]
        ksw_vary_ksw_slice[circy_6, circx_6] = test_dict_ksw_vary_ksw[5]
        
        conc_vary_sig_slice[circy_1, circx_1] = test_dict_conc_vary_sig[0,:,:]
        conc_vary_sig_slice[circy_2, circx_2] = test_dict_conc_vary_sig[1,:,:]
        conc_vary_sig_slice[circy_3, circx_3] = test_dict_conc_vary_sig[2,:,:]
        conc_vary_sig_slice[circy_4, circx_4] = test_dict_conc_vary_sig[3,:,:]
        conc_vary_sig_slice[circy_5, circx_5] = test_dict_conc_vary_sig[4,:,:]
        conc_vary_sig_slice[circy_6, circx_6] = test_dict_conc_vary_sig[5,:,:]
        
        ksw_vary_sig_slice[circy_1, circx_1] = test_dict_ksw_vary_sig[0,:,:]
        ksw_vary_sig_slice[circy_2, circx_2] = test_dict_ksw_vary_sig[1,:,:]
        ksw_vary_sig_slice[circy_3, circx_3] = test_dict_ksw_vary_sig[2,:,:]
        ksw_vary_sig_slice[circy_4, circx_4] = test_dict_ksw_vary_sig[3,:,:]
        ksw_vary_sig_slice[circy_5, circx_5] = test_dict_ksw_vary_sig[4,:,:]
        ksw_vary_sig_slice[circy_6, circx_6] = test_dict_ksw_vary_sig[5,:,:]
        
    elif index == 1:
        conc_vary_conc_slice[circy_1, circx_1] = test_dict_conc_vary_conc[6]
        conc_vary_conc_slice[circy_2, circx_2] = test_dict_conc_vary_conc[7]
        conc_vary_conc_slice[circy_3, circx_3] = test_dict_conc_vary_conc[8]
        conc_vary_conc_slice[circy_4, circx_4] = test_dict_conc_vary_conc[9]
        conc_vary_conc_slice[circy_5, circx_5] = test_dict_conc_vary_conc[10]
        conc_vary_conc_slice[circy_6, circx_6] = test_dict_conc_vary_conc[0]
        
        conc_vary_ksw_slice[circy_1, circx_1] = test_dict_conc_vary_ksw[6]
        conc_vary_ksw_slice[circy_2, circx_2] = test_dict_conc_vary_ksw[7]
        conc_vary_ksw_slice[circy_3, circx_3] = test_dict_conc_vary_ksw[8]
        conc_vary_ksw_slice[circy_4, circx_4] = test_dict_conc_vary_ksw[9]
        conc_vary_ksw_slice[circy_5, circx_5] = test_dict_conc_vary_ksw[10]
        conc_vary_ksw_slice[circy_6, circx_6] = test_dict_conc_vary_ksw[0]
        
        ksw_vary_conc_slice[circy_1, circx_1] = test_dict_ksw_vary_conc[6]
        ksw_vary_conc_slice[circy_2, circx_2] = test_dict_ksw_vary_conc[7]
        ksw_vary_conc_slice[circy_3, circx_3] = test_dict_ksw_vary_conc[8]
        ksw_vary_conc_slice[circy_4, circx_4] = test_dict_ksw_vary_conc[9]
        ksw_vary_conc_slice[circy_5, circx_5] = test_dict_ksw_vary_conc[10]
        ksw_vary_conc_slice[circy_6, circx_6] = test_dict_ksw_vary_conc[11]
        
        ksw_vary_ksw_slice[circy_1, circx_1] = test_dict_ksw_vary_ksw[6]
        ksw_vary_ksw_slice[circy_2, circx_2] = test_dict_ksw_vary_ksw[7]
        ksw_vary_ksw_slice[circy_3, circx_3] = test_dict_ksw_vary_ksw[8]
        ksw_vary_ksw_slice[circy_4, circx_4] = test_dict_ksw_vary_ksw[9]
        ksw_vary_ksw_slice[circy_5, circx_5] = test_dict_ksw_vary_ksw[10]
        ksw_vary_ksw_slice[circy_6, circx_6] = test_dict_ksw_vary_ksw[11]
        
        conc_vary_sig_slice[circy_1, circx_1] = test_dict_conc_vary_sig[6,:,:]
        conc_vary_sig_slice[circy_2, circx_2] = test_dict_conc_vary_sig[7,:,:]
        conc_vary_sig_slice[circy_3, circx_3] = test_dict_conc_vary_sig[8,:,:]
        conc_vary_sig_slice[circy_4, circx_4] = test_dict_conc_vary_sig[9,:,:]
        conc_vary_sig_slice[circy_5, circx_5] = test_dict_conc_vary_sig[10,:,:]
        conc_vary_sig_slice[circy_6, circx_6] = test_dict_conc_vary_sig[0,:,:]
        
        ksw_vary_sig_slice[circy_1, circx_1] = test_dict_ksw_vary_sig[6,:,:]
        ksw_vary_sig_slice[circy_2, circx_2] = test_dict_ksw_vary_sig[7,:,:]
        ksw_vary_sig_slice[circy_3, circx_3] = test_dict_ksw_vary_sig[8,:,:]
        ksw_vary_sig_slice[circy_4, circx_4] = test_dict_ksw_vary_sig[9,:,:]
        ksw_vary_sig_slice[circy_5, circx_5] = test_dict_ksw_vary_sig[10,:,:]
        ksw_vary_sig_slice[circy_6, circx_6] = test_dict_ksw_vary_sig[11,:,:]
        
    elif index == 2: 
        conc_vary_conc_slice[circy_1, circx_1] = test_dict_conc_vary_conc[0]
        conc_vary_conc_slice[circy_2, circx_2] = test_dict_conc_vary_conc[0]
        conc_vary_conc_slice[circy_3, circx_3] = test_dict_conc_vary_conc[0]
        conc_vary_conc_slice[circy_4, circx_4] = test_dict_conc_vary_conc[0]
        conc_vary_conc_slice[circy_5, circx_5] = test_dict_conc_vary_conc[0]
        conc_vary_conc_slice[circy_6, circx_6] = test_dict_conc_vary_conc[0]
        
        conc_vary_ksw_slice[circy_1, circx_1] = test_dict_conc_vary_ksw[0]
        conc_vary_ksw_slice[circy_2, circx_2] = test_dict_conc_vary_ksw[0]
        conc_vary_ksw_slice[circy_3, circx_3] = test_dict_conc_vary_ksw[0]
        conc_vary_ksw_slice[circy_4, circx_4] = test_dict_conc_vary_ksw[0]
        conc_vary_ksw_slice[circy_5, circx_5] = test_dict_conc_vary_ksw[0]
        conc_vary_ksw_slice[circy_6, circx_6] = test_dict_conc_vary_ksw[0]
        
        ksw_vary_conc_slice[circy_1, circx_1] = test_dict_ksw_vary_conc[12]
        ksw_vary_conc_slice[circy_2, circx_2] = test_dict_ksw_vary_conc[0]
        ksw_vary_conc_slice[circy_3, circx_3] = test_dict_ksw_vary_conc[0]
        ksw_vary_conc_slice[circy_4, circx_4] = test_dict_ksw_vary_conc[0]
        ksw_vary_conc_slice[circy_5, circx_5] = test_dict_ksw_vary_conc[0]
        ksw_vary_conc_slice[circy_6, circx_6] = test_dict_ksw_vary_conc[0]
        
        ksw_vary_ksw_slice[circy_1, circx_1] = test_dict_ksw_vary_ksw[12]
        ksw_vary_ksw_slice[circy_2, circx_2] = test_dict_ksw_vary_ksw[0]
        ksw_vary_ksw_slice[circy_3, circx_3] = test_dict_ksw_vary_ksw[0]
        ksw_vary_ksw_slice[circy_4, circx_4] = test_dict_ksw_vary_ksw[0]
        ksw_vary_ksw_slice[circy_5, circx_5] = test_dict_ksw_vary_ksw[0]
        ksw_vary_ksw_slice[circy_6, circx_6] = test_dict_ksw_vary_ksw[0]
        
        conc_vary_sig_slice[circy_1, circx_1] = test_dict_conc_vary_sig[0,:,:]
        conc_vary_sig_slice[circy_2, circx_2] = test_dict_conc_vary_sig[0,:,:]
        conc_vary_sig_slice[circy_3, circx_3] = test_dict_conc_vary_sig[0,:,:]
        conc_vary_sig_slice[circy_4, circx_4] = test_dict_conc_vary_sig[0,:,:]
        conc_vary_sig_slice[circy_5, circx_5] = test_dict_conc_vary_sig[0,:,:]
        conc_vary_sig_slice[circy_6, circx_6] = test_dict_conc_vary_sig[0,:,:]
        
        ksw_vary_sig_slice[circy_1, circx_1] = test_dict_ksw_vary_sig[12,:,:]
        ksw_vary_sig_slice[circy_2, circx_2] = test_dict_ksw_vary_sig[0,:,:]
        ksw_vary_sig_slice[circy_3, circx_3] = test_dict_ksw_vary_sig[0,:,:]
        ksw_vary_sig_slice[circy_4, circx_4] = test_dict_ksw_vary_sig[0,:,:]
        ksw_vary_sig_slice[circy_5, circx_5] = test_dict_ksw_vary_sig[0,:,:]
        ksw_vary_sig_slice[circy_6, circx_6] = test_dict_ksw_vary_sig[0,:,:]
    
    conc_vary_conc_image.append(conc_vary_conc_slice)
    conc_vary_ksw_image.append(conc_vary_ksw_slice)
    ksw_vary_conc_image.append(ksw_vary_conc_slice)
    ksw_vary_ksw_image.append(ksw_vary_ksw_slice)
    conc_vary_sig_image.append(conc_vary_sig_slice)
    ksw_vary_sig_image.append(ksw_vary_sig_slice)
    
    vial_1_mask[circy_1, circx_1] = 0
    vial_2_mask[circy_2, circx_2] = 0
    vial_3_mask[circy_3, circx_3] = 0
    vial_4_mask[circy_4, circx_4] = 0
    vial_5_mask[circy_5, circx_5] = 0
    vial_6_mask[circy_6, circx_6] = 0
    
    mask_list = [vial_1_mask, vial_2_mask, vial_3_mask, vial_4_mask, vial_5_mask, vial_6_mask]
    
    masks.append(mask_list)
    

conc_vary_conc_image = np.array(conc_vary_conc_image)
conc_vary_ksw_image = np.array(conc_vary_ksw_image)
ksw_vary_conc_image = np.array(ksw_vary_conc_image)
ksw_vary_ksw_image = np.array(ksw_vary_ksw_image)
conc_vary_sig_image = np.array(conc_vary_sig_image)
ksw_vary_sig_image = np.array(ksw_vary_sig_image)

masks = np.array(masks)

masks = np.reshape(masks, (18, 128, 128))

conc_vary_sig_image_30 = conc_vary_sig_image
conc_vary_sig_image_9 = np.delete(conc_vary_sig_image, range(9, 30), axis=3)
ksw_vary_sig_image_30 = ksw_vary_sig_image
ksw_vary_sig_image_9 = np.delete(ksw_vary_sig_image, range(9, 30), axis=3)

conc_vary_sig_image_30 = np.swapaxes(conc_vary_sig_image_30, 0, 3)
conc_vary_sig_image_9 = np.swapaxes(conc_vary_sig_image_9, 0, 3)
ksw_vary_sig_image_30 = np.swapaxes(ksw_vary_sig_image_30, 0, 3)
ksw_vary_sig_image_9 = np.swapaxes(ksw_vary_sig_image_9, 0, 3)

conc_vary_sig_image_30 = conc_vary_sig_image_30 / np.sqrt(np.sum(conc_vary_sig_image_30 ** 2, axis=0))
conc_vary_sig_image_30[np.isnan(conc_vary_sig_image_30)] = 0 
conc_vary_sig_image_9 = conc_vary_sig_image_9 / np.sqrt(np.sum(conc_vary_sig_image_9 ** 2, axis=0))
conc_vary_sig_image_9[np.isnan(conc_vary_sig_image_9)] = 0 
ksw_vary_sig_image_30 = ksw_vary_sig_image_30 / np.sqrt(np.sum(ksw_vary_sig_image_30 ** 2, axis=0))
ksw_vary_sig_image_30[np.isnan(ksw_vary_sig_image_30)] = 0 
ksw_vary_sig_image_9 = ksw_vary_sig_image_9 / np.sqrt(np.sum(ksw_vary_sig_image_9 ** 2, axis=0))
ksw_vary_sig_image_9[np.isnan(ksw_vary_sig_image_9)] = 0 

conc_vary_sig_image_30 = np.swapaxes(conc_vary_sig_image_30, 0, 3)
conc_vary_sig_image_9 = np.swapaxes(conc_vary_sig_image_9, 0, 3)
ksw_vary_sig_image_30 = np.swapaxes(ksw_vary_sig_image_30, 0, 3)
ksw_vary_sig_image_9 = np.swapaxes(ksw_vary_sig_image_9, 0, 3)


conc_vary_9 = []
conc_vary_30 = []
ksw_vary_9 = []
ksw_vary_30 = []

for i in range(np.size(conc_vary_sig_image, axis=4)):
    conc_vary_predict_9 = model_9.predict(conc_vary_sig_image_9[:,:,:,:,i])
    conc_vary_9.append(conc_vary_predict_9)
    conc_vary_predict_30 = model_30.predict(conc_vary_sig_image_30[:,:,:,:,i])
    conc_vary_30.append(conc_vary_predict_30)
    ksw_vary_predict_9 = model_9.predict(ksw_vary_sig_image_9[:,:,:,:,i])
    ksw_vary_9.append(ksw_vary_predict_9)
    ksw_vary_predict_30 = model_30.predict(ksw_vary_sig_image_30[:,:,:,:,i])
    ksw_vary_30.append(ksw_vary_predict_30)

conc_vary_9 = np.array(conc_vary_9)    
conc_vary_30 = np.array(conc_vary_30)   
ksw_vary_9 = np.array(ksw_vary_9) 
ksw_vary_30 = np.array(ksw_vary_30)  

conc_vary_conc_predict_9 = conc_vary_9[:,:,:,:,0]
conc_vary_ksw_predict_9 = conc_vary_9[:,:,:,:,1]
ksw_vary_conc_predict_9 = ksw_vary_9[:,:,:,:,0]
ksw_vary_ksw_predict_9 = ksw_vary_9[:,:,:,:,1]
conc_vary_conc_predict_30 = conc_vary_30[:,:,:,:,0]
conc_vary_ksw_predict_30 = conc_vary_30[:,:,:,:,1]
ksw_vary_conc_predict_30 = ksw_vary_30[:,:,:,:,0]
ksw_vary_ksw_predict_30 = ksw_vary_30[:,:,:,:,1]

def un_normalize_range_y(normalized_array, original_min, original_max, new_min, new_max):
    a = original_min
    b = original_max
    c = new_min
    d = new_max
    return (normalized_array - c) / (d - c) * (b - a) + a

conc_vary_conc_predict_9 = un_normalize_range_y(conc_vary_conc_predict_9, original_min = 0, original_max = 130, new_min = 0, new_max = 1)
conc_vary_ksw_predict_9 = un_normalize_range_y(conc_vary_ksw_predict_9, original_min = 0, original_max = 1500, new_min = 0, new_max = 1)
ksw_vary_conc_predict_9 = un_normalize_range_y(ksw_vary_conc_predict_9, original_min = 0, original_max = 130, new_min = 0, new_max = 1)
ksw_vary_ksw_predict_9 = un_normalize_range_y(ksw_vary_ksw_predict_9, original_min = 0, original_max = 1500, new_min = 0, new_max = 1)
conc_vary_conc_predict_30 = un_normalize_range_y(conc_vary_conc_predict_30, original_min = 0, original_max = 130, new_min = 0, new_max = 1)
conc_vary_ksw_predict_30 = un_normalize_range_y(conc_vary_ksw_predict_30, original_min = 0, original_max = 1500, new_min = 0, new_max = 1)
ksw_vary_conc_predict_30 = un_normalize_range_y(ksw_vary_conc_predict_30, original_min = 0, original_max = 130, new_min = 0, new_max = 1)
ksw_vary_ksw_predict_30 = un_normalize_range_y(ksw_vary_ksw_predict_30, original_min = 0, original_max = 1500, new_min = 0, new_max = 1)

conc_masked_real = []
conc_masked_predict_9 = []
conc_masked_predict_30 = []
ksw_masked_real = []
ksw_masked_predict_9 = []
ksw_masked_predict_30 = []

for i in range(0,13,1):
    vial = i
    cpv_9 = []
    cpv_30 = []
    kpv_9 = []
    kpv_30 = []
    
    if vial < 6:
        conc_vial_real_mask = np.ma.masked_where(masks[vial, :, :]==1, conc_vary_conc_image[0,:,:]).compressed()
        ksw_vial_real_mask = np.ma.masked_where(masks[vial, :, :]==1, ksw_vary_ksw_image[0,:,:]).compressed()
        for i in range(0, 10, 1):
            conc_vial_predict_9_mask = np.ma.masked_where(masks[vial, :, :]==1, conc_vary_conc_predict_9[i,0,:,:]).compressed()
            conc_vial_predict_30_mask = np.ma.masked_where(masks[vial, :, :]==1, conc_vary_conc_predict_30[i,0,:,:]).compressed()
            ksw_vial_predict_9_mask = np.ma.masked_where(masks[vial, :, :]==1, ksw_vary_ksw_predict_9[i,0,:,:]).compressed()
            ksw_vial_predict_30_mask = np.ma.masked_where(masks[vial, :, :]==1, ksw_vary_ksw_predict_30[i,0,:,:]).compressed()
            cpv_9.append(conc_vial_predict_9_mask)
            cpv_30.append(conc_vial_predict_30_mask)
            kpv_9.append(ksw_vial_predict_9_mask)
            kpv_30.append(ksw_vial_predict_30_mask)
    elif 5 < vial < 12:
        conc_vial_real_mask = np.ma.masked_where(masks[vial, :, :]==1, conc_vary_conc_image[1,:,:]).compressed()
        ksw_vial_real_mask = np.ma.masked_where(masks[vial, :, :]==1, ksw_vary_ksw_image[1,:,:]).compressed()
        for i in range(0, 10, 1):
            conc_vial_predict_9_mask = np.ma.masked_where(masks[vial, :, :]==1, conc_vary_conc_predict_9[i,1,:,:]).compressed()
            conc_vial_predict_30_mask = np.ma.masked_where(masks[vial, :, :]==1, conc_vary_conc_predict_30[i,1,:,:]).compressed()
            ksw_vial_predict_9_mask = np.ma.masked_where(masks[vial, :, :]==1, ksw_vary_ksw_predict_9[i,1,:,:]).compressed()
            ksw_vial_predict_30_mask = np.ma.masked_where(masks[vial, :, :]==1, ksw_vary_ksw_predict_30[i,1,:,:]).compressed()
            cpv_9.append(conc_vial_predict_9_mask)
            cpv_30.append(conc_vial_predict_30_mask)
            kpv_9.append(ksw_vial_predict_9_mask)
            kpv_30.append(ksw_vial_predict_30_mask)
    elif vial > 11:
        conc_vial_real_mask = np.ma.masked_where(masks[vial, :, :]==1, conc_vary_conc_image[2,:,:]).compressed()
        ksw_vial_real_mask = np.ma.masked_where(masks[vial, :, :]==1, ksw_vary_ksw_image[2,:,:]).compressed()
        for i in range(0, 10, 1):
            conc_vial_predict_9_mask = np.ma.masked_where(masks[vial, :, :]==1, conc_vary_conc_predict_9[i,2,:,:]).compressed()
            conc_vial_predict_30_mask = np.ma.masked_where(masks[vial, :, :]==1, conc_vary_conc_predict_30[i,2,:,:]).compressed()
            ksw_vial_predict_9_mask = np.ma.masked_where(masks[vial, :, :]==1, ksw_vary_ksw_predict_9[i,2,:,:]).compressed()
            ksw_vial_predict_30_mask = np.ma.masked_where(masks[vial, :, :]==1, ksw_vary_ksw_predict_30[i,2,:,:]).compressed()
            cpv_9.append(conc_vial_predict_9_mask)
            cpv_30.append(conc_vial_predict_30_mask)
            kpv_9.append(ksw_vial_predict_9_mask)
            kpv_30.append(ksw_vial_predict_30_mask)
            
    conc_masked_real.append(conc_vial_real_mask)
    ksw_masked_real.append(ksw_vial_real_mask)
    conc_masked_predict_9.append(cpv_9)
    conc_masked_predict_30.append(cpv_30)
    ksw_masked_predict_9.append(kpv_9)
    ksw_masked_predict_30.append(kpv_30)

conc_masked_real = np.array(conc_masked_real)
conc_masked_predict_9 = np.reshape(np.swapaxes(np.array(conc_masked_predict_9),2,1), (13,450))
conc_masked_predict_30 = np.reshape(np.swapaxes(np.array(conc_masked_predict_30),2,1), (13,450))
ksw_masked_real = np.array(ksw_masked_real)
ksw_masked_predict_9 = np.reshape(np.swapaxes(np.array(ksw_masked_predict_9),2,1), (13,450))
ksw_masked_predict_30 = np.reshape(np.swapaxes(np.array(ksw_masked_predict_30),2,1), (13,450))
                                  
std_conc_9 = []
mean_conc_9=[]
std_conc_30 = []
mean_conc_30=[]
std_ksw_9 = []
mean_ksw_9=[]
std_ksw_30 = []
mean_ksw_30=[]

for i in range(0,13):
    vial_conc_9_std = np.std(conc_masked_predict_9[i,:])
    vial_conc_30_std = np.std(conc_masked_predict_30[i,:])
    vial_ksw_9_std = np.std(ksw_masked_predict_9[i,:])
    vial_ksw_30_std = np.std(ksw_masked_predict_30[i,:])
    vial_conc_9_mean = np.mean(conc_masked_predict_9[i,:])
    vial_conc_30_mean = np.mean(conc_masked_predict_30[i,:])
    vial_ksw_9_mean = np.mean(ksw_masked_predict_9[i,:])
    vial_ksw_30_mean = np.mean(ksw_masked_predict_30[i,:])
    
    std_conc_9.append(vial_conc_9_std)
    std_conc_30.append(vial_conc_30_std)
    std_ksw_9.append(vial_ksw_9_std)
    std_ksw_30.append(vial_ksw_30_std)
    mean_conc_9.append(vial_conc_9_mean)
    mean_conc_30.append(vial_conc_30_mean)
    mean_ksw_9.append(vial_ksw_9_mean)
    mean_ksw_30.append(vial_ksw_30_mean)
    
std_conc_9 = np.delete(np.array(std_conc_9), range(11,13), axis=0)
std_conc_30 = np.delete(np.array(std_conc_30), range(11,13), axis=0)
std_ksw_9 = np.array(std_ksw_9)
std_ksw_30 = np.array(std_ksw_30)
mean_conc_9 = np.delete(np.array(mean_conc_9), range(11,13), axis=0)
mean_conc_30 = np.delete(np.array(mean_conc_30), range(11,13), axis=0)
mean_ksw_9 = np.array(mean_ksw_9)
mean_ksw_30 = np.array(mean_ksw_30)

plt.figure()
plt.title('[L-arg] (30 inputs)')
plt.errorbar(test_dict_conc_vary_conc, mean_conc_30, yerr = std_conc_30, fmt = 'none', capsize = 3, linewidth = 1)
plt.xlabel('Dictionary [L-arg] (mM)')
plt.ylabel('Predicted [L-arg] (mM)')
plt.axline((0, 0), slope=1, color='black', linewidth = 0.75)
plt.savefig('monte_conc_30.pdf')
plt.figure()
plt.title('[L-arg] (9 inputs)')
plt.errorbar(test_dict_conc_vary_conc, mean_conc_9, yerr = std_conc_9, fmt = 'none', capsize = 3, linewidth = 1)
plt.xlabel('Dictionary [L-arg] (mM)')
plt.ylabel('Predicted [L-arg] (mM)')
plt.axline((0, 0), slope=1, color='black', linewidth = 0.75)
plt.savefig('monte_conc_9.pdf')
plt.figure()
plt.title('k$_{sw}$ (30 inputs)')
plt.errorbar(test_dict_ksw_vary_ksw, mean_ksw_30, yerr = std_ksw_30, fmt = 'none', capsize = 3, linewidth = 1)
plt.xlabel('Dictionary k$_{sw}$ (s$^{-1}$)')
plt.ylabel('Predicted k$_{sw}$ (s$^{-1}$)')
plt.axline((0, 0), slope=1, color='black', linewidth = 0.75)
plt.savefig('monte_ksw_30.pdf')
plt.figure()
plt.title('k$_{sw}$ (9 inputs)')
plt.errorbar(test_dict_ksw_vary_ksw, mean_ksw_9, yerr = std_ksw_9, fmt = 'none', capsize = 3, linewidth = 1)
plt.xlabel('Dictionary k$_{sw}$ (s$^{-1}$)')
plt.ylabel('Predicted k$_{sw}$ (s$^{-1}$)')
plt.axline((0, 0), slope=1, color='black', linewidth = 0.75)
plt.savefig('monte_ksw_9.pdf')

test_dict_conc_vary_conc = test_dict_conc_vary_conc.flatten()
test_dict_ksw_vary_ksw = test_dict_ksw_vary_ksw.flatten()

test_dict_conc_vary_conc = np.repeat(test_dict_conc_vary_conc, 450)
test_dict_ksw_vary_ksw = np.repeat(test_dict_ksw_vary_ksw, 450)
list_conc_9 = np.reshape(conc_masked_predict_9, 5850)
list_conc_30 = np.reshape(conc_masked_predict_30, 5850)
list_conc_9 = np.delete(list_conc_9, range(4950, 5850), axis = 0)
list_conc_30 = np.delete(list_conc_30, range(4950, 5850), axis = 0)
list_ksw_9 = np.reshape(ksw_masked_predict_9, 5850)
list_ksw_30 = np.reshape(ksw_masked_predict_30, 5850)

r_conc_9, p_conc_9 = pearsonr(test_dict_conc_vary_conc, list_conc_9)
r_conc_30, p_conc_30 = pearsonr(test_dict_conc_vary_conc, list_conc_30)
r_ksw_9, p_ksw_9 = pearsonr(test_dict_ksw_vary_ksw, list_ksw_9)
r_ksw_30, p_ksw_30 = pearsonr(test_dict_ksw_vary_ksw, list_ksw_30)

print('---9-inputs---\n[L-arg]: r =', r_conc_9, ', p =', p_conc_9, '\nksw: r =', r_ksw_9, ', p =', p_ksw_9)
print('---30-inputs---\n[L-arg]: r =', r_conc_30, ', p =', p_conc_30, '\nksw: r =', r_ksw_30, ', p =', p_ksw_30)
    
conc_9 = np.concatenate((test_dict_conc_vary_conc, list_conc_9))    
conc_30 = np.concatenate((test_dict_conc_vary_conc, list_conc_30))    
ksw_9 = np.concatenate((test_dict_ksw_vary_ksw, list_ksw_9))
ksw_30 = np.concatenate((test_dict_ksw_vary_ksw, list_ksw_30))

pixels_conc = [*range(len(list_conc_9))]
pixels_conc = np.asarray(pixels_conc)
pixels_conc = np.concatenate((pixels_conc, pixels_conc))
pixels_ksw = [*range(len(list_ksw_9))]
pixels_ksw = np.asarray(pixels_ksw)
pixels_ksw = np.concatenate((pixels_ksw, pixels_ksw))

labels_conc = np.concatenate((np.repeat('r', len(conc_9)/2), np.repeat('p', len(conc_9)/2)))
labels_ksw = np.concatenate((np.repeat('r', len(ksw_9)/2), np.repeat('p', len(ksw_9)/2)))

data_conc_9 = np.stack((pixels_conc, labels_conc, conc_9), axis = 1)
data_conc_30 = np.stack((pixels_conc, labels_conc, conc_30), axis = 1)
data_ksw_9 = np.stack((pixels_ksw, labels_ksw, ksw_9), axis = 1)
data_ksw_30 = np.stack((pixels_ksw, labels_ksw, ksw_30), axis = 1)

data_conc_9 = pd.DataFrame(data_conc_9)
data_conc_30 = pd.DataFrame(data_conc_30)
data_ksw_9 = pd.DataFrame(data_ksw_9)
data_ksw_30 = pd.DataFrame(data_ksw_30)

data_conc_9.columns = ['Pixels', 'Labels', 'Values']
data_conc_30.columns = ['Pixels', 'Labels', 'Values']
data_ksw_9.columns = ['Pixels', 'Labels', 'Values']
data_ksw_30.columns = ['Pixels', 'Labels', 'Values']

icc_conc_9 = pg.intraclass_corr(data=data_conc_9, targets='Pixels', raters='Labels', ratings='Values')
print('---ICC [L-arg] 9---')
print(icc_conc_9)
icc_conc_30 = pg.intraclass_corr(data=data_conc_30, targets='Pixels', raters='Labels', ratings='Values')
print('---ICC [L-arg] 30---')
print(icc_conc_30)
icc_ksw_9 = pg.intraclass_corr(data=data_ksw_9, targets='Pixels', raters='Labels', ratings='Values')
print('---ICC ksw 9---')
print(icc_ksw_9)
icc_ksw_30 = pg.intraclass_corr(data=data_ksw_30, targets='Pixels', raters='Labels', ratings='Values')
print('---ICC ksw 30---')
print(icc_ksw_30)






