# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:01:04 2021

@author: Bonewheel
"""

import os
import time
import numpy as np
from GAN_Data import x_train, y_train
from keras.models import load_model
import matplotlib.pyplot as plt
#from cuda import numba

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

###Load model and predict###
model = load_model('mmri-gan_model.h5')

start_time = time.time()
predicted_output = model.predict(x_train)
print("--- %s seconds ---" % (time.time() - start_time))
#Example phantom (without augmentation) is loaded here for demonstration purposes
#because the network was trained on this phantom, results will not be represenative.

###Reshape and normalize###
def un_normalize_range_y(normalized_array, original_min, original_max, new_min, new_max):
    a = original_min
    b = original_max
    c = new_min
    d = new_max
    return (normalized_array - c) / (d - c) * (b - a) + a

y_predict = np.reshape(predicted_output, (1, 128, 128, 2))
y_real = np.reshape(y_train, (1, 128, 128, 2))

y_predict_mmol = y_predict[:, :, :, 0]
y_predict_hz = y_predict[:, :, :, 1]
y_real_mmol = y_real[:, :, :, 0]
y_real_hz = y_real[:, :, :, 1]

y_real_mmol = un_normalize_range_y(y_real_mmol, original_min=10, original_max=120, new_min=0, new_max=1)
y_real_hz = un_normalize_range_y(y_real_hz, original_min=100, original_max=1400, new_min=0, new_max=1)

y_predict_mmol = un_normalize_range_y(y_predict_mmol, original_min=10, original_max=120, new_min=0, new_max=1)
y_predict_hz = un_normalize_range_y(y_predict_hz, original_min=100, original_max=1400, new_min=0, new_max=1)

y_real = np.stack((y_real_mmol, y_real_hz), axis = 3)
y_predict = np.stack((y_predict_mmol, y_predict_hz), axis = 3)

for slice_ind in range(np.size(y_predict, 0)):
     fig, axs = plt.subplots(2, 2, constrained_layout=False)
     fig.suptitle('Slice %i' %slice_ind)
     axs[0, 0].set_title('[L-arg] (mM)')
     axs[0, 1].set_title('k$_{sw}$ (s$^{-1}$)')
     axs[1, 0].set_ylabel('MRF-Based')
     axs[0, 0].set_ylabel('GAN-Based')

     mm = axs[1, 0].imshow(y_real[slice_ind, :, :, 0], cmap='viridis', clim=(0, 120))
     hz = axs[1, 1].imshow(y_real[slice_ind, :, :, 1], cmap='magma', clim=(0, 1400))
     axs[0, 0].imshow(y_predict[slice_ind, :, :, 0], cmap='viridis', clim=(0, 120))
     axs[0, 1].imshow(y_predict[slice_ind, :, :, 1], cmap='magma', clim=(0, 1400))
     plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

     fig.colorbar(mm, ax=axs[(0, 1), 0], orientation='vertical')
     fig.colorbar(hz, ax=axs[(0, 1), 1], orientation='vertical')

     plt.show()

print("--- %s seconds ---" % (time.time() - start_time))