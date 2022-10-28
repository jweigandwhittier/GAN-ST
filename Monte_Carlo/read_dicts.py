import numpy as np
import scipy.io as sio

# Train set
train_dict = sio.loadmat('Dicts/train_dict_sep.mat')
train_conc = train_dict['conc']
train_ksw = train_dict['ksw']
train_sig = train_dict['sig']
# print(train_sig.shape)

# Test set where ksw varies
test_dict_ksw_vary = sio.loadmat('Dicts/test_dict_ksw_vary_sep.mat')
test_dict_ksw_vary_conc = test_dict_ksw_vary['conc']
test_dict_ksw_vary_ksw = test_dict_ksw_vary['ksw']
test_dict_ksw_vary_sig = test_dict_ksw_vary['sig']
# print(test_dict_ksw_vary_sig.shape)

# Test set where conc is varied
test_dict_conc_vary = sio.loadmat('Dicts/test_dict_conc_vary_sep.mat')
test_dict_conc_vary_conc = test_dict_conc_vary['conc']
test_dict_conc_vary_ksw = test_dict_conc_vary['ksw']
test_dict_conc_vary_sig = test_dict_conc_vary['sig']
# print(test_dict_conc_vary_sig.shape)

