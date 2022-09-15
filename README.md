# molecular-MRI-GAN

## Requirements

### Dependencies 
* NumPy
* SciPy.io
* Functools
* TensorFlow
* Keras
* Matplotlib

  #### Suggested installation routes
  * [pip](https://pip.pypa.io/en/stable/)
  * [Anaconda](https://www.anaconda.com/products/distribution)
    * A YAML file for creating the relevant environment on ARM64 based M1 Macs is available in the 'Conda' folder


### Data Files
* A pre-trained VGG network is required, it can be downloaded [here](https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat). Place the file in the top level of this repository. 

* Image files for training
  * A single L-arginine phantom slice is included as a demonstration for both training and inference. The network expects sets of 9 128x128 L2 normalized MRF image per slice as input, and sets of 2 128x128 linearly scaled CEST maps (concentration and chemical exchange rate) as output. 

## Citations 
Jason Brownlee, How to Implement Pix2Pix GAN Models From Scratch With Keras, Machine Learning Mastery, Available from https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/, Accessed October 14, 2021.
