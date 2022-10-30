# molecular-MRI-GAN
The purpose of this code is to train a GAN for accelerating CEST and MT quantitative parameter mapping. 

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
* A pre-trained VGG network is required for implementing the perceptual loss, it can be downloaded [here](https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat). Place the file in the top level of this repository. 

* Image files for training
  * A single L-arginine phantom slice is included as a demonstration for both training and inference. The network expects sets of 9 128x128 L2 normalized MRF image per slice as input, and sets of 2 128x128 linearly scaled CEST maps (concentration and chemical exchange rate) as output. 
  
* Trained netowrks can be found at: https://figshare.com/s/c91bf3f02e91f91edaf9

## Citations 
1. Simonyan Karen, Zisserman Andrew. Very deep convolutional networks for large-scale image recognition arXiv preprint arXiv:1409.1556. 2014.

2. Isola Phillip, Zhu Jun-Yan, Zhou Tinghui, Efros Alexei A. Image-to-image translation with conditional adversarial networks in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition:1125–1134 2017.

3. Jason Brownlee, How to Implement Pix2Pix GAN Models From Scratch With Keras, Machine Learning Mastery, Available from https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/, Accessed October 14, 2021.

4. Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." European conference on computer vision. Springer, Cham, 2016.

5. Cheng-Bin Jin, Real-Time Style Transfer, 2018, https://github.com/ChengBinJin/Real-time-style-transfer/

## This repository is associated with the paper
[Accelerated and Quantitative 3D Semisolid MT/CEST Imaging using a Generative Adversarial Network](https://arxiv.org/abs/2207.11297)
