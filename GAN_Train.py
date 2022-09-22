# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:29:22 2022

"""

# GAN code adapted from: Jason Brownlee, How to Implement Pix2Pix GAN Models From Scratch With Keras, Machine Learning Mastery
# Available from https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/
# Accessed October 14, 2021

#Perceptual loss function adapted by Or Perlman

###Import packages###
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import functools
import tensorflow as tf
from operator import mul
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.utils.vis_utils import plot_model
from tensorflow.python.client import device_lib
import os
# from numba import cuda # May be necessary for training on Windows computers

###Import augmented datasets from GAN_Data.py###
from GAN_Data import x_train_aug, y_train_aug

###Set GPUs as visible###
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


###Or's perceptual loss function###
class VGG19(object):
    data_path = 'imagenet-vgg-verydeep-19.mat/'

    def __init__(self, data_path):
        self.data = scipy.io.loadmat(data_path)
        self.weights = self.data['layers'][0]
        self.mean_pixel = np.asarray([123.68, 116.779, 103.939])
        self.layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
        )

    def __call__(self, img, name='vgg', is_reuse=False):
        with tf.compat.v1.variable_scope(name, reuse=is_reuse):
            img_pre = self.preprocess(img)

            net_dic = {}
            current = img_pre

            for i, name in enumerate(self.layers):
                kind = name[:4]

                if kind == 'conv':
                    kernels, bias = self.weights[i][0][0][0][0]

                    kernels = np.transpose(kernels, (1, 0, 2, 3))
                    bias = bias.reshape(-1)
                    current = self._conv_layer(current, kernels, bias)

                elif kind == 'relu':
                    current = tf.nn.relu(current)

                elif kind == 'pool':
                    current = self._pool_layer(current)

                net_dic[name] = current

            assert len(net_dic) == len(self.layers)

        return net_dic

    @staticmethod
    def _conv_layer(input_, weights, bias):
        conv = tf.nn.conv2d(input_, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
        return tf.nn.bias_add(conv, bias)

    @staticmethod
    def _pool_layer(input_):
        return tf.nn.max_pool(input_, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

    def preprocess(self, img):
        return img - self.mean_pixel

    def unprocess(self, img):
        return img + self.mean_pixel


class Perceptual(object):
    def __init__(self, y_true, y_pred, batch_size):

        self.vgg_path = 'imagenet-vgg-verydeep-19.mat'

        #Not too bad, original backup
        # self.content_weight = 7.5  # used in style transfer. Consider playing with it
        # self.tv_weight = 200  # used in style transfer. Consider playing with it
        # self.l1_weight = 100  # OP added it. Consider making much smaller or increasing content weight

        self.content_weight = 0.1  # Used in style transfer. Consider playing with it
        self.tv_weight = 1.0  # Used in style transfer. Consider playing with it
        self.l1_weight = 1.0  # OP added it. Consider making much smaller or increasing content weight

        self.batch_size = batch_size  # Make sure it matches Jonah's code part batch size!!!

        self.content_shape = [None, 256, 256, 3]

        self.content_layer = 'relu2_2'  # original paper
        self.content_loss, self.tv_loss, self.l1_loss = None, None, None
        self.preds = y_pred
        self.content_img_ph = y_true
        self._build_net()
        self._tensorboard()

    def _build_net(self):
        # ph: tensorflow placeholder
        # self.content_img_ph = tf.placeholder(tf.float32, shape=self.content_shape, name='content_img')

        self.vgg = VGG19(self.vgg_path)

        # # step 1: extract style_target feature
        # vgg_dic = self.vgg(self.style_img_ph)
        # for layer in self.style_layers:
        #     features = self.sess.run(vgg_dic[layer], feed_dict={self.style_img_ph: self.style_target})
        #     features = np.reshape(features, (-1, features.shape[3]))
        #     gram = np.matmul(features.T, features) / features.size
        #     self.style_target_gram[layer] = gram

        # step 2: extract content_target feature
        content_target_feature = {}
        vgg_content_dic = self.vgg(self.content_img_ph, is_reuse=True)
        content_target_feature[self.content_layer] = vgg_content_dic[self.content_layer]

        # step 3: tranfer content image to predicted image
        # self.preds = self.transfer(self.content_img_ph / 255.0)

        # step 4: extract vgg feature of the predicted image
        preds_dict = self.vgg(self.preds, is_reuse=True)
        # self.sample_pred = self.transfer(self.sample_img_ph/255.0, is_reuse=True)

        self.content_loss_func(preds_dict, content_target_feature)
        self.tv_loss_func(self.preds)
        self.l1_loss_func(self.preds, self.content_img_ph)
        self.total_loss = self.content_loss + self.tv_loss + self.l1_loss

    def _tensorboard(self):
        tf.summary.scalar('loss/content_loss', self.content_loss)
        tf.summary.scalar('loss/tv_loss', self.tv_loss)
        tf.summary.scalar('loss/l1_loss_func', self.l1_loss)
        tf.summary.scalar('loss/total_loss', self.total_loss)

        self.summary_op = tf.compat.v1.summary.merge_all()

    def l1_loss_func(self, preds, content_img_ph):
        mae = tf.keras.losses.MeanAbsoluteError()  # mean absolute error loss (mae)
        self.l1_loss = self.l1_weight * mae(preds, content_img_ph)

    def content_loss_func(self, preds_dict, content_target_feature):
        # Calculate content size and check the feature dimension between content and predicted image
        content_size = self._tensor_size(content_target_feature[self.content_layer]) * self.batch_size

        assert self._tensor_size(content_target_feature[self.content_layer]) == self._tensor_size(
            preds_dict[self.content_layer])

        self.content_loss = self.content_weight * (2 * tf.nn.l2_loss(
            preds_dict[self.content_layer] - content_target_feature[self.content_layer]) / content_size)

    def tv_loss_func(self, preds):
        # Total variation de-noising
        tv_y_size = self._tensor_size(preds[:, 1:, :, :])
        tv_x_size = self._tensor_size(preds[:, :, 1:, :])

        y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :self.content_shape[1] - 1, :, :])
        x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :self.content_shape[2] - 1, :])
        self.tv_loss = self.tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / self.batch_size

    def calc_loss_step(self):
        all_losses = [self.content_loss, self.tv_loss, self.l1_loss, self.total_loss]
        # feed_dict = {self.content_img_ph: imgs}
        # content_loss, tv_loss, l1_loss, total_loss, summary = self.sess.run(all_losses, feed_dict=feed_dict)

        return all_losses  # [content_loss, tv_loss, l1_loss, total_loss]

    @staticmethod
    def _tensor_size(tensor):
        return functools.reduce(mul, (tensor.get_shape()[1:]), 1)


# sess = tf.compat.v1.Session()
# percept = Perceptual(sess)

# Creating custom losses
# >>>
# full_percep_loss = percept.calc_loss_step(np.random.rand(2, 256, 256, 3))

def percept_loss_func(y_true, y_pred):
    # Separating the loss for kssw and fss
    y_true0 = tf.expand_dims(y_true[:, :, :, 0], axis=3)
    y_true1 = tf.expand_dims(y_true[:, :, :, 1], axis=3)

    y_pred0 = tf.expand_dims(y_pred[:, :, :, 0], axis=3)
    y_pred1 = tf.expand_dims(y_pred[:, :, :, 1], axis=3)

    print('Size of y_pred01 and y_true01 at the beginning of percept_loss_func: ')
    print(y_true0.shape)
    print(y_true1.shape)
    print(y_pred0.shape)
    print(y_pred1.shape)

    # Reformat y labels in the VGG19 format - 3 channels of 256 x 256 images
    y_true1 = tf.concat([y_true1, y_true1, y_true1], 3)  # 3 index chose since tensor was [none, 128, 128, 1]
    y_pred1 = tf.concat([y_pred1, y_pred1, y_pred1], 3)
    y_true1 = tf.image.resize(y_true1, [256, 256])
    y_pred1 = tf.image.resize(y_pred1, [256, 256])

    y_true0 = tf.concat([y_true0, y_true0, y_true0], 3)  # 3 index chose since tensor was [none, 128, 128, 1]
    y_pred0 = tf.concat([y_pred0, y_pred0, y_pred0], 3)
    y_true0 = tf.image.resize(y_true0, [256, 256])
    y_pred0 = tf.image.resize(y_pred0, [256, 256])

    percept1 = Perceptual(y_true1, y_pred1, Batch_Size)
    _, _, _, full_percep_loss1 = percept1.calc_loss_step()

    percept0 = Perceptual(y_true0, y_pred0, Batch_Size)
    _, _, _, full_percep_loss0 = percept0.calc_loss_step()

    full_percep_loss = 0.5 * (full_percep_loss0 + full_percep_loss1)

    return full_percep_loss


# Perceptual + TV loss + l1 loss implementation
# <<< # <<< # <<< # <<< # <<< # <<< # <<<

###Clip/normalize input data###
dataset = x_train_aug, y_train_aug

number_filters = 64 # Base number of CNN filters used in U-Net generator
N_Epochs = 400 # Number of training epochs, can be modified to speed up training (400 used in manuscript)
Batch_Size = 1 # Batch size, larger numbers can cause issues on machines with lower amounts of memory (VRAM + system RAM)

###Define PatchGAN discriminator###
def define_discriminator(input_shape, output_shape):
    init = RandomNormal(stddev=0.02)
    in_src_image = Input(shape=input_shape)
    in_target_image = Input(shape=output_shape)
    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(16, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)

    patch_out = Activation('sigmoid')(d)
    model = Model([in_src_image, in_target_image], patch_out)
    opt = Adam(lr=0.0005, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


###Define encoder block###
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (4, 4), strides=(1, 1), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    g = MaxPooling2D(pool_size=(2, 2))(g)
    return g


###Define decoder block###
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = Activation('relu')(g)
    return g


###Define U-Net generator###
def define_generator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    e1 = define_encoder_block(in_image, number_filters, batchnorm=False)
    e2 = define_encoder_block(e1, number_filters * 2)
    e3 = define_encoder_block(e2, number_filters * 4)
    e4 = define_encoder_block(e3, number_filters * 8)
    e5 = define_encoder_block(e4, number_filters * 8)
    e6 = define_encoder_block(e5, number_filters * 8)
    b = Conv2D(number_filters * 16, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e6)
    b = Activation('relu')(b)
    d1 = decoder_block(b, e6, number_filters * 8)
    d2 = decoder_block(d1, e5, number_filters * 8)
    d3 = decoder_block(d2, e4, number_filters * 8)
    d4 = decoder_block(d3, e3, number_filters * 8)
    d5 = decoder_block(d4, e2, number_filters * 4, dropout=False)
    d6 = decoder_block(d5, e1, number_filters * 2, dropout=False)
    g = Conv2DTranspose(2, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d6)
    out_image = Activation('tanh')(g)
    model = Model(in_image, out_image)
    return model

###Define GAN###
def define_gan(g_model, d_model, image_shape):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    model = Model(in_src, [dis_out, gen_out])
    opt = Adam(lr=0.0001, beta_1=0.5)
    #model.compile(loss = ['binary_crossentropy', 'mae'], optimizer = opt, loss_weights = [1,100]) # For training using MAE loss
    model.compile(loss=['binary_crossentropy', percept_loss_func], optimizer=opt, loss_weights=[1, 100]) # For training using perceptual loss
    return model

def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    ix = randint(0, x_train_aug.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

def generate_real_samples_validation(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    X1, Y1 = trainA, trainB
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, Y1], y

def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

###Save GAN model###
def summarize_performance(step, g_model):
    filename2 = 'mmri-gan_model.h5'
    g_model.save(filename2)
    print('>Saved: %s' % (filename2))

def train(d_model, g_model, gan_model, dataset, n_epochs=N_Epochs, n_batch=Batch_Size):
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs

    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))

        if (i + 1) % (bat_per_epo * 400) == 0: # Can be modified to save model more often (i.e. every 10 or 100 epochs)
            summarize_performance(i, g_model)

print('Loaded', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]
input_shape = dataset[0].shape[1:]
output_shape = dataset[1].shape[1:]
print('Input shape', input_shape)
print('Output shape', output_shape)
print('Image shape', image_shape)

model = define_generator(image_shape)
model.summary()

d_model = define_discriminator(input_shape, output_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)
gan_model.summary()

train(d_model, g_model, gan_model, dataset)

plot_model(model, to_file='generator_model_plot.png', show_shapes=True, show_layer_names=True)
