import os
import math
import keras
import numpy as np
import tensorflow
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation,Reshape,Dropout, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dot, Conv2DTranspose,Cropping2D,UpSampling2D, Concatenate, Lambda
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img  
import cv2
from keras.applications.vgg16 import VGG16
from keras.layers.merge import concatenate
from keras.layers import DepthwiseConv2D

import keras.backend as K
K.set_learning_phase(1)

def identity_block(X, filters):

    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2, kernel_size = (3, 3), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    return X
    
def convolutional_block(X, filters, s = 2):

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides = (s,s), padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(F2, (3, 3), strides = (1, 1), padding='same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    X = Conv2D(F3, (1, 1), strides = (1, 1),padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)

    X_shortcut = Conv2D(F3, (1, 1), strides = (s, s),padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X
    
def Deeplab_v3(input_shape, n_classes):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(X)

    X = convolutional_block(X, filters = [64, 64, 256], s = 1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])
    X_temp = X

    X = convolutional_block(X, filters=[128,128,512],s = 2)
    X = identity_block(X, filters=[128,128,512])
    X = identity_block(X, filters=[128,128,512])
    X = identity_block(X, filters=[128,128,512])

    X = convolutional_block(X, filters=[256, 256, 1024],s = 2)
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])


    X = convolutional_block(X, filters=[512, 512, 2048], s = 1)
    X = identity_block(X, filters=[256, 256, 2048])
    X = identity_block(X, filters=[256, 256, 2048])
    
    h = X.shape[1]
    w = X.shape[2]
    conv_1x1 = Conv2D(256, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    conv_1x1 = BatchNormalization(axis = 3)(conv_1x1)
    conv_1x1 = Activation('relu')(conv_1x1)
    conv_3x3_1 = Conv2D(256, kernel_size = (3, 3),strides = (1,1) , padding = 'same', dilation_rate=6)(X)
    conv_3x3_1 = BatchNormalization(axis = 3)(conv_3x3_1)
    conv_3x3_1 = Activation('relu')(conv_3x3_1)
    conv_3x3_2 = Conv2D(256, kernel_size = (3, 3),strides = (1,1) , padding = 'same', dilation_rate=12)(X)
    conv_3x3_2 = BatchNormalization(axis = 3)(conv_3x3_2)
    conv_3x3_2 = Activation('relu')(conv_3x3_2)
    conv_3x3_3 = Conv2D(256, kernel_size = (3, 3),strides = (1,1) , padding = 'same', dilation_rate=18)(X)
    conv_3x3_3 = BatchNormalization(axis = 3)(conv_3x3_3)
    conv_3x3_3 = Activation('relu')(conv_3x3_3)
    image_level_features = keras.layers.pooling.GlobalMaxPooling2D()(X)
    image_level_features = Reshape((1,1,2048))(image_level_features)
    image_level_features = Conv2D(256, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(image_level_features)
    image_level_features = Reshape((1,1,256))(image_level_features)
    image_level_features = Lambda(lambda x: tensorflow.image.resize(x, (h, w)))(image_level_features)
    X_concat = Concatenate(axis=-1)([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features])
    X_concat = Conv2D(256, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X_concat)
    X_concat = UpSampling2D(size=(4, 4))(X_concat)
    
    X_decoder = Conv2D(48, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X_temp)
    X_decoder = BatchNormalization(axis = 3)(X_decoder)
    X_decoder = Activation('relu')(X_decoder)
    X = Concatenate(axis=-1)([X_decoder, X_concat])
    X = Conv2D(256, kernel_size = (3, 3), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = Conv2D(256, kernel_size = (3, 3), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(1, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = UpSampling2D(size=(4, 4))(X)
    
    model = Model(inputs = X_input, outputs = X)
    
    return model
