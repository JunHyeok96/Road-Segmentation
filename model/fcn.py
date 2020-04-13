import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.layers import  Flatten, Conv2D, MaxPooling2D, Dense, Dropout, Softmax, Conv2DTranspose, BatchNormalization
from tensorflow_examples.models.pix2pix import pix2pix


def vgg16(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding ="same", activation = "relu", input_shape=input_shape))
    model.add(Conv2D(64, (3,3), padding ="same", activation = "relu"))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Conv2D(128, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(128, (3,3), padding = "same", activation = "relu"))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Conv2D(256, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(256, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(256, (3,3), padding = "same", activation = "relu"))
    model.add(MaxPooling2D((2,2), strides=(2, 2), name = 'block3_pool'))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(MaxPooling2D((2,2), strides=(2, 2), name = 'block4_pool'))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(MaxPooling2D((2,2), strides=(2, 2), name = 'block5_pool'))
    return model


def fcn(input_shape, n_class):
    vgg_model = vgg16(input_shape)
    vgg_model.load_weights("./model/vgg16.h5")
    #vgg_model = VGG16(input_shape = input_shape, weights = 'imagenet', include_top=False)
    vgg_model.trainable = False
    layer_names  = ['block3_pool', 'block4_pool', 'block5_pool']
    layers = [vgg_model.get_layer(name).output for name in layer_names]
    fcn_model = tf.keras.Model(inputs=vgg_model.input, outputs=layers)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    skip = fcn_model(x)
    vgg_last_layer = skip[-1]
    x = vgg_last_layer
    x =  Conv2D(4096, (7,7), padding = "same", activation = "relu")(x)
    x = Conv2D(3, (1,1), padding = "same", activation = "relu")(x)
    x = Conv2DTranspose(512, 4, (2,2), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([skip[1], x])
    x = Conv2DTranspose(256, 4, (2,2), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([skip[0], x])
    x = BatchNormalization()(x)
    x = Conv2DTranspose(n_class, 16, (8,8), padding = "same", activation = "relu")(x)
    model =  tf.keras.Model(inputs = inputs , outputs = x)
    return model
