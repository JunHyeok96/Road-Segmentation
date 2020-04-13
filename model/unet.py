import numpy as np
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import  Flatten, Conv2D, MaxPooling2D, Dense, Dropout, Softmax, Conv2DTranspose, BatchNormalization, Activation


def contract_path(input_shape):
    input= tf.keras.layers.Input(shape = input_shape)
    x =  Conv2D(64, (3,3), padding = "same", activation = "relu")(input)
    x =  Conv2D(64, (3,3), padding = "same", activation = "relu", name = "copy_crop1")(x)
    x = MaxPooling2D((2, 2))(x)
    x =  Conv2D(128, (3,3), padding = "same", activation = "relu")(x)
    x =  Conv2D(128, (3,3), padding = "same", activation = "relu", name = "copy_crop2")(x)
    x = MaxPooling2D((2, 2))(x)
    x =  Conv2D(256, (3,3), padding = "same", activation = "relu")(x)
    x =  Conv2D(256, (3,3), padding = "same", activation = "relu", name = "copy_crop3")(x)
    x = MaxPooling2D((2, 2))(x)
    x =  Conv2D(512, (3,3), padding = "same", activation = "relu")(x)
    x =  Conv2D(512, (3,3), padding = "same", activation = "relu", name = "copy_crop4")(x)
    x = MaxPooling2D((2, 2))(x)
    x =  Conv2D(1024, (3,3), padding = "same", activation = "relu")(x)
    x =  Conv2D(1024, (3,3), padding = "same", activation = "relu", name = "last_layer")(x)
    contract_path =  tf.keras.Model(inputs = input, outputs = x)
    return contract_path

def conv_block(input, filters, stride):
    x =  Conv2D(filters, stride , padding = "same")(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def unet(input_shape, n_classes):
    contract_model = contract_path(input_shape=input_shape)
    layer_names  = ["copy_crop1", "copy_crop2",  "copy_crop3" ,"copy_crop4", "last_layer"]
    layers = [contract_model.get_layer(name).output for name in layer_names]

    extract_model = tf.keras.Model(inputs=contract_model.input, outputs=layers)
    input= tf.keras.layers.Input(shape =input_shape)
    output_layers = extract_model(inputs = input)
    last_layer = output_layers[-1]

    x = Conv2DTranspose(512, 4, (2,2), padding = "same", activation = "relu")(last_layer)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[3]])

    x = conv_block(x, 256, (3,3))
    x = conv_block(x, 256, (3,3))

    x = Conv2DTranspose(256, 4, (2,2), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[2]])

    x = conv_block(x, 128, (3,3))
    x = conv_block(x, 128, (3,3))


    x = Conv2DTranspose(128, 4, (2,2), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[1]])

    x = conv_block(x, 64, (3,3))
    x = conv_block(x, 64, (3,3))

    x = Conv2DTranspose(64, 4, (2,2), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[0]])

    x = conv_block(x, 64, (3,3))
    x = conv_block(x, 64, (3,3))

    x =  Conv2D(n_classes, (1,1), activation = "relu")(x)

    return tf.keras.Model(inputs = input , outputs = x)