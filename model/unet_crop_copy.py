import numpy as np
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import  Flatten, Conv2D, MaxPooling2D, Dense, Dropout, Softmax, Conv2DTranspose, BatchNormalization


def contract_path(input_shape):
    input= tf.keras.layers.Input(shape = input_shape)
    x =  Conv2D(64, (3,3), activation = "relu")(input)
    x =  Conv2D(64, (3,3), activation = "relu", name = "copy_crop1")(x)
    x = MaxPooling2D((2, 2))(x)
    x =  Conv2D(128, (3,3), activation = "relu")(x)
    x =  Conv2D(128, (3,3), activation = "relu", name = "copy_crop2")(x)
    x = MaxPooling2D((2, 2))(x)
    x =  Conv2D(256, (3,3), activation = "relu")(x)
    x =  Conv2D(256, (3,3), activation = "relu", name = "copy_crop3")(x)
    x = MaxPooling2D((2, 2))(x)
    x =  Conv2D(512, (3,3), activation = "relu")(x)
    x =  Conv2D(512, (3,3), activation = "relu", name = "copy_crop4")(x)
    x = MaxPooling2D((2, 2))(x)
    x =  Conv2D(1024, (3,3), activation = "relu")(x)
    x =  Conv2D(1024, (3,3), activation = "relu", name = "last_layer")(x)
    contract_path =  tf.keras.Model(inputs = input, outputs = x)
    return contract_path

def cropped_layer(input, crop_input):
    crop = crop_input
    x_shape =input.shape
    crop_shape = crop.shape
    offsets = [0, (crop_shape[1] - x_shape[1]) // 2, (crop_shape[2] - x_shape[2]) // 2, 0]
    size = [-1, x_shape[1], x_shape[2], -1]
    cropped = tf.slice(crop, offsets,size)
    x = tf.keras.layers.Concatenate()([cropped, input])
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
    x = cropped_layer(x, output_layers[3])

    x =  Conv2D(256, (3,3), activation = "relu")(x)
    x = BatchNormalization()(x)

    x =  Conv2D(256, (3,3), activation = "relu")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(256, 4, (2,2), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = cropped_layer(x, output_layers[2])

    x =  Conv2D(128, (3,3), activation = "relu")(x)
    x = BatchNormalization()(x)
    x =  Conv2D(128, (3,3), activation = "relu")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(128, 4, (2,2), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = cropped_layer(x, output_layers[1])


    x =  Conv2D(64, (3,3), activation = "relu")(x)
    x = BatchNormalization()(x)
    x =  Conv2D(64, (3,3), activation = "relu")(x)
    x = BatchNormalization()(x)


    x = Conv2DTranspose(64, 4, (2,2), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = cropped_layer(x, output_layers[0])

    x =  Conv2D(64, (3,3), activation = "relu")(x)
    x = BatchNormalization()(x)
    x =  Conv2D(64, (3,3), activation = "relu")(x)
    x = BatchNormalization()(x)
    x =  Conv2D(n_classes, (1,1), activation = "relu")(x)
    return tf.keras.Model(inputs = input , outputs = x)

model = unet(input_shape=(500,500,1), n_classes = 2)
print(model.summary())