import numpy as np
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import  Flatten, Conv2D, MaxPooling2D, Dense, Dropout, Softmax, Conv2DTranspose, BatchNormalization,Activation, AveragePooling2D


def vgg16(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding ="same", activation = "relu", input_shape=input_shape))
    model.add(Conv2D(64, (3,3), padding ="same", activation = "relu", name = "copy_crop1"))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Conv2D(128, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(128, (3,3), padding = "same", activation = "relu", name = "copy_crop2"))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Conv2D(256, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(256, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(256, (3,3), padding = "same", activation = "relu", name = "copy_crop3"))
    model.add(MaxPooling2D((2,2), strides=(2, 2), name = 'block3_pool'))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu", name = "copy_crop4"))
    model.add(MaxPooling2D((2,2), strides=(2, 2), name = 'block4_pool'))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(MaxPooling2D((2,2), strides=(2, 2), name = "last_layer"))
    return model

def global_average_pooling(input, gap_size):
    w, h, c = (input.shape[1:])
    x = AveragePooling2D((w/gap_size, h/gap_size))(input)
    x = Conv2D(c//4, (1,1), padding="same", activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.image.resize(x, (w, h))
    return x


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

def pspunet(input_shape, n_classes):
    contract_model = contract_path(input_shape=input_shape)
    layer_names  = ["copy_crop1", "copy_crop2",  "copy_crop3" ,"copy_crop4"]
    layers = [contract_model.get_layer(name).output for name in layer_names]

    extract_model = tf.keras.Model(inputs=contract_model.input, outputs=layers)
    input= tf.keras.layers.Input(shape =input_shape)
    output_layers = extract_model(inputs = input)
    last_layer = output_layers[-1]
    
    feature_map = last_layer
    pooling_1 = global_average_pooling(feature_map, 1)
    pooling_2 = global_average_pooling(feature_map, 2)
    pooling_3 = global_average_pooling(feature_map, 3)
    pooling_4 = global_average_pooling(feature_map, 6)
    x = tf.keras.layers.Concatenate(axis=-1)([pooling_1,pooling_2,pooling_3,pooling_4])
    x =  Conv2D(256, (1,1), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = tf.keras.layers.Concatenate()([x, output_layers[3]])

    x =  Conv2D(256, (3,3), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x =  Conv2D(256, (3,3), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(256, 4, (2,2), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[2]])

    x =  Conv2D(128, (3,3), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x =  Conv2D(128, (3,3), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(128, 4, (2,2), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[1]])


    x =  Conv2D(64, (3,3), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x =  Conv2D(64, (3,3), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = Conv2DTranspose(64, 4, (2,2), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[0]])

    x =  Conv2D(64, (3,3), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x =  Conv2D(64, (3,3), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x =  Conv2D(n_classes, (1,1), activation = "relu")(x)
    
    return tf.keras.Model(inputs = input , outputs = x)