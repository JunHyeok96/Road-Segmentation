import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense,Lambda, Activation,Concatenate,Reshape,Dropout, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dot, Conv2DTranspose,Cropping2D

        
def convolutional_block(X, filters,s = 2,rate=1):
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(F1, (1, 1), strides = (s, s))(X)
    X = BatchNormalization(axis = 3,momentum = 0.95)(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((rate, rate))(X)
    X = Conv2D(F2, (3, 3), strides = (1, 1),dilation_rate=rate)(X)
    X = BatchNormalization(axis = 3,momentum = 0.95)(X)
    X = Activation('relu')(X)
    X = Conv2D(F3, (1, 1), strides = (1, 1))(X)
    X = BatchNormalization(axis = 3,momentum = 0.95)(X)
    X_proj = Conv2D(F3, (1, 1), strides = (s, s))(X_shortcut)
    X_proj = BatchNormalization(axis = 3,momentum = 0.95)(X_proj)
    X = Add()([X,X_proj])
    X = Activation('relu')(X)
    return X

def identity_block(X, filters):
    
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2, kernel_size = (3, 3), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis=3)(X)

    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    return X

    
def global_average_pooling(input, gap_size):
    w, h, c = (input.shape[1:])
    x = AveragePooling2D((w/gap_size, h/gap_size))(input)
    x = Conv2D(c//4, (1,1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(lambda x: tf.image.resize(x, size=(w, h)))(x)
    return x

def resnet(input):
    X = ZeroPadding2D((1, 1))(input)
    X = Conv2D(64, (3, 3), strides = (2, 2))(X)
    X = BatchNormalization(axis = 3,momentum = 0.95)(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(64, (3, 3), strides = (1, 1))(X)
    X = BatchNormalization(axis = 3,momentum = 0.95)(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(128, (3, 3), strides = (1, 1))(X)
    X = BatchNormalization(axis = 3,momentum = 0.95)(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, [64,64,256],s = 1)
    X = identity_block(X,[64,64,256])
    X = identity_block(X,[64,64,256])

    X = convolutional_block(X, [128,128,512],s = 2)
    X = identity_block(X,[128,128,512])
    X = identity_block(X,[128,128,512])
    X = identity_block(X,[128,128,512])

    X = convolutional_block(X, [256,256,1024],s = 1,rate=2)
    X = identity_block(X,[256,256,1024])
    X = identity_block(X,[256,256,1024])
    X = identity_block(X,[256,256,1024])
    X = identity_block(X,[256,256,1024])
    X = identity_block(X,[256,256,1024])

    X = convolutional_block(X, [512,512,2048],s = 1,rate=4)
    X = identity_block(X,[512,512,2048])
    X = identity_block(X,[512,512,2048])
    return X

def pspnet(input, input_shape, n_classes):
    feature_map = resnet(input)      
    pooling_1 = global_average_pooling(feature_map, 1)
    pooling_2 = global_average_pooling(feature_map, 2)
    pooling_3 = global_average_pooling(feature_map, 3)
    pooling_4 = global_average_pooling(feature_map, 6)
    x = tf.keras.layers.Concatenate(axis=-1)([pooling_1,pooling_2,pooling_3,pooling_4])

    x = Conv2D(512, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_classes, (1,1), padding="same", activation='relu')(x)
    return  x


def cff(x):
    x = Lambda(lambda x: tf.image.resize(x, (x.shape[1]*2, x.shape[2]*2)))(x)
    x = Conv2D(64, (3, 3), padding="same", dilation_rate = 2)(x)
    x = BatchNormalization()(x)
    return x

def top_branch(input, input_shape, n_classes):   #1/4 -> 1/32
    top_input = Lambda(lambda x: tf.image.resize(x, (input_shape[0], input_shape[1])))(input)
    x = pspnet(top_input, input_shape, n_classes)

    cff_1 = cff(x)
    x = Lambda(lambda x: tf.image.resize(x, (x.shape[1]*2, x.shape[2]*2)))(x)
    top_predict = Conv2D(n_classes, (1,1), padding="same", name ="sub4")(x)
    
    return cff_1, top_predict

def middle_branch(input,input_shape, n_classes): #1/2 -> 1/16
    middle_input = Lambda(lambda x: tf.image.resize(x, (input_shape[0], input_shape[1])))(input)

    x = Conv2D(128, (3,3), padding= "same")(middle_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)


    x = Conv2D(128, (3,3), padding= "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, (3,3), padding= "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (1,1), padding= "same")(x)
    x = BatchNormalization()(x)

    cff_1, top_predict = top_branch(middle_input, (input_shape[0]//2, input_shape[1]//2, 3), n_classes)
    x = Add()([x, cff_1])
    x = Activation("relu")(x)

    cff2 = cff(x)
    
    x = Lambda(lambda x: tf.image.resize(x, (x.shape[1]*2, x.shape[2]*2)))(x)
    middle_predict = Conv2D(n_classes, (1,1), name ="sub24")(x)
    
    return cff2, top_predict, middle_predict

def icnet(input_shape, n_classes, train_mode = False): #1-> 1/4
    
    input= tf.keras.layers.Input(shape =input_shape)
    
    x = Conv2D(128, (3,3), padding= "same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    
    x = Conv2D(128, (3,3), padding= "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, (3,3), padding= "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (1,1) , padding= "same")(x)
    x = BatchNormalization()(x)

    cff_2, top_predict, middle_predict = middle_branch(input, (input_shape[0]//2, input_shape[1]//2, 3), n_classes)
    
    x = Add()([x, cff_2])
    x = Activation("relu")(x)
    
    x = Lambda(lambda x: tf.image.resize(x, (x.shape[1]*2, x.shape[2]*2)))(x)

    bottom_predict = Conv2D(n_classes, (1,1), padding = "same", name ="sub124")(x)
  
    if train_mode:
        return tf.keras.Model(inputs=input, outputs=[top_predict, middle_predict, bottom_predict])

    else:
        x = Lambda(lambda x: tf.image.resize(x, (bottom_predict.shape[1]*4, bottom_predict.shape[2]*4)))(bottom_predict)
        return tf.keras.Model(inputs=input, outputs=[x])
    