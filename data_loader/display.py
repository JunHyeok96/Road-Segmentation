from IPython.display import clear_output
from data_loader import *
import matplotlib.pyplot as plt
import tensorflow as tf


def display(display_list):
    plt.figure(figsize=(7, 7))
    for i in range(3):
        plt.subplot(3, 3, i*3+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i][0]/255))
        plt.axis('off')
        plt.subplot(3, 3, i*3+2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i][1]/255))
        plt.axis('off')
        plt.subplot(3, 3, i*3+3)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i][2]/255))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]   

def show_predictions(image, label, model):

    if model:
        pred_mask = [model.predict(image[tf.newaxis, ...]) for image in image]
        display_list = [[image[i], label[i], create_mask(pred_mask[i])] for i in range(3)]
        display(display_list)
    else :  
        display_list = [[image[i], label[i], image[i]] for i in range(3)]
        display(display_list)
        