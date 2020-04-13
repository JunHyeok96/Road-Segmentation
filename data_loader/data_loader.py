import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

label_rgb =dict()
label_rgb["background"] = np.array([0, 0, 0])
label_rgb["sidewalk_blocks"] = np.array([0, 0, 255])
label_rgb["sidewalk_cement"] = np.array([217, 217, 217])
label_rgb["sidewalk_urethane"] = np.array([198, 89, 17])
label_rgb["sidewalk_asphalt"] = np.array([128, 128, 128])
label_rgb["sidewalk_soil_stone"] = np.array([255, 230, 153])
label_rgb["sidewalk_damaged"] = np.array([55, 86, 35])
label_rgb["sidewalk_other"] = np.array([110, 168, 70])
label_rgb["braille_guide_blocks_normal"] = np.array([255, 255, 0])
label_rgb["braille_guide_blocks_damaged"] = np.array([128, 96, 0])
label_rgb["roadway_normal"] = np.array([255, 128, 255])
label_rgb["roadway_crosswalk"] =np.array([255, 0, 255])
label_rgb["alley_normal"] = np.array([230, 170, 255])
label_rgb["alley_crosswalk"] = np.array([208, 88, 255])
label_rgb["alley_speed_bump"] = np.array([138, 60, 200])
label_rgb["alley_damaged"] = np.array([88, 38, 128])
label_rgb["bike_lane_normal"] = np.array([255, 155, 155])
label_rgb[ "caution_zone_stairs"] = np.array([255, 192, 0])
label_rgb[ "caution_zone_manhole"] = np.array([255, 0, 0])
label_rgb[ "caution_zone_tree_zone"] = np.array([0, 255, 0])
label_rgb[ "caution_zone_grating"] = np.array([255, 128, 0])
label_rgb[ "caution_zone_repair_zone"] = np.array([105, 105, 255])

label_list = list()
label_list.append(["background"])
label_list.append(["bike_lane_normal", "sidewalk_asphalt", "sidewalk_urethane"])
label_list.append(["caution_zone_stairs", "caution_zone_manhole", "caution_zone_tree_zone", "caution_zone_grating", "caution_zone_repair_zone"])
label_list.append(["alley_crosswalk","roadway_crosswalk"])
label_list.append(["braille_guide_blocks_normal", "braille_guide_blocks_damaged"])
label_list.append(["roadway_normal","alley_normal","alley_speed_bump", "alley_damaged"])
label_list.append(["sidewalk_blocks","sidewalk_cement" , "sidewalk_soil_stone", "sidewalk_damaged","sidewalk_other"])


AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_WIDTH = 480
IMG_HEIGHT = 272

def train_images():
    top_foloder = "./dataset/surface_img/"
    data_folder = os.listdir(top_foloder)
    data_folder.sort()
    data_folder = list(map(lambda x : top_foloder + x, data_folder))
    return data_folder

def mask_images():
    top_foloder = "./dataset/surface_label/"
    mask_folder = os.listdir(top_foloder)
    mask_folder.sort()
    mask_folder = list(map(lambda x : top_foloder + x, mask_folder))
    return mask_folder

def get_label(file_path):
    img = tf.io.read_file(file_path)
    label_img = tf.image.decode_jpeg(img, channels=3)
    label_img = tf.image.resize(label_img, [IMG_HEIGHT, IMG_WIDTH])
    return label_img

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

def data_load():
    list_ds_train = tf.data.Dataset.list_files("./dataset/surface_img/*", shuffle=False)
    list_ds_train_label = tf.data.Dataset.list_files("./dataset/surface_label/*", shuffle=False)


    # Set `num_parallel__calls` so multiple images are loaded/processed in parallel.
    train_ds = list_ds_train.map(process_path, num_parallel_calls=AUTOTUNE)
    train_label_ds = list_ds_train_label.map(get_label, num_parallel_calls=AUTOTUNE)


    train_dataset = tf.data.Dataset.zip((train_ds, train_label_ds))


    list_ds_test = tf.data.Dataset.list_files("./dataset/test_img/*", shuffle=False)
    list_ds_test_label = tf.data.Dataset.list_files("./dataset/test_label/*", shuffle=False)


    # Set `num_parallel__calls` so multiple images are loaded/processed in parallel.
    test_ds = list_ds_test.map(process_path, num_parallel_calls=AUTOTUNE)
    test_label_ds = list_ds_test_label.map(get_label, num_parallel_calls=AUTOTUNE)


    test_dataset = tf.data.Dataset.zip((test_ds, test_label_ds))

    return train_dataset, test_dataset

def convert_class(label_img):
    for i in range(len(label_img)):
        for j in label_list[1]:
            label_img[i][(label_img[i]==label_rgb[j]).all(axis=2)] = 1
        for j in label_list[2]:
            label_img[i][(label_img[i]==label_rgb[j]).all(axis=2)] = 2
        for j in label_list[3]:
            label_img[i][(label_img[i]==label_rgb[j]).all(axis=2)] = 3
        for j in label_list[4]:
            label_img[i][(label_img[i]==label_rgb[j]).all(axis=2)] = 4
        for j in label_list[5]:
            label_img[i][(label_img[i]==label_rgb[j]).all(axis=2)] = 5
        for j in label_list[6]:
            label_img[i][(label_img[i]==label_rgb[j]).all(axis=2)] = 6
            
        label_img[i][(label_img[i]!=0 ) & (label_img[i]!=1 ) &(label_img[i]!= 2) &(label_img[i]!= 3) &(label_img[i]!= 4) &(label_img[i]!= 5) &(label_img[i]!= 6)]=0    
    label_img = np.array(label_img)
    label_img= label_img[:,:,:,0]
    label_img = label_img[..., tf.newaxis]

    return label_img