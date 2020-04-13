from data_loader import train_images, mask_images
import shutil

train_images = train_images()
mask_images = mask_images()

train_num = int(len(train_images)*0.75)

test_img = train_images[train_num:]
test_label = mask_images[train_num:]

test_img_path = ['../dataset/test_img/' +test_img[i].split("/")[-1] for i in range(len(test_img))]
test_label_path = ['../dataset/test_label/' +test_label[i].split("/")[-1] for i in range(len(test_label))]

for i in range(len(test_img)):
    shutil.move(test_img[i], test_img_path[i])

for i in range(len(test_label)):
    shutil.move(test_label[i], test_label_path[i])