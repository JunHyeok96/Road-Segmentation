# Road Segmentation 
 

![ezgif com-resize](https://user-images.githubusercontent.com/52908154/79122972-03e37b00-7dd4-11ea-895b-c8be61958a38.gif)
![ezgif com-resize (1)](https://user-images.githubusercontent.com/52908154/79123208-96841a00-7dd4-11ea-9f86-aa9909b77f39.gif)



* ### Development environment


  ubuntu 18.0.4, tensorflow 2.0.0, opencv-python 4.2.0.32, numpy 1.18.2

* ### model


  [FCN](https://github.com/seraaaayeo/SellyDev/blob/road_segmentation/model/fcn.py)(test x)
  
  [UNet](https://github.com/seraaaayeo/SellyDev/blob/road_segmentation/model/unet.py) (Test complete)
  
   [PSPNet](https://github.com/seraaaayeo/SellyDev/blob/road_segmentation/model/pspnet.py)(Modification Required)
   
   [PSPUNet](https://github.com/seraaaayeo/SellyDev/blob/road_segmentation/model/pspunet.py)(PSPNet + Unet, Test complete) 
   
   [ICNet](https://github.com/seraaaayeo/SellyDev/blob/road_segmentation/model/icnet.py)(Test complete)

![image](https://user-images.githubusercontent.com/52908154/79126562-2a58e480-7ddb-11ea-90ee-0488cffe1ad2.png)


* ### Learning environment

  * Dataset - [AI Hub](http://www.aihub.or.kr/) sidewalk walking image

  * train data : 38000, val data : 12800

  * IMG_WIDTH = 480

  * IMG_HEIGHT = 272

  * n_classes = 7

  * data argumentation - random_flip_horizon, random brightness

  * learning rate : 1e-4  ->  epoch>10 lr decay(1e-5) 

* ### Class

|class|label|
|------|---|
|Background|0|
|Bike_lane|1|
|Caution_zone|2|
|Crosswalk|3|
|braille_guide_blocks|4|
|Roadway|5|
|Sidewalk|6|

[detail](https://github.com/seraaaayeo/SellyDev/blob/road_segmentation/data_loader/data_loader.py)

* ### Quick start 
I provide a pretraining model weight. (pspunet loss 0.3160 mIoU 74.5% acc 90.2%)

```
git clone https://github.com/seraaaayeo/SellyDev.git
cd road_segmentation
python3 demo.py 
```

#### PSPUnet 

<img width="527" alt="pspunet 0 3160_miou_0 745_acc90 2" src="https://user-images.githubusercontent.com/52908154/79119948-908a3b00-7dcc-11ea-990d-ec6c3482f367.png">

#### UNet

<img width="527" alt="unet_0 35704005_acc89 1" src="https://user-images.githubusercontent.com/52908154/79119959-97b14900-7dcc-11ea-98e0-f651eb9ba7d2.png">

* ### performance evaluation


  TEST GPU - RTX2060 SUPER

|model|accuracy|loss|mIoU|FPS|Size|
|------|---|---|---|---|--|
|PSPUnet|90.2%|0.3160|74.5%|24.8|39.6MB|
|UNet|89.1%|0.3570|70.9%|22.7|131MB|
