# Face-Detection-and-Recogition-TF-FN-OCV

# Face Detection and Recogition Using Tensorflow, FaceNet, OpenCV

## A General ML Framework:
![](https://github.com/E-B-Manohar/Black-Hole-Source/blob/master/Images/References/DS/a%20general%20ml%20model.png)

## 1. Dataset Extraction:
The sample images of various celebrities like Bill Hader, Bobby Moynihan, Jason Sudeikis, Kate McKinnon, Kenan Thompson, Kristen Wiig are from various sources. These data contains a large variance so that the model learns maximum features with no bias in the images. I've tried to obtain faces with glasses, makeup, lots of expressions like laugh, smile, sad, angry, neutral, side angles so on to avoid biasness in recognizing the faces from single point of view.

This data is stored in the directory "Raw Images" with labels as the directory names within the raw_images directory.
These labeled directories contain upto 25 images of these people with different backgroud textures.

## Labels

Bill Hader            |  Bobby Moynihan   |  Jason Sudeikis
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/E-B-Manohar/Face-Detection-and-Recogition-TF-FN-OCV/blob/master/raw_images/Bill_Hader/00007.jpg" height="300" width="300">  |  <img src="https://github.com/E-B-Manohar/Face-Detection-and-Recogition-TF-FN-OCV/blob/master/raw_images/bobby_moynihan/00025.jpg" height="300" width="300">|  <img src="https://github.com/E-B-Manohar/Face-Detection-and-Recogition-TF-FN-OCV/blob/master/raw_images/jason_sudeikis/00024.jpg" height="300" width="300">


Kate McKinnon           |  Kenan Thompson   |  Kristen Wiig
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/E-B-Manohar/Face-Detection-and-Recogition-TF-FN-OCV/blob/master/raw_images/Kate_McKinnon/00007.jpg" height="300" width="300">  |  <img src="https://github.com/E-B-Manohar/Face-Detection-and-Recogition-TF-FN-OCV/blob/master/raw_images/kenan_thompson/00018.jpg" height="300" width="300">|  <img src="https://github.com/E-B-Manohar/Face-Detection-and-Recogition-TF-FN-OCV/blob/master/raw_images/kristen_wiig/00012.jpg" height="300" width="300">




# FaceNet Architecture Flow:
![FaceNet Architecture Flow:](https://github.com/E-B-Manohar/Object-Classification-with-Keras-using-Transfer-Learning/blob/master/FaceNet_Archicecture.PNG)










References:
https://github.com/davidsandberg/facenet
http://www.aisangam.com/blog/real-time-face-recognition-using-facenet/
A Discriminative Feature Learning Approach for Deep Face Recognition
Deep Face Recognition
