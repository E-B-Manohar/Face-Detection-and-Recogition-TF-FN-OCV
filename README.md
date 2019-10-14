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


## Data Preprocessing:
run the step 1 ipynb file
### What is the internal process?
1. The images are sent to preprocess library which uses the detect faces, facenet mudules in the library which convolves with **pretrained weights** stored in form of **.npy** in the npy folder to detect the faces in a given image(suppose to contain single person to append with the class/person_name label)
2. Once the face is detected, the facenet module and detect faces mudule marks the boundaries of the face in the image and crops at the boundary and saves in the "face_dataset" directory


## Facial Feature Extraction by Training the model on the given 6 labels:
run the step 2 ipynb file
###  FaceNet Architecture Flow:
![FaceNet Architecture Flow:](https://github.com/E-B-Manohar/Object-Classification-with-Keras-using-Transfer-Learning/blob/master/FaceNet_Archicecture.PNG)

*The model file is large to upload, for time being it is refered from gdrive share. The file name in this folder should be "20191012-185253.pb" This readme file should be deleted once the modle file is placed in the model directory for limiting the execution errors.
Access file from: https://drive.google.com/open?id=1QXCPcUVt5l_h_JC92mL-vg05nlwyyC7e 
*

# Testing the model.
## Image (randomly pulled the pictures from the internet and combined together in MS Paint)
run Step 3 - Recognizing Faces in an Image File.ipynb which contains the name of the file as "example_02.jpg"
![](https://github.com/E-B-Manohar/Face-Detection-and-Recogition-TF-FN-OCV/blob/master/example_02.jpg)

## Output:

Accuracy : [0.93148507] 'Bill_Hader'

Accuracy : [0.93587521] 'jason_sudeikis'

Accuracy : [0.91804383] 'Kate_McKinnon'

Accuracy : [0.93157867] 'bobby_moynihan'

Accuracy : [0.93045618] 'jason_sudeikis'

Accuracy : [0.83106043] 'kenan_thompson'

Accuracy : [0.9621055]  'kristen_wiig'

Accuracy : [0.5370428]  'Kate_McKinnon'

![](https://github.com/E-B-Manohar/Face-Detection-and-Recogition-TF-FN-OCV/blob/master/example_02(Result).JPG)



## Video File: Picked from YouTube:
Video Name: "Fox News: End of an Era - SNL" and the Url: https://www.youtube.com/watch?v=2j3beRaYExU

The video file inputted is : "example_mov.avi"

The output result is       : example_mov Result(fast forwarded).avi

Two snapshots of the result video file:

Recognised "Jason Sudeikis"
![](https://github.com/E-B-Manohar/Face-Detection-and-Recogition-TF-FN-OCV/blob/master/Capture1.JPG)

Since the model is not trained to recognize "Fred Armisen", the actor was not recognized by the model:
![](https://github.com/E-B-Manohar/Face-Detection-and-Recogition-TF-FN-OCV/blob/master/Capture2.JPG)

Accuracy Results for the face recognition in the video file is saved in the respective ipynb file : "Step 3 - Recognizing Faces in each frame of a video file.ipynb"



References of various codes, research:
1. https://github.com/davidsandberg/facenet
2. http://www.aisangam.com/blog/real-time-face-recognition-using-facenet/
3. A Discriminative Feature Learning Approach for Deep Face Recognition
4. Deep Face Recognition
5. https://drive.google.com/open?id=1QXCPcUVt5l_h_JC92mL-vg05nlwyyC7e
