# Activity 8.01: Food Classification with Fine-Tuning.

The “10_food_classes_all_data” dataset 
https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip

This data set consists of 10 food categories, with 10,000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels. Our aim is to train a deep learning model which can successfully classify food images. 

In this activity, you are tasked with training a NASNetMobile model to recognize images of different varieties of foods (classification into 10 different classes). You will use fine-tuning to train the final layers of this model. 

Follow these steps will help you to complete this activity:

1.	Import the dataset and unzip the file using TensorFlow.
2.	Create a data generator with the following data augmentation:

![](https://github.com/Neha-Chiluka/tf/blob/main/images/23.png?raw=true)

3.	Load a pre-trained NASNETMobile model from TensorFlow.

4.	Freeze the first 600 layers of the model.

5.	Add two fully connected layers on top of NASNETMobile

-	A fully connected layer with Dense(1000, activation=relu)
-	A fully connected layer with Dense(10, activation=’softmax’)

6.	Specify an Adam Optimizer with a learning rate 0.001
7.	Train the model.
8.	Evaluate the model on the test set.

The expected output is as follows:

![](https://github.com/Neha-Chiluka/tf/blob/main/images/24.png?raw=true)

