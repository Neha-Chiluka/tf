# Activity 8.02: Transfer Learning with TensorFlow Hub

In this activity, you are required to correctly classify images of pizzas and steaks using transfer learning. Rather than training a model from scratch, you will benefit from the EfficientNet B0 feature vector from TensorFlow Hub, which contains pre-computed weights that can recognize different types of objects.

You can find the dataset here
https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip

The following steps will help you to complete this activity:
1.	Import the dataset and unzip the file using TensorFlow.
2.	Create a data generator that will perform rescaling.
3.	Load a pre-trained EfficientNet B0 feature vector from TensorFlow Hub.
4.	Add two fully connected layers on top of the feature vector:
– A fully connected layer with Dense(500, activation=relu)
– A fully connected layer with Dense(1, activation='sigmoid')
5.	Specify an Adam optimizer with a learning rate of 0.001.
6.	Train the model.
7.	Evaluate the model on the test set.
The expected output is as follows:
 
![](https://github.com/Neha-Chiluka/tf/blob/main/images/25.png?raw=true)
