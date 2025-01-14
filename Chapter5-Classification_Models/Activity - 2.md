# Activity 5.02: Building a Handwritten Digit Classification model with TensorFlow.

In this activity, you are tasked with building and training a multi-label classifier that will classify the 10 digits label of numeric values from 0 to 9 from images. In this dataset, we have 784-pixel feature variable which are independent variable and 1 target label variable with values 0 to 9. The goal of this model is to determine which of the 10 digit each observation belongs to.

The training dataset can be accessed [here](http://https://github.com/fenago/tf/blob/main/Chapter5-Classification_Models/datasets/MNIST_train.csv?raw=true "here")
The testing dataset can be accessed [here](http://https://github.com/fenago/tf/blob/main/Chapter5-Classification_Models/datasets/MNIST_test.csv?raw=true "here")

1.	Load the data with read_csv() from pandas. 
2.	Extract the target variable with pop() method from pandas. 
3.	Split the data into training and test sets.
4.	Build the multi-class classifier with five fully connected layers of 512, 512, 128, 128, and 26 units, respectively. 
5.	Train this model on the training set. 
6.	Evaluate its performance on the test set with evaluate() method from TensorFlow

![](https://github.com/Neha-Chiluka/tf/blob/main/images/14.png?raw=true)
