# Activity 5.01: Building a Multi-Class with TensorFlow

In this activity, you are tasked with building and training a multi-class classifier that will predict Motor Failure Time for given configuration of blades.

In this dataset, we have three independent variables pctid, x, y, and z and dependent variable wconfid

The dataset can be accessed from [here](http://https://raw.githubusercontent.com/fenago/tf/main/Chapter5-Classification_Models/datasets/accelerometer.csv "here")

1.	Load the data with read_csv() from pandas. 
2.	Extract the target variable with pop() method from pandas. 
3.	Split the data into training and test sets.
4.	Build the multi-class classifier with five fully connected layers of 512, 512, 128, 128, and 26 units, respectively. 
5.	Train this model on the training set. 
6.	Evaluate its performance on the test set with evaluate() method from TensorFlow. 
7.	Print the confusion matrix with confusion_matrix() from TensorFlow.


![](https://github.com/Neha-Chiluka/tf/blob/main/images/13.png?raw=true)

