# Activity 6.02: Predicting Motor Failure with Bayesian Optimization from Keras Tuner

In this activity, you are tasked with building and training a multi-class classifier that will predict Motor Failure Time for given configuration of blades.

In this dataset, we have three independent variables pctid, x, y, and z and dependent variable wconfid

You will perform automatic hyperparameter tuning with keras Tuner and find the best combination of hyperparameters for the learning rate, the number of units for the input layer, and L2 regularization with Bayesian optimization.

The dataset can be accessed from [here](http://https://raw.githubusercontent.com/fenago/tf/main/Chapter5-Classification_Models/datasets/accelerometer.csv "here")

1.	Open a new Jupyter notebook. 
2.	Import the required libraries.
3.	Create a data_url and load the data using the read_csv() method.
4.	Separate the X and y variables 
5.	Split the data into train and test sets
6.	Build the multi-class classifier with five fully connected layers of, respectively, 512, 512, 128, 128, and 26 units and the three different hyperparameters to be tuned: the learning rate (between 0.01 and 0.001), the number of units for the input layer (between 128 and 512 and a step of 64), and L2 regularization (between 0.1, 0.01, and 0.001).
7.	Find the best combination of hyperparameters with Bayesian optimization
8.	Train the model on the training set with the best hyper parameters found

![](https://github.com/Neha-Chiluka/tf/blob/main/images/16.png?raw=true)
