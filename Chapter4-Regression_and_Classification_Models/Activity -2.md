# Activity 4.02: Creating a Multi-Layer Classification ANN with TensorFlow.

This dataset was generated for use on 'Prediction of Motor Failure Time Using An Artificial Neural Network' project. A cooler fan with weights on its blades was used to generate vibrations. To this fan cooler was attached an accelerometer to collect the vibration data. With this data, motor failure time predictions were made, using an artificial neural networks. To generate three distinct vibration scenarios, the weights were distributed in two different ways: 1) 'red' - normal configuration: two weight pieces positioned on neighboring blades; angle; 2) 'green' - opposite configuration: two weight pieces positioned on opposite blades.

Your target will have a 1 value for red weight configuration and 0 for green weight configuration.


You can find the “accelerometer.csv” file can be found [here](http://https://github.com/fenago/tf/blob/main/Chapter4-Regression_and_Classification_Models/dataset/accelerometer.csv "here")

Perform the following steps to complete this activity:

1)	Open a Jupyter notebook to complete the activity. 
2)	Import the TensorFlow and pandas libraries. 
3)	Load in the accelerometer.csv dataset. 
4)	Drop any rows that have null values. 
5)	Create feature and target variables.
6)	Initialize a model of the Keras Sequential class. 
7)	Add an input layer, three hidden layers of sizes 32, 16, and 8, and an output layer with a sigmoid activation function of size 1 to the model. 
8)	Compile the model with an RMSprop optimizer with a learning rate equal to 0.0001 and binary cross-entropy for the loss and compute the accuracy metric. 
9)	Add a callback to write logs to TensorBoard. 
10)	Fit the model to the training data for 50 epochs and a validation split equal to 0%. 
11)	Evaluate the model on the training data. 
12)	View the model architecture and model-fitting process in TensorBoard.


