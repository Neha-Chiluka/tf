# Activity 4.01: Creating a Multi-Layer ANN with TensorFlow

The market historical data set of real estate valuation are collected from Sindian Dist., New Taipei City, Taiwan. Given a independent features like house age, nearest MRT station, etc. we need to predict the house price of unit area.

You can find the dataset in here.


1)	Open a new Jupyter notebook to implement this activity.
2)	Import the TensorFlow and pandas libraries. 
3)	Load in the Real estate valuation data set.csv dataset. 
4)	Drop any rows that have null values. 
5)	Set the target as the house price of unit area column and the feature dataset as the remaining columns. 
6)	Rescale the feature dataset using a standard scaler. 
7)	Initialize a model of the Keras Sequential class. 
8)	Add an input layer, four hidden layers of sizes 64, 32, 16, and 8, and an output layer of size 1 to the model. Add a ReLU activation function to the first hidden layer. 
9)	Compile the model with an RMSprop optimizer with a learning rate equal to 0.001 and the mean squared error for the loss. 
10)	Add a callback to write logs to TensorBoard. 
11)	Fit the model to the training data for 100 epochs, with a batch size equal to 32 and a validation split equal to 20%. 
12)	Evaluate the model on the training data. 
13)	View the model architecture in TensorBoard.

You should get an output like the following:
 
![](https://github.com/Neha-Chiluka/tf/blob/main/images/11.png?raw=true)

14)	Visualize the model-fitting process in TensorBoard. You should get the following output:
 
![](https://github.com/Neha-Chiluka/tf/blob/main/images/12.png?raw=true)
