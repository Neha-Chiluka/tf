# Chapter 4: LAB - 4: Creating a Logistic Regression Model as an ANN with TensorFlow.

This dataset was generated for use on 'Prediction of Motor Failure Time Using An Artificial Neural Network' project. A cooler fan with weights on its blades was used to generate vibrations. To this fan cooler was attached an accelerometer to collect the vibration data. With this data, motor failure time predictions were made, using an artificial neural networks. To generate three distinct vibration scenarios, the weights were distributed in two different ways: 1) 'red' - normal configuration: two weight pieces positioned on neighboring blades; angle; 2) 'green' - opposite configuration: two weight pieces positioned on opposite blades.


#### Task 1: Import Libraries and Load Dataset.

Import required libraries (tensorflow for neural networks and pandas for data manipulation).

Load the dataset from a remote URL using pd.read_csv. The low_memory=False ensures efficient loading for large datasets..

```python
import tensorflow as tf
import pandas as pd
url = "https://raw.githubusercontent.com/fenago/tf/main/Chapter4-Regression_and_Classification_Models/dataset/accelerometer.csv"
df = pd.read_csv(url, low_memory=False)

```

#### Task 2: Clean the Dataset.

Removes rows with missing values (NaN) to ensure clean and complete data for training.
Uses inplace=True to directly modify the dataframe without creating a copy.

```python
df.dropna(inplace=True)
```

#### Task 3: Split Dataset into Features and Target.  

Assign the column wconfid as the target variable (y).

Assign the remaining columns as features (X) by dropping the wconfid column.


```python
target = df['wconfid']
features = df.drop('wconfid', axis=1)

```

#### Task 4: Build a Neural Network Model. 

nitialize a Sequential neural network model.
Add an input layer with shape corresponding to the number of features in the dataset.
Add an output layer with a single neuron and a sigmoid activation function, suitable for binary classification tasks.

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(features.shape[1],), name='Input_layer'))
model.add(tf.keras.layers.Dense(1, name='Output_layer', activation='sigmoid'))

```

#### Task 5: Compile the Model.

Configure the model for training using the RMSprop optimizer with a learning rate of 0.0001.
Use binary_crossentropy as the loss function, appropriate for binary classification.
Track accuracy as the evaluation metric during training.


```python
model.compile(tf.optimizers.RMSprop(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
```

#### Task 6: Train the Model with TensorBoard Callback.

Add a TensorBoard callback to log training and validation metrics for visualization.
Train the model using features and target as input and output data, respectively.
Set the number of epochs to 50 and split 20% of the data for validation during training.

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs4")
model.fit(x=features.to_numpy(), y=target.to_numpy(), epochs=50, callbacks=[tensorboard_callback], validation_split=0.2)


```

#### Task 7: Evaluate the Model.

Evaluate the model on the entire dataset to compute accuracy and loss.
Print the evaluation results.

```python
loss = model.evaluate(features.to_numpy(), target.to_numpy())
print('loss:', loss)

```

#### Task 8: Visualize Training with TensorBoard.

Load the TensorBoard extension for Jupyter Notebook.

Launch TensorBoard to visualize training and validation metrics, including loss and accuracy trends.
 
 ```python
%load_ext tensorboard
%tensorboard --host 0.0.0.0 --logdir="logs4"


```

In this exercise, you have learned how to build a classification model to discriminate between the binding properties of various molecules based on their other molecular attributes. The classification model was equivalent to a logistic regression model since it had only one layer and was preceded by a sigmoid activation function. With only one layer, there is a weight for each input feature and a single value for the bias. The sigmoid activation function transforms the output of the layer into a value between 0 and 1, which is then rounded to represent your two classes. 0.5 and above represents one class, the molecule with binding properties, and below 0.5 represents the other class, molecules with non-binding properties.
