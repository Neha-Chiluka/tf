# Chapter 4: LAB - 1: Creating an ANN with TensorFlow.

In this exercise, you will create your first sequential ANN in TensorFlow. You will have an input layer, a hidden layer with four units and a ReLU activation function, and an output layer with one unit. Then, you will create some simulation data by generating random numbers and passing it through the model, using the model's predict method to simulate a prediction for each data example.


#### Task 1: Import TensorFlow Library.

Import TensorFlow to create and train machine learning models.

```python
import tensorflow as tf
```

#### Task 2:Initialize a Sequential Model.

Create a sequential model to stack layers in a linear fashion.

```python
model = tf.keras.Sequential()

```

#### Task 3: Add an Input Layer.  

Define the input shape for the model using an InputLayer as the first layer..


```python
model.add(tf.keras.layers.InputLayer(input_shape=(10,), name='Input_layer_1'))


```

#### Task 4: Add Dense Layers to the Model. 

Add a fully connected Dense layer with 32 neurons and ReLU activation, followed by another Dense layer with 10 neurons and Softmax activation for output.

```python
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

```

#### Task 5: Display Model Variables.

List the model's trainable variables, including weights and biases for each layer.


```python
model.variables

```

#### Task 6: Create Random Input Data.

Generate random input data with a shape of (40, 10) to simulate training or inference inputs.

```python
data = tf.random.normal((40, 10))

```

#### Task 7: Inspect the Random Data

Display the generated random input data for verification.

```python
data

```

#### Task 8: Make Predictions Using the Model

 Use the predict method to generate predictions for the given input data.
 
 ```python
prediction = model.predict(data)
prediction

```

Calling the predict() method on the sample data will propagate the data through the network. In each layer, there will be a matrix multiplication of the data with the weights, and the bias will be added before the data is passed as input data to the next layer. This process continues until the final output layer.
In this exercise, you created a sequential model with multiple layers. You initialized a model, added an input layer to accept data with eight features, added a hidden layer with four units, and added an output layer with one unit. Before fitting a model to training data, you must first compile the model with an optimizer and choose a loss function to minimize the value it computes by updating weights in the training process. 



