# Chapter 4: LAB - 3: Creating a Multi-Layer ANN with TensorFlow.

In this exercise, you will create a multi-layer ANN using TensorFlow. This model will have four hidden layers. You will add multiple layers to the model and activation functions to the output of the layers. The first hidden layer will have 16 units, the second will have 8 units, and the third will have 4 units. The output layer will have 2 units. You will utilize the same dataset as in Exercise 4.02, Creating a Linear Regression Model as an ANN with TensorFlow, which describes the bias correction of air temperature forecasts for Seoul, South Korea. The exercise aims to predict the next maximum and minimum temperature given measurements of the prior timepoints and attributes of the weather station.


#### Task 1: Import Libraries.

This task imports the TensorFlow and Pandas libraries. TensorFlow is used for building and training machine learning models, while Pandas is used for handling and manipulating the dataset.

```python
import tensorflow as tf
import pandas as pd

```

#### Task 2: Load Dataset.

Here, a weather dataset is loaded from a specified URL using Pandas. The low_memory=False argument ensures efficient memory usage for large datasets.

```python
url = "https://raw.githubusercontent.com/fenago/tf/main/Chapter4-Regression_and_Classification_Models/dataset/Summary_of_Weather.csv"
df = pd.read_csv(url, low_memory=False)

```

#### Task 3: Data Preprocessing - Drop Columns.  

Unnecessary columns, STA and Date, are dropped. Rows containing missing values are also removed to clean the dataset and prepare it for further processing.


```python
df.drop(['STA','Date'], inplace=True, axis=1)
df.dropna(inplace=True)
```

#### Task 4: Feature and Target Selection. 

The dataset is divided into two parts:

target: Contains the columns MaxTemp and MinTemp, which represent the values to be predicted.

features: Contains the remaining columns used as input for the prediction.

```python
target = df[['MaxTemp', 'MinTemp']]
features = df.drop(['MaxTemp', 'MinTemp'], axis=1)
```

#### Task 5: Feature Scaling.

The features are normalized to a range between 0 and 1 using the MinMaxScaler from Scikit-learn. This scaling helps the model train efficiently by ensuring all inputs are on the same scale.


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_array = scaler.fit_transform(features)
features = pd.DataFrame(feature_array, columns=features.columns)

```

#### Task 6: Build the Neural Network.

A sequential neural network model is created with the following layers:

Input Layer: Takes the number of features as input.
Dense Layer 1: Fully connected layer with 16 neurons.
Dense Layer 2: Fully connected layer with 8 neurons.
Dense Layer 3: Fully connected layer with 4 neurons.
Output Layer: Fully connected layer with 2 neurons, corresponding to the two target values (MaxTemp and MinTemp).

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(features.shape[1],), name='Input_layer'))
model.add(tf.keras.layers.Dense(16, name='Dense_layer_1'))
model.add(tf.keras.layers.Dense(8, name='Dense_layer_2'))
model.add(tf.keras.layers.Dense(4, name='Dense_layer_3'))
model.add(tf.keras.layers.Dense(2, name='Output_layer'))

```

#### Task 7: Compile the Model.

The model is compiled with the following configurations:

Optimizer: RMSprop with a learning rate of 0.001.
Loss Function: Mean Squared Error (MSE), suitable for regression tasks.

```python
model.compile(tf.optimizers.RMSprop(0.001), loss='mse')
```

#### Task 8: Define Callback for TensorBoard.

A TensorBoard callback is defined to monitor and visualize training metrics during and after the training process.
 
 ```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs1")

```

#### Task 9: Train the Model.
The model is trained for 50 epochs using the normalized features as input and the target values as output. TensorBoard is used to log training metrics.

```python
model.fit(x=features.to_numpy(), y=target.to_numpy(), epochs=50, callbacks=[tensorboard_callback])

```
#### Task 10: Evaluate the Model. 

The trained model is evaluated on the same dataset, and the Mean Squared Error (MSE) loss is printed.

```python
loss = model.evaluate(features.to_numpy(), target.to_numpy())
print('loss:', loss)


```

#### Task 11: Launch TensorBoard

TensorBoard is launched to visualize the training metrics, including loss curves and layer details, providing insights into the model's performance

```python
%load_ext tensorboard
%tensorboard --host 0.0.0.0 --logdir="logs1"

```
In this exercise, you have created an ANN with multiple hidden layers. The loss you obtained was lower than that achieved using linear regression, which demonstrates the power of ANNs. With some tuning to the hyperparameters (such as varying the number of layers, the number of units within each layer, adding activation functions, and changing the loss and optimizer), the loss could be even lower. In the next activity, you will put your model-building skills into action on a new dataset.
