# Chapter 4: LAB - 2: Creating a Linear Regression Model as an ANN with TensorFlow.

In this exercise, you will create a linear regression model as an ANN using TensorFlow. The dataset, Summary of Weather.csv, describes the temperature at the time of world-war-II. The fields represent temperature measurements of the given data, the weather station at which the metrics were measured, model forecasts of weather-related metrics such as snowfall, humidity, wind etc. You are required to predict the next maximum and minimum temperature given measurements of the prior timepoints and attributes of the weather station.


#### Task 1: Task 1: Load the Weather Dataset.

Import the weather dataset from a remote URL and load it into a pandas DataFrame for preprocessing.

```python
import pandas as pd

url = "https://raw.githubusercontent.com/fenago/tf/main/Chapter4-Regression_and_Classification_Models/dataset/Summary_of_Weather.csv"
df = pd.read_csv(url, low_memory=False)

```

#### Task 2:Clean and Preprocess the Data.

Drop irrelevant columns and handle missing values by removing rows with null entries.

```python
df.drop(['STA', 'Date'], inplace=True, axis=1)
df.dropna(inplace=True)


```

#### Task 3: Split Features and Target Variables.  

Separate the dataset into feature variables (features) and target variables (target) for model training.


```python
target = df[['MaxTemp', 'MinTemp']]
features = df.drop(['MaxTemp', 'MinTemp'], axis=1)


```

#### Task 4: Scale the Feature Variables. 

Normalize the feature variables to the range [0, 1] using MinMaxScaler for better model performance.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
feature_array = scaler.fit_transform(features)
features = pd.DataFrame(feature_array, columns=features.columns)


```

#### Task 5: Create the Neural Network Model.

Build a simple neural network with an input layer and a dense output layer for regression.


```python
import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(features.shape[1],), name='Input_layer'))
model.add(tf.keras.layers.Dense(2, name='Output_layer'))


```

#### Task 6: Compile the Model.

Configure the model with the RMSprop optimizer and Mean Squared Error (MSE) as the loss function.

```python
model.compile(tf.optimizers.RMSprop(0.001), loss='mse')


```

#### Task 7: Set Up TensorBoard Logging.

Initialize a TensorBoard callback to monitor the training progress.

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")


```

#### Task 8: Train the Model.

Train the model for 50 epochs using the preprocessed feature and target variables.
 
 ```python
model.fit(
    x=features.to_numpy(),
    y=target.to_numpy(),
    epochs=50,
    callbacks=[tensorboard_callback]
)


```

#### Task 9: Evaluate the Model.
Assess the model's performance on the same data used for training by calculating the loss

```python
loss = model.evaluate(features.to_numpy(), target.to_numpy())
print('loss:', loss)

```
#### Task 10: Visualize Training with TensorBoard

Launch TensorBoard to visualize the model's training metrics like loss

```python
%load_ext tensorboard
%tensorboard --host 0.0.0.0 --logdir="logs"

```

In this exercise, you have learned how to create, train, and evaluate an ANN with TensorFlow by using Keras layers. You recreated the linear regression algorithm by creating an ANN with an input layer and an output layer that has one unit for each output. Here, there were two outputs representing the maximum and minimum values of the temperature; thus, the output layer has two units.
