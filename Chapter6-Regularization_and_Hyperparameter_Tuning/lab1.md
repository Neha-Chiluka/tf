# Chapter 6: LAB - 1: Predicting the radiator position of a space shuttle using L2 Regularizer.

In this exercise, you will reuse the same logistic regression model as in Exercise 5.01, Building a Logistic Regression Model, and assess its performance by looking at different performance metrics: accuracy, precision, recall, and F1 score.

#### Task 1: Data Loading and Inspection.

The first step involves loading the dataset from the provided URL and displaying the first few rows to understand its structure. After that, check for missing values in the dataset.

```python
import pandas as pd

data_url = "https://raw.githubusercontent.com/fenago/tf/main/Chapter6-Regularization_and_Hyperparameter_Tuning/dataset/shuttle.trn"

data = pd.read_table(data_url, header=None, sep=' ')
data.head()
data.isnull().sum()

```

#### Task 2: Splitting Data into Training and Testing Sets.

Use train_test_split to divide the dataset into training and testing sets, with 30% of the data allocated for testing. Make sure the class distribution is preserved using the stratify=y parameter.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)

```

#### Task 3: Building a Neural Network Model without Regularization.  

Create a simple neural network with 4 hidden layers and an output layer with a softmax activation. Ensure that the model has the correct input shape, number of neurons, and activation functions.


```python
model = tf.keras.Sequential()
fc1 = Dense(512, input_shape=(9,), activation='relu')
fc2 = Dense(512, activation='relu')
fc3 = Dense(128, activation='relu')
fc4 = Dense(128, activation='relu')
fc5 = Dense(8, activation='softmax')
model.add(fc1)
model.add(fc2)
model.add(fc3)
model.add(fc4)
model.add(fc5)
model.summary()

```

#### Task 4: Compile and Train the Model. 

Compile the model with the Adam optimizer and sparse categorical cross-entropy loss. Then, train the model using the training data (X_train, y_train) for 5 epochs, and include validation during training by using validation_split=0.2.

```python
loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_split=0.2)


```

#### Task 5: Add L2 Regularization to the Model.

Modify the model to include L2 regularization (with l2=0.1) in the hidden layers. Use the kernel_regularizer argument to apply this regularization to the weights of the neurons.


```python
reg_fc1 = Dense(512, input_shape=(9,), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.1))
reg_fc2 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.1))
reg_fc3 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.1))
reg_fc4 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.1))
reg_fc5 = Dense(8, activation='softmax')


```

#### Task 6: Build a New Model with Regularization.

Create a new model using the regularized layers and ensure that the model architecture is correctly constructed. Make sure to add all the layers (including regularization) to the model.

```python
model2 = tf.keras.Sequential()
model2.add(reg_fc1)
model2.add(reg_fc2)
model2.add(reg_fc3)
model2.add(reg_fc4)
model2.add(reg_fc5)
model2.summary()

```

#### Task 7: Compile and Train the Regularized Model.

Compile the model with the same optimizer and loss function as before. Train the model for 5 epochs, using the same training data as before, and also include validation during training.

```python
optimizer2 = tf.keras.optimizers.Adam(0.001)
model2.compile(optimizer=optimizer2, loss=loss, metrics=['accuracy'])
model2.fit(X_train, y_train, epochs=5, validation_split=0.2)

```



With addition of L2 regularization, the model has almost similar accuracy scores between training and test sets. The model is not overfitting


