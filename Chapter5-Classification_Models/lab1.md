# Chapter 5: LAB - 1: Building a Logistic Regression Model.

In this exercise, you will build and train a logistic regression model in TensorFlow that will predict whether the signal shows the presence of some object, or just empty air.
You will be working on The Ionosphere dataset contains features obtained from radar signals focused on the ionosphere layer of the Earth's atmosphere.

#### Task 1: Load and Prepare the Dataset

Import a dataset from a URL, clean it by removing unnecessary columns, and convert categorical labels to numerical format. 

```python
import pandas as pd

# Load data from URL
data_url = "https://raw.githubusercontent.com/fenago/tf/main/Chapter5-Classification_Models/datasets/ion.csv"
data = pd.read_csv(data_url)

# Remove the unnamed column
data.pop('Unnamed: 0')

# Convert categorical labels to numerical
reduce_map = {"Class": {"good": 1, "bad": 0}}
data.replace(reduce_map, inplace=True)

# Separate features and target
y = data.pop('Class')
X = data
```

#### Task 2: Split the Data into Training and Test Sets.

Split the dataset into 70% training data and 30% test data for model evaluation.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Check dimensions
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
```

#### Task 3: Define a Neural Network Architecture.  

Create a sequential neural network model with dense layers and ReLU activations for the hidden layers and a sigmoid activation for the output.


```python
import tensorflow as tf
from tensorflow.keras.layers import Dense

tf.random.set_seed(8)

# Define model architecture
model = tf.keras.Sequential([
    Dense(512, input_shape=(34,), activation='relu'),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

```

#### Task 4: Compile the Model. 

Compile the model with a binary cross-entropy loss function and Adam optimizer.

```python
loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.001)

model.compile(optimizer=optimizer, loss=loss)

```

#### Task 5: Train the Model.

Train the model on the training dataset for 5 epochs and observe the loss reduction.


```python
model.fit(X_train, y_train, epochs=5)
```

#### Task 6: Evaluate the Model.

Predict on the test set and compare predictions with actual values.

```python
# Make predictions
preds = model.predict(X_test)

# Display first 5 predictions and actual values
print(f"Predictions (first 5):\n{preds[:5]}")
print(f"Actual values (first 5):\n{y_test[:5].values}")


```

#### Task 7: Save the Model.

Save the trained model for future use.

```python
model.save("Chapter5_lab_01_model.h5")
```


