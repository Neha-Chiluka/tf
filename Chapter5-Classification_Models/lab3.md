# Chapter 5: LAB - 3: Building a Multi-Class Model.

In this exercise, you will build and train a multi-class classifier in TensorFLow that will to correctly classify the type of surface defects in stainless steel plates, with six types of possible defects (plus “other”). The input vector was made up of 27 indicators that describe the geometric shape of the defect and its outline.

#### Task 1: Load and Understand the Dataset.

The goal is to load the dataset, inspect it, and check the distribution of the target classes. 

```python
import pandas as pd

# Load dataset
data_url = "https://github.com/fenago/tf/raw/main/Chapter5-Classification_Models/datasets/faults.csv"
data = pd.read_csv(data_url, sep=",")
print(data.head())

# Check the distribution of the target column
print(data['target'].value_counts())

```

#### Task 2: Preprocess the Dataset.

Map the target labels to numeric values, handle missing values, and split the dataset into features (X) and target (y).

```python
# Map target labels to numeric values
reduce_map = {"target": {'Pastry': 0, 'Z_Scratch': 1, 'K_Scatch': 2, 'Stains': 3, 'Dirtiness': 4, 'Bumps': 5, 'Other_Faults': 6}}
data.replace(reduce_map, inplace=True)

# Drop missing values
data.dropna(inplace=True)

# Split into features (X) and target (y)
y = data.pop('target')
X = data


```

#### Task 3: Split Data into Training and Testing Sets.  

Split the data into training and testing sets, ensuring class distribution is maintained using stratified sampling.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

```

#### Task 4: Build a Neural Network Model. 

Construct a sequential neural network model with dense layers for classification.

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense

tf.random.set_seed(8)

# Define the model
model = tf.keras.Sequential([
    Dense(512, input_shape=(27,), activation='relu'),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')  # Output layer with 7 classes
])

# Summary of the model
model.summary()

```

#### Task 5: Compile the Model.

Specify the loss function, optimizer, and evaluation metrics for training.


```python
loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.001)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

```

#### Task 6: Train the Model

Train the model on the training data for a specified number of epochs.

```python
model.fit(X_train, y_train, epochs=50)

```

#### Task 7: Evaluate the Model

Evaluate the model's performance on the test set to check its accuracy and loss

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

```



In this exercise, you learned how to build and train a multi-class classifier to 	predict an outcome composed of eight different classes. Your model achieved an    	accuracy score close to 0.45 on both the training and test sets, which is not up to the 	mark. This implies that your model predicts half of the observations correct and rest 	as incorrect. 
