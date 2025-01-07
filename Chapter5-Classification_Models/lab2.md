# Chapter 5: LAB - 2: Classification Evaluation Metrics.

In this exercise, you will reuse the same logistic regression model as in Exercise 5.01, Building a Logistic Regression Model, and assess its performance by looking at different performance metrics: accuracy, precision, recall, and F1 score.

#### Task 1: Load and Preprocess the Data.

Load the dataset from the URL, clean it, and preprocess it. Replace string class labels with numeric ones for modeling purposes. 

```python
import pandas as pd

# Load data
data_url = "https://raw.githubusercontent.com/fenago/tf/main/Chapter5-Classification_Models/datasets/ion.csv"
data = pd.read_csv(data_url)

# Remove unnecessary column
data.pop('Unnamed: 0')

# Replace string labels with numeric ones
reduce_map = {"Class": {"good": 1, "bad": 0}}
data.replace(reduce_map, inplace=True)
```

#### Task 2: Split Data into Features and Target.

Separate features (X) and the target variable (y). Then, split the data into training and testing sets.

```python
from sklearn.model_selection import train_test_split

# Separate features and target
y = data.pop('Class')
X = data

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Print shapes of the splits
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

```

#### Task 3: Load a Pre-trained Model.  

Load a pre-trained model and inspect its architecture.


```python
import tensorflow as tf

# Load pre-trained model
model = tf.keras.models.load_model("/content/Chapter5_lab_01_model.h5")

# Model summary
model.summary()
```

#### Task 4: Make Predictions. 

Use the model to make probability predictions on the test set. Convert probabilities to binary class predictions using a threshold.

```python
# Make probability predictions
preds_proba = model.predict(X_test)

# Convert probabilities to binary predictions
preds = preds_proba >= 0.5
print(preds[:5])

```

#### Task 5: Evaluate the Model.

Calculate performance metrics: Accuracy, Precision, Recall, and F1-score.


```python
from tensorflow.keras.metrics import Accuracy, Precision, Recall

# Initialize metrics
acc = Accuracy()
prec = Precision()
rec = Recall()

# Update metrics with predictions and true labels
acc.update_state(preds, y_test)
prec.update_state(preds, y_test)
rec.update_state(preds, y_test)

# Calculate metrics
acc_results = acc.result().numpy()
prec_results = prec.result().numpy()
rec_results = rec.result().numpy()

# Calculate F1-score
f1 = 2 * (prec_results * rec_results) / (prec_results + rec_results)

# Print results
print(f"Accuracy: {acc_results:.4f}")
print(f"Precision: {prec_results:.4f}")
print(f"Recall: {rec_results:.4f}")
print(f"F1-score: {f1:.4f}")

```

Overall, the model has achieved excellent score close to 0.98 for all four different metrics: accuracy, precision, recall and F1 score. So, this model makes almost correct predictions.

