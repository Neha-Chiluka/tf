# Chapter 6: LAB - 3: Predicting the radiator position of a space shuttle using random Search from Keras Tuner.



#### Task 1: Load the Dataset.

In this task, we load the shuttle dataset from the given URL using pandas. We also inspect the first few rows of the data and separate the target column (y) from the feature set (X).

```python
import pandas as pd

data_url = "https://raw.githubusercontent.com/fenago/tf/main/Chapter6-Regularization_and_Hyperparameter_Tuning/dataset/shuttle.trn"

data = pd.read_table(data_url, header=None, sep=' ')
data.head()

y = data.pop(9)
X = data.copy()

```

#### Task 2: Splitting Data into Training and Testing Sets.

In this task, we split the dataset into training and testing sets using the train_test_split function from sklearn. The stratify=y parameter ensures that the data is split in such a way that the class distribution in both the training and test sets remains the same.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)


```

#### Task 3: Install and Import Keras Tuner.

This task involves installing keras-tuner, which is used for hyperparameter tuning, and importing it into the script. This will allow us to search for the best set of hyperparameters for the neural network model.


```python
!pip install keras-tuner
import keras_tuner as kt

```

#### Task 4: Build the Model with Hyperparameter Search. 

In this task, we define a function model_builder that builds the neural network model. We include L2 regularization for each dense layer, and the regularization strength (l2) is selected from a list of possible values using Keras Tuner’s Choice function.

```python
def model_builder(hp):
    model = tf.keras.Sequential()
    hp_l2 = hp.Choice('l2', values = [0.1, 0.01, 0.001, 0.0001])
    reg_fc1 = Dense(512, input_shape=(9,), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=hp_l2))
    reg_fc2 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=hp_l2))
    reg_fc3 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=hp_l2))
    reg_fc4 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=hp_l2))
    reg_fc5 = Dense(8, activation='softmax')
    
    model.add(reg_fc1)
    model.add(reg_fc2)
    model.add(reg_fc3)
    model.add(reg_fc4)
    model.add(reg_fc5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


```

#### Task 5: Set Up Hyperparameter Search with Keras Tuner.

In this task, we use Keras Tuner to search for the best hyperparameters by using a random search strategy. The search will optimize the model based on validation accuracy, and we will set a maximum number of trials to 10.


```python
tuner = kt.RandomSearch(model_builder, objective='val_accuracy', max_trials=10)
tuner.search(X_train, y_train, validation_data=(X_test, y_test))
```

#### Task 6: Retrieve the Best Hyperparameters.

After the search is complete, the best hyperparameters are retrieved using get_best_hyperparameters(). These are the values that gave the best validation accuracy during the search.

```python
best_hps = tuner.get_best_hyperparameters()[0]
best_l2 = best_hps.get('l2')
best_l2

```

#### Task 7: Train the Model Using the Best Hyperparameters.

 In this task, we build a new model using the best hyperparameters selected from the search and train it on the training data for 5 epochs. The performance is evaluated on the validation set.

```python
model = tuner.hypermodel.build(best_hps)
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

```


With addition of L2 regularization, the model has almost similar accuracy scores between training and test sets. The model is not overfitting


