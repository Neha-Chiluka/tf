# Chapter 6: LAB - 3: Predicting the radiator position of a space shuttle using Hyperband from Keras Tuner.



#### Task 1: Data Loading and Preprocessing.

In this task, the dataset is loaded from a URL, and the features (X) and target (y) are separated. Then, the data is split into training and testing sets. The train_test_split function is used to ensure that the data is evenly distributed between the training and testing datasets, with 30% of the data used for testing and the remaining 70% for training.

```python
import pandas as pd

data_url = "https://raw.githubusercontent.com/fenago/tf/main/Chapter6-Regularization_and_Hyperparameter_Tuning/dataset/shuttle.trn"
data = pd.read_table(data_url, header=None, sep=' ')
y = data.pop(9)
X = data.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)

```

#### Task 2: Hyperparameter Tuning Setup.

This task sets up the hyperparameter tuning using the Keras Tuner library. The model_builder function defines a model where the number of units in the first layer (hp_units) and learning rate (hp_learning_rate) are tuned using Hyperband. The model consists of multiple Dense layers, with L2 regularization applied.

```python
import keras_tuner as kt

def model_builder(hp):
    model = tf.keras.Sequential()
    hp_units = hp.Int('units', min_value=128, max_value=512, step=64)
    reg_fc1 = Dense(hp_units, input_shape=(9,), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001))
    reg_fc2 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001))
    reg_fc3 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001))
    reg_fc4 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001))
    reg_fc5 = Dense(8, activation='softmax')
    model.add(reg_fc1)
    model.add(reg_fc2)
    model.add(reg_fc3)
    model.add(reg_fc4)
    model.add(reg_fc5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
    optimizer = tf.keras.optimizers.Adam(hp_learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=5, overwrite=True)



```

#### Task 3: Hyperparameter Tuning Execution.

In this task, the Hyperband tuner searches for the best hyperparameters by training multiple models with different configurations of hyperparameters (units and learning_rate). It uses validation accuracy (val_accuracy) as the evaluation metric. After tuning, the best hyperparameters are selected.


```python
tuner.search(X_train, y_train, validation_data=(X_test, y_test))

```

#### Task 4: Extracting Best Hyperparameters.

After the tuning process is complete, you extract the best hyperparameters that gave the highest validation accuracy. The selected hyperparameters are then used to build a model.

```python
best_hps = tuner.get_best_hyperparameters()[0]
best_units = best_hps.get('units')
best_lr = best_hps.get('learning_rate')


```

#### Task 5: Model Training with Best Hyperparameters.

After identifying the best hyperparameters, you rebuild the model using the best_hps and train it for a fixed number of epochs. This training is done using the optimal model configuration to see how well it performs on the test data.


```python
model = tuner.hypermodel.build(best_hps)
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

```


