# Chapter 7: LAB - 3: Building a CNN.

Now, you'll take a brief look at image classification with the MNIST dataset. The dataset consists of 10 classes with a training set of 60,000 28x28 grayscale images and 10,000 test images.

#### Task 1: Import the libraries.

This imports libraries required for data processing (tensorflow_datasets), model building (keras, layers), and visualization (matplotlib). These tools enable tasks like loading datasets, building neural networks, and plotting results.

```python
import os
import matplotlib.pyplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, Activation, Rescaling

```

#### Task 2: Check for GPU Availability.

This checks if a GPU is available for TensorFlow to use. If a GPU is found, memory growth is enabled to prevent TensorFlow from allocating all GPU memory upfront. If no GPU is found, TensorFlow will use the CPU. 

```python
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPUs found. TensorFlow will use CPU.")

```

#### Task 3:  Load MNIST Dataset.  

This loads the MNIST dataset for training and testing, returning the data in a tuple format (image, label) and dataset metadata (ds_info). The dataset is shuffled and split into training and test sets.


```python
(ds_train, ds_test), ds_info = tfds.load("mnist", split=["train", "test"], as_supervised=True, with_info=True)

```

#### Task 4: Display Dataset Information. 

This retrieves and prints the shape of the images, the number of classes (10 for MNIST digits), and the names of these classes (0-9). This is useful to understand the dataset structure.

```python
image_shape = ds_info.features["image"].shape
num_classes = ds_info.features["label"].num_classes
names_of_classes = ds_info.features["label"].names


```

#### Task 5: Normalize Images.

This function normalizes the images by casting them to float32 and scaling pixel values to a [0, 1] range by dividing by 255. This is a common preprocessing step to improve model training.


```python
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

```

#### Task 6: Prepare Dataset for Training.

This pipeline normalizes, caches, shuffles, batches, and prefetches the training dataset for efficient training. These steps help in improving data loading and model performance.

```python
ds_train = ds_train.map(normalize_img).cache().shuffle(ds_info.splits["train"].num_examples).batch(BATCH_SIZE).prefetch(AUTOTUNE)

```

#### Task 7: Create the Model

This defines a convolutional neural network (CNN) with two Conv2D layers, dropout for regularization, and a fully connected Dense layer to output predictions. The model ends with a softmax activation for multi-class classification.

```python
model = keras.Sequential([ 
    keras.Input((28, 28, 1)), 
    layers.Conv2D(32, 3, activation="relu", strides=2),
    layers.Conv2D(64, 3, activation="relu", strides=2),
    layers.Flatten(),
    Dropout(rate=0.2),
    layers.Dense(512, activation="relu"),
    Dropout(rate=0.2),
    tf.keras.layers.Dense(10, activation="softmax"),
])


```

#### Task 8: Compile the Model

This compiles the model with the Adam optimizer, sparse categorical crossentropy loss (for multi-class classification), and accuracy as the metric. These are standard choices for classification tasks.

```python
model.compile(optimizer=keras.optimizers.Adam(0.001), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

```

#### Task 9: Train the Model.

 This trains the model using the training dataset (ds_train) and validates it using the test dataset (ds_test). The model runs for 15 epochs and outputs training progress for each epoch.
 
 ```python
history = model.fit(ds_train, validation_data=ds_test, epochs=15, verbose=2)

```

#### Task 10: Plot Training and Validation Accuracy.

This visualizes the training and validation accuracy over epochs to assess the model’s learning progress. The plot_trend_by_epoch function plots accuracy for both training and validation datasets.

```python
plot_trend_by_epoch(tr_accuracy, val_accuracy, "Accuracy")

```

#### Task 11: Plot Training and Validation Loss

This plots the loss values for both the training and validation datasets over epochs. Monitoring loss helps track how well the model minimizes error during training.

`plot_trend_by_epoch(tr_loss, val_loss, "Loss")
`

As you can see from the accuracy and loss curves as a function of epochs, the accuracy increases, and loss decreases. On the validation set, both to plateau, which is good signal to stop training to prevent overfitting to the training datast.

Now, that you have completed this chapter, it’s time to put everything that you’ve learned to the test with Activity 7.01, Building a CNN with More ANN Layers, where you’ll be building a CNN with additional ANN layers.

