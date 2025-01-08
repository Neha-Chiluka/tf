# Chapter 7: LAB - 2: Creating a Pooling Layer for a CNN.

In this exercise, You receive an email from your potential employer for the TensorFlow freelancing job that you applied for in Exercise 7.01, Creating the First Layer to Build a CNN. The email asks whether you can show how you would code a pooling layer for a CNN. In this exercise, you will build your base model by adding a pooling layer, as requested by your potential employer.

#### Task 1: Import Necessary Modules from TensorFlow.

import tensorflow as tf: This line imports the TensorFlow library as tf, allowing access to its functions and modules. TensorFlow is an open-source framework used for machine learning and deep learning.
from tensorflow.keras import models, layers: This imports the models and layers modules from Keras (a high-level API in TensorFlow). The models module is used to define and train neural network models, and the layers module contains various building blocks for constructing the layers of a neural network.

```python
import tensorflow as tf
from tensorflow.keras import models, layers
```

#### Task 2: Define Image Shape for the Model.

This line defines the shape of the input image that the model will process. image_shape = (300, 300, 3) means that each input image will have a size of 300x300 pixels with 3 color channels (RGB). This shape will be used to configure the input layer of the model. 

```python
image_shape = (300, 300, 3)
```

#### Task 3: Create a Sequential Model with Conv2D and ReLU Activation Layers.  

layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=image_shape): This adds a 2D convolutional layer to the model with 16 filters, each having a size of 3x3 pixels. The input_shape=image_shape specifies the expected shape of the input data as (300, 300, 3). This layer extracts features from the input image.
layers.Activation('relu'): This applies the ReLU (Rectified Linear Unit) activation function, which introduces non-line


```python
our_first_model = models.Sequential([
    layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=image_shape),
    layers.Activation('relu')
])

```

In this model, you have created a CNN with a convolutional layer followed by a ReLU activation function then a max pooling layer. The models take images of size 300x300 with three color channels.

Now that you have successfully added a MaxPool2D layer to your CNN, the next step is to add a flattening layer so that your model can use all the data.

