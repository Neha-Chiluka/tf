# Chapter 7: LAB - 1: Creating the First Layer to Build a CNN.

As a TensorFlow freelancer, you've been asked to show your potential employer a few lines of code that demonstrate how you might build the first layer in a CNN. They ask that you keep it simple but provide the first few steps to create a CNN layer. In this exercise, you will complete the first step in creating a CNN—that is, adding the first convolutional layer.

#### Task 1: Import the libraries.

The goal is to load the dataset, inspect it, and check the distribution of the target classes. 

```python
import tensorflow as tf
from tensorflow.keras import models, layers
```

#### Task 2: Check TensorFlow Version.

This task checks the version of TensorFlow installed in the environment. The tf.__version__ command returns the TensorFlow version. In the example output, TensorFlow 2.17.1 is printed. This step is essential to ensure the environment is set up with the correct version of TensorFlow, which can affect the code execution and compatibility of features. 

```python
print(tf.__version__)
```

#### Task 3: Define Image Shape.  

This task defines the shape of the input image for the model. Here, image_shape = (300, 300, 3) specifies that the input images are 300x300 pixels in size, with 3 color channels (representing RGB). This shape is used later in the code to define the model input dimensions.


```python
image_shape = (300, 300, 3)
```

#### Task 4: Create a Sequential Model with a Convolutional Layer. 

This task involves creating a Sequential model in TensorFlow using Keras. The model starts with a 2D convolutional layer (Conv2D), followed by an activation layer (Activation). The Conv2D layer has 16 filters with a kernel size of (3,3) and expects the input shape to be (300, 300, 3) (as defined earlier). The activation function applied is ReLU, which is commonly used in convolutional layers to introduce non-linearity and help the model learn complex patterns

```python
our_first_layer = models.Sequential([
    layers.Conv2D(filters=16, kernel_size=(3,3), input_shape=image_shape),
    layers.Activation('relu')
])

```

Simple enough. You have just taken the first steps in creating your first CNN
You will now move on to the type of layer that usually follows a convolutional layer-the pooling layer.

