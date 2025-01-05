# Chapter 3: LAB - 3: Downloading a Model from TensorFlow Hub.

In this exercise, you will download a model from TensorFlow Hub and then view the architecture of the model in TensorBoard. The model that will be downloaded is the InceptionV3 model. This model was created in TensorFlow 1 and so requires some additional steps for displaying the model details as we're using TensorFlow 2. This model contains two parts: a part that includes convolutional layers to extract features from the images, and a classification part with fully connected layers. 
The distinct layers will be visible in TensorBoard as they have been named appropriately by the original author.


#### Task 1: Import Required Libraries.

Import TensorFlow, TensorFlow Hub, and TensorFlow's internal components for session management, logging, and operations.

```python
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.client import session
from tensorflow.python.summary import summary
from tensorflow.python.framework import ops


```

#### Task 2: Set Up TensorBoard Logging Directory

Specify the directory where TensorBoard logs will be saved for visualization.

```python
logdir = 'logs/'
```

#### Task 3: Load a Pretrained Model from TensorFlow Hub.  

Use TensorFlow Hub to load the Inception-ResNet-v2 classification model pretrained on ImageNet.


```python
module = hub.load('https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5')

```

#### Task 4: Access the Model's Default Signature. 

Retrieve the default signature from the loaded TensorFlow Hub module.

```python
model = module.signatures['serving_default']


```

#### Task 5: Create a New TensorFlow Session.

 Set up a new session to run operations and interact with the graph of the loaded model.


```python
with session.Session(graph=ops.Graph()) as sess:
    # Operations will be executed within this session


```

#### Task 6: Log the Model Graph to TensorBoard


Use TensorFlow's summary capabilities to log the model's computation graph for visualization in TensorBoard.

```python
    file_writer = summary.FileWriter(logdir)
    file_writer.add_graph(model.graph)


```

#### Task 7: Load TensorBoard Extension in Colab

Enable the TensorBoard extension in Google Colab to visualize the logged model graph..

```python
%load_ext tensorboard
%tensorboard --host 0.0.0.0 --logdir="logs"

```

The result in TensorBoard is the architecture of the InceptionV3 model. Here, you can view all the details about each layer of the model, including the input, output, and activation functions.
In this exercise, you successfully downloaded a model into a Jupyter notebook environment using the TensorFlow Hub library. Once the model was loaded into the environment, you visualized the architecture of the model using TensorBoard. This can be a helpful way to visualize your model's architecture for debugging purposes.
In this section, you have explored how to use TensorFlow Hub as a way to utilize the many brilliant models that have been created by experts in the machine learning field. As you will discover in later chapters, these models can be used to solve slightly different applications than those for which they were developed; this is known as transfer learning. In the next section, you will learn how to use Google Colab, an environment similar to Jupyter Notebooks that can be used to collaboratively develop applications in Python online, on Google servers.

