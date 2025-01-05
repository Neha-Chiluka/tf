# Chapter 3: LAB - 2: Using TensorBoard to Visualize Matrix Multiplication.

In this exercise, you will load a dataset from google drive that has temperature during world war II dataset.
Note
You can find the Summary of Weather.csv file here:
To perform the exercise, you will have to navigate to https://colab.research.google.com/ and create a new notebook to work in. You will need to connect to a GPU-enabled environment to speed up TensorFlow operations such as tensor multiplication. Once the data has been loaded into the development environment, you will view the first five rows. Next, you'll drop the Date field since matrix multiplication requires numerical fields. Then, you will perform tensor multiplication of the dataset with a tensor or uniformly random variables.

#### Task 1: Upload Dataset to Google Colab

Explanation: Use the files module to upload a zipped dataset to your Colab environment.

```python
from google.colab import files
files.upload()  # Prompt to upload a file

```

#### Task 2: Unzip the Dataset

Extract the uploaded dataset file to the specified directory.

```python
!unzip -q /content/dataset.zip -d /content/

```

#### Task 3: Verify Dataset Extraction.  

Check the contents of the directory to ensure the dataset is extracted correctly..


```python
!ls /content/
```

#### Task 4: Import TensorFlow and Setup Image Augmentation. 

Import TensorFlow and configure the ImageDataGenerator for image rescaling.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

```

#### Task 5: Configure TensorBoard Logging.

Set up a TensorBoard writer to log training data for visualization.


```python
logdir = 'logs/'
writer = tf.summary.create_file_writer(logdir)

```

#### Task 6: Load the Dataset with ImageDataGenerator

Use flow_from_directory to load images, rescale them, and prepare for binary classification.

```python
batch_size = 3
training_set = train_datagen.flow_from_directory(
    'dataset/image_data',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

```

#### Task 7: Log Training Data to TensorBoard

Write sample training images to TensorBoard for visualization.

```python
with writer.as_default():
    tf.summary.image(
        "Training data",
        next(training_set)[0],
        max_outputs=batch_size,
        step=0
    )
```

#### Task 8: Load TensorBoard Extension

Enable the TensorBoard extension in Colab for visualization of logs

```python
%load_ext tensorboard
%tensorboard --host 0.0.0.0 --logdir="logs"

```

The result in TensorBoard is the images from the first batch. You can see that they are images of cats and dogs. TensorBoard also provides you with the ability to adjust the brightness and contrast of the images; however, that affects only the images in TensorBoard and not the underlying image data.
In this exercise, you viewed a batch of images from an image data generator using TensorBoard. This is an excellent way to verify the quality of your training data. It may not be necessary to verify every image for quality, but sample batches can be viewed easily using TensorBoard.


