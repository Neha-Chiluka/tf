# Chapter 2: LAB - 3: Loading Image Data for Batch Processing

In this exercise, you'll learn how to load in image data for batch processing. The image_data folder contains a set of images of cats and dogs. You will load the images of cats and dogs for batch processing and rescale them so that the image values range between 0 and 1. You are then tasked with printing the labeled images of a batch from the data generator.

#### Task 1 - Open Google Collab

1. Open Google Colab:

- Go to Google Colab.
- If prompted, sign in with your Google account.

2. Create a New Notebook:
- Once you're logged in, click on File in the top left corner.
- Select New Notebook.

#### Task 2 - Prepare the Environment and Data

Unzip the provided dataset (dataset.zip) to access the image data folder (image_data). This step ensures the data is available for processing.

```python
from google.colab import files

# Step 2: Unzip the file (replace 'dataset.zip' with the actual file name)
!unzip -q /content/dataset.zip -d /content/

# Step 3: Verify the extracted content
!ls /content/


```

#### Task 3: Import Required Libraries

Import the necessary library (ImageDataGenerator) from tensorflow.keras.preprocessing.image to preprocess and generate image data for training.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

```

#### Task 4: Initialize the ImageDataGenerator.

Instantiate the ImageDataGenerator class with the rescale argument set to 1./255 to normalize the image pixel values to the range [0, 1]. This normalization aids in model training and convergence.


```python
train_datagen = ImageDataGenerator(rescale=1./255)
```

#### Task 5: Create a Data Generator Using flow_from_directory


- Use the flow_from_directory method to configure the data generator to read images from the image_data folder.
- Set the target image size to (64, 64) for uniformity.
- Specify a batch size of 25 to process images in batches during training.
- Use class_mode='binary' since the task involves binary classification (cats vs. dogs).

```python
training_set = train_datagen.flow_from_directory(
    'dataset/image_data/',
    target_size=(64, 64),
    batch_size=25,
    class_mode='binary'
)

```

#### Task 6: Define a Function to Display Images and Labels

- Create a function to display the first 15 images in a 5x5 grid from a batch, along with their labels.
- Use the training_set.class_indices dictionary to map numeric labels back to class names (e.g., 0 -> 'cat', 1 -> 'dog').
- Visualize the images using matplotlib.pyplot.

```python
import matplotlib.pyplot as plt

def show_batch(image_batch, label_batch):
    # Map numeric labels to class names
    lookup = {v: k for k, v in training_set.class_indices.items()}
    label_batch = [lookup[label] for label in label_batch]
    
    # Plot images in a 5x5 grid
    plt.figure(figsize=(10, 10))
    for n in range(15):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n].title())
        plt.axis('off')


```

#### Task 7:  Extract a Batch and Display the Images

- Retrieve a batch of images and labels using the next() function on the data generator.
- Pass the extracted batch to the show_batch function to visualize the images and their corresponding labels.

```python
# Retrieve a batch of images and labels
image_batch, label_batch = next(training_set)

# Display the batch using the defined function
show_batch(image_batch, label_batch)

```
Here, you can see the output of a batch of images of cats and dogs that can be input into a model. Note that all the images are the same size, which was achieved by modifying the aspect ratio of the images. This ensures consistency in the images as they are passed into an ANN.
In this exercise, you learned how to import images in batches so they can be used for training ANNs. Images are loaded one batch at a time and by limiting the number of training images per batch, you can ensure that the RAM of the machine is not exceeded.


