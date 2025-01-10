#### Activity 2.02: Loading Image Data for Batch Processing

In this activity, you will load image data for batch processing and augment the images in the process. The image_data2 folder contains a set of images of mickey_mouse and donald_duck. You are required to load in image data for batch processing and adjust the input data with random perturbations such as rotations, flipping the image horizontally, and adding shear to the images. This will create additional training data from the existing image data and will lead to more accurate and robust machine learning models by increasing the number of different training examples even if only a few are available. You are then tasked with printing the labeled images of a batch from the data generator.

The steps for this activity are as follows:

1.	Open a new Jupyter notebook to implement this activity.
2.	Import the ImageDataGenerator class from tensorflow.keras.preprocessing.image.
3.	Instantiate ImageDataGenerator and set the rescale=1./255, shear_range=0.2, rotation_range=180, zoom_range=0.2, and horizontal_flip=True arguments.
4.	Use the flow_from_directory method to direct the data generator to the images while passing in the target size as 64x64, a batch size of 15, and the class mode as binary.
5.	Create a function to display the first 15 images in a 5x5 array with their associated labels.
6.	Take a batch from the data generator and pass it to the function to display the images and their labels.
Note

In this activity, you augmented images in batches so they could be used for training ANNs. You've seen that when images are used as input, they can be augmented to generate a larger number of effective training examples.
