# Activity 7.01: Building a CNN with More ANN Layers

The start-up that you have been working for has loved your work so far. They have tasked you with creating a new model classifying images from 10 different classes.

In this activity, you will be putting everything that you've learned to use as you build your own classifier with dataset, with 10 classes, and is commonly used for benchmarking performance in machine learning research.

1. Start a new Jupyter notebook. 
2. Import the TensorFlow library. 
3. Import the additional libraries that you will need, including NumPy, Matplotlib, Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, Activation, Model, confusion_matrix, and itertools. 
4. Load the CIFAR-10 dataset directly from tensorflow_datasets and view its properties from the metadata, and build a train and test data pipeline:

![](https://github.com/Neha-Chiluka/tf/blob/main/images/17.png?raw=true)

![](https://github.com/Neha-Chiluka/tf/blob/main/images/18.png?raw=true)

5. 	Create a function to rescale images. Then, build a test and train data pipeline by rescaling, caching, shuffling, batching, and prefetching the images.
6. •	Build the model using the functional API using Conv2D and Flatten, among others.
7. •	Compile and fit the model using model.compile and model.fit:

![](https://github.com/Neha-Chiluka/tf/blob/main/images/19.png?raw=true)

8. •	Plot the loss with plt.plot. Remember to use the history collected during the model.fit() procedure:

 ![](https://github.com/Neha-Chiluka/tf/blob/main/images/20.png?raw=true)

9. •	Plot the accuracy with plt.plot:

 ![](https://github.com/Neha-Chiluka/tf/blob/main/images/21.png?raw=true)

10. •	Specify the labels for the different classes in your dataset.
11. •	Display a misclassified example with plt.imshow:

![](https://github.com/Neha-Chiluka/tf/blob/main/images/22.png?raw=true)

