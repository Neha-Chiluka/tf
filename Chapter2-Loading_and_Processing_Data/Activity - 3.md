# Activity 2.03: Loading Audio Data for Batch Processing

In this activity, you will load audio data for batch processing. The audio preprocessing techniques that will be performed include taking the absolute value and using the logarithm of 1 plus the value. This will ensure the resulting values are non-negative and logarithmically scaled. The result will be a preprocessed dataset object that can be input into a TensorFlow model for training.

The steps for this activity are as follows:
1.	Open a new Jupyter notebook to implement this activity. 
2.	Import the TensorFlow and os libraries. 
3.	Create a function that will load and then decode an audio file using TensorFlow's read_file function followed by the decode_wav function, respectively. Return the transpose of the resultant tensor from the function. 
4.	Load the file paths into the audio data as a list using os.list_dir. 
5.	Create a function that takes a dataset object, shuffles it, loads the audio using the function you created in step 2, and applies the absolute value and the log1p function to the dataset. This function adds 1 to each value in the dataset and then applies the logarithm to the result. Next, repeat the dataset object, batch it, and prefetch it with a buffer size equal to the batch size. 
6.	Create a dataset object using TensorFlow's from_tensor_slices function and pass in the paths to the audio files. Then, apply the function you created in Step 4 to the dataset created in Step 5. 
7.	Take the first batch of the dataset and print it out. 
8.	Plot the first audio file from the batch.

The output will look as follows:
 
![](https://github.com/Neha-Chiluka/tf/blob/main/images/8.png?raw=true)


