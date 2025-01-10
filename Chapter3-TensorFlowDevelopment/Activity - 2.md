# Activity 3.02: Performing Word Embedding from a Pre-Trained Model from TensorFlow Hub

In this activity, you will practice working in the Google Colab environment. You will download a universal sentence encoder from TensorFlow Hub from the following URL: https://tfhub.dev/google/universal-sentence-encoder/4. Once the model has been loaded into memory, you will use it to encode some sample text.

Follow these steps:

1.	Import TensorFlow and TensorFlow Hub and print the version of the library.
2.	Set the handle for the module as the URL for the universal sentence encoder. 
3.	Use the TensorFlow Hub KerasLayer class to create a hub layer, passing in the following arguments: module_handle, input_shape, and dtype.
4.	Create a list containing a string, “Performing word embedding”, to encode with the encoder.
5.	Apply hub_layer to the text to embed the sentence as a vector

Your final output should be like the following:

![](https://github.com/Neha-Chiluka/tf/blob/main/images/10.png?raw=true)
