# Activity 3.01: Using TensorBoard to Visualize Tensor Transformations

You are given two tensors of shape 7x7x7. You are required to create TensorFlow functions to perform a tensor transformation and view a visual representation of the transformation.

The steps you will take are as follows:

1.	Import the TensorFlow library and set the seed to 10. 
2.	Set a log directory and initialize a file writer object to write the trace. 
3.	Create a TensorFlow function to multiply two tensors, add a value of 1 to all elements in the resulting tensor using the ones_like function to create a tensor of the same shape as the result of the matrix multiplication. Then, apply a sigmoid function to each value of the tensor. 
4.	Create two tensors with the shape 7x7x7. 
5.	Turn on graph tracing. 
6.	Apply the function to the two tensors and export the trace to the log directory. 
7.	Launch TensorBoard in the command line and view the graph in a web browser:
 
![](https://github.com/Neha-Chiluka/tf/blob/main/images/9.png?raw=true)
