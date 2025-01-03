# Chapter 3: LAB - 1: Using TensorBoard to Visualize Matrix Multiplication.

In this exercise, you will perform matrix multiplication of 5x5 matrices with random values and trace the computation graph and profiling information. Following that, you will view the computation graph using TensorBoard. This exercise will be performed in a Jupyter notebook. Launching TensorBoard will require running a command on the command line, as shown in the final step

#### Task 1 - Open Google Collab

1. Open Google Colab:

- Go to Google Colab.
- If prompted, sign in with your Google account.

2. Create a New Notebook:
- Once you're logged in, click on File in the top left corner.
- Select New Notebook.

#### Task 2 - Import TensorFlow and Set the Random Seed

First, you import the TensorFlow library and set the random seed to ensure reproducibility of results. This will ensure that the random values generated in your matrices are the same every time the code is run.

```python
import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(11)


```

#### Task 3: Create a File Writer Object and Set the Directory for Logs

create a tf.summary.create_file_writer object that will store the logs in a specific directory. In Google Colab, you can specify the path to the log directory, for example, /content/logs/:

```python
import os

# Create a log directory for TensorBoard
logdir = '/content/logs/'  # Use a full path in Colab
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create a summary writer
writer = tf.summary.create_file_writer(logdir)

```

#### Task 4: Create a TensorFlow Function for Matrix Multiplication. 

You define a simple TensorFlow function that will multiply two matrices. This function will be used later in the computation.


```python
@tf.function
def my_matmult(x, y):
    result = tf.matmul(x, y)
    return result
```

#### Task 5: Create Sample Data (Random 5x5 Matrices)

create the random input matrices x and y, both of shape 5x5, using tf.random.uniform. These will be multiplied later.

```python
# Generate random input matrices of shape 5x5
x = tf.random.uniform((5, 5))
y = tf.random.uniform((5, 5))

```

#### Task 6:  Turn On Graph Tracing Using TensorFlow's Summary Class

To trace the graph and profile the execution, you need to use tf.summary.trace_on(). This enables the collection of the computation graph and profiling data. It’s important to enable this before performing any computation.


```python
# Reset profiling state (optional but recommended)
tf.summary.trace_off()

# Enable tracing and profiling, passing profiler_outdir to trace_on()
tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=logdir)

```

#### Task 7: Apply the Function and Export the Trace

Now, apply the matrix multiplication function to the random tensors (x and y) and export the trace. The trace_export function saves the computation graph for visualization.

```python
# Perform matrix multiplication (this triggers the trace)
z = my_matmult(x, y)

# After the trace has been run, export the trace to TensorBoard
# The profiler_outdir is already passed in trace_on(), so no need to pass it in trace_export()
with writer.as_default():
    tf.summary.trace_export(
        name="my_trace",
        step=0
    )


```

#### Task 8: Launch TensorBoard in Google Colab

In Colab, you use %tensorboard magic to launch TensorBoard directly within the notebook. This allows you to visualize the computation graph.

```python
# Launch TensorBoard in the notebook
%load_ext tensorboard
%tensorboard --logdir='/content/logs' --host 0.0.0.0


```

In TensorBoard, you can view the process of a tensor multiplying the two matrices to produce another matrix. By selecting the various elements, you can view information about each individual object in the computational graph, depending on the type of object. Here, you have created two tensors, named x and y, represented by the nodes at the bottom. By selecting one of the nodes, you can view attributes about the tensor, including its data type (float), its user-specified name (x or y), and the name of the output node (MatMul). These nodes representing the input tensors are then input into another node representing the tensor multiplication process labeled MatMul after the TensorFlow function. Selecting this node reveals attributes of the function, including the input arguments, the input nodes (x and y), and the output node (Identity). The final two nodes, labeled Identity and identity_RetVal, represent the creation of the output tensor.

In this exercise, you used TensorBoard to visualize a computational graph. You created a simple function to multiply two tensors together and you recorded the process by tracing the graph and logging the results. After logging the graph, you were able to visualize it by launching TensorBoard and directing the tool to the location of the logs.


