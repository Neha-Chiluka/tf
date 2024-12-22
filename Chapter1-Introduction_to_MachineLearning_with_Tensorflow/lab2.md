# LAB - 2: Creating Scalars, Vectors, Matrices, and Tensors in TensorFlow:



#### Task 1 - Open Google Collab

1. Open Google Colab:

- Go to Google Colab.
- If prompted, sign in with your Google account.

2. Create a New Notebook:
- Once you're logged in, click on File in the top left corner.
- Select New Notebook.


3. Import TensorFlow:
- In the first cell of the notebook, enter the following code to import TensorFlow

`import tensorflow as tf`

- Press Shift + Enter to run the cell.

4. Create an Integer Variable (Scalar)

- Use TensorFlow's Variable class to create a scalar representing the number of days in a year (365).

- Specify tf.int16 to set the datatype explicitly

```python
day_in_year = tf.Variable(365, dtype=tf.int16)
print(day_in_year)

```

5 . Print the Rank of the Scalar Variable
Use the tf.rank() function to print the rank of the scalar variable.

`print(tf.rank(day_in_year))
`
4. Access the Scalar's Value (NumPy Representation)
Access the integer value using the .numpy() attribute.

`print(day_in_year.numpy())`

5. Print the Shape of the Scalar
Call the shape attribute to find the shape of the tensor.

`print(day_in_year.shape)`

print(day_in_year.shape)

6. Print the Shape of the Scalar as a Python List. 
Convert the shape to a Python list for better visualization

`print(day_in_year.shape.as_list())
`

7. Create a Vector Variable
- Use tf.Variable to create a vector (1D tensor) representing student scores.
- Specify tf.float32 as the datatype to ensure it's a float.

```python
scores = tf.Variable([85.5, 90.0, 88.5], dtype=tf.float32)
print(scores)

```

8. Print the Rank of the Vector Variable
Use tf.rank() to get the rank of the vector.

`print(tf.rank(scores))`

9. Print the Shape of the Vector Variable as a Python List
Print the shape of the vector variable as a list

`print(scores.shape.as_list())`

10. Create a Matrix Variable

- Use tf.Variable to create a 2D matrix representing the marks of three students in two exams.
- Specify tf.int32 for the datatype.

```python
marks_matrix = tf.Variable([[85, 90], [88, 92], [75, 80]], dtype=tf.int32)
print(marks_matrix)

```

11. Print the Rank of the Matrix Variable
Use tf.rank() to get the rank of the matrix.

`print(tf.rank(marks_matrix))`


In this exercise, you have successfully created tensors of various ranks from political voting data using TensorFlow's Variable class. First, you created scalars, which are tensors that have a rank of 0. Next, you created vectors, which are tensors with a rank of 1. Matrices were then created, which are tensors of rank 2. Finally, tensors were created that have rank 3 or more. You confirmed the rank of the tensors you created by using TensorFlow's rank function and verified their shape by calling the tensor's shape attribute. 
