# LAB - 4: Performing Tensor Reshaping and Transposition in TensorFlow

In this exercise, you will learn how to perform tensor reshaping and transposition using the TensorFlow library.

#### Task 1 - Open Google Collab

1. Open Google Colab:

- Go to Google Colab.
- If prompted, sign in with your Google account.

2. Create a New Notebook:
- Once you're logged in, click on File in the top left corner.
- Select New Notebook.

#### Task 2 - Import TensorFlow and Create a Matrix

- Import TensorFlow:
-  Create a matrix with two rows and four columns using TensorFlow's Variable class.


```python
import tensorflow as tf
matrix1 = tf.Variable([[11,12,13,14], [15,16,17,18]])
```

#### Task 3: Verify the Shape of the Matrix

Access and print the shape of the matrix as a Python list.

```python
matrix1.shape.as_list()
```

#### Task 4: Reshape the Matrix to 4x2

- Use TensorFlow's reshape function to reshape the matrix into a 4x2 matrix.
- Verify and print the new shape as a Python list.

```python
reshape1 = tf.reshape(matrix1, shape=[4, 2])
reshape1
```

#### Task 5: Reshape the Matrix to 1x8

- Reshape the matrix to one row and eight columns using TensorFlow's reshape function.
- Verify and print the new shape as a Python list.

`reshape1.shape.as_list()
`
#### Task 6: Reshape the Matrix to 8x1

- Reshape the matrix to eight rows and one column using TensorFlow's reshape function.
- Verify and print the new shape as a Python list.

```python
reshape2 = tf.reshape(matrix1, shape=[1, 8])
reshape2
```

#### Task 7:  Reshape the Matrix to 2x2x2

- Reshape the matrix into a tensor of size 2x2x2 using TensorFlow's reshape function.
- Verify and print the new shape as a Python list.

`reshape2.shape.as_list()`

#### Task 8: Verify Rank of the Reshaped Tensor

- Use TensorFlow's rank function to verify the rank of the reshaped tensor.
- Convert the result to a NumPy variable and print it.

```python
reshape3 = tf.reshape(matrix1, shape=[8, 1])
reshape3
```

#### Task 9: Verify Vector Addition by Performing Element-wise Addition
9. Verify the Addition:

- Confirm the element-wise addition of each vector by performing a simple addition of corresponding elements in the vectors.

```python
print((vec1[0] + vec2[0] + vec3[0]).numpy())
print((vec1[1] + vec2[1] + vec3[1]).numpy())
print((vec1[2] + vec2[2] + vec3[2]).numpy())
```

Task 10: Create Matrices to Represent Votes for Political Candidates

10. Create Matrices for Votes:
- Create three matrices representing the votes cast for candidates of each political party in three different districts.

```python
matrix1 = tf.Variable([[41, 38, 51], \
                           [36, 95, 80]], tf.int32)
matrix2 = tf.Variable([[75, 67, 70], \
                           [59, 78, 45]], tf.int32)
matrix3 = tf.Variable([[62, 69, 65], \
                           [62, 98, 48]], tf.int32)

```

#### Task 11: Verify that the Matrices Have the Same Shape

11. Verify Matrices Shape:

- Ensure that all matrices (votes matrices) have the same shape by printing their shapes.

`matrix1.shape == matrix2.shape == matrix3.shape
`

#### Task 12: Create a Variable for Total Marks in Each University in Both Exams

12. Create a Variable for Total Marks in Both Exams:

- Create a new variable to store the total marks scored in each university.

`matrix_sum = matrix1 + matrix2 + matrix3
`

#### Task 13: Print the Result of the Total Marks as a NumPy Array

13. Print the Result of the Summation:

- Print the result of the total marks scored in both exams as a NumPy array.

`matrix_sum.numpy()`

#### Task 14: Verify Matrix Addition by Performing Element-wise Addition

14. Verify the Matrix Addition:

- Perform and verify the element-wise addition of the three matrices representing votes cast for candidates across different districts.

```python
print((matrix1[0][0] + matrix2[0][0] + matrix3[0][0]).numpy())
print((matrix1[0][1] + matrix2[0][1] + matrix3[0][1]).numpy())
print((matrix1[0][2] + matrix2[0][2] + matrix3[0][2]).numpy())
print((matrix1[1][0] + matrix2[1][0] + matrix3[1][0]).numpy())
print((matrix1[1][1] + matrix2[1][1] + matrix3[1][1]).numpy())
print((matrix1[1][2] + matrix2[1][2] + matrix3[1][2]).numpy())
```

You can see that the + operation is equivalent to the element-wise addition of the three matrices created.

In this exercise, you successfully performed tensor addition on data representing votes cast for political candidates. The transformation can be applied by using the + operation. You also verified that addition is performed element by element, and that one way to ensure that the transformation is valid is for the tensors to have the same rank and shape.
