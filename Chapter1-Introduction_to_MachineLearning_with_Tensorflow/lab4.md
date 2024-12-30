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
reshape1.shape.as_list()
```

#### Task 5: Reshape the Matrix to 1x8

- Reshape the matrix to one row and eight columns using TensorFlow's reshape function.
- Verify and print the new shape as a Python list.

```python
reshape2 = tf.reshape(matrix1, shape=[1, 8])
reshape2.shape.as_list()
```

#### Task 6: Reshape the Matrix to 8x1

- Reshape the matrix to eight rows and one column using TensorFlow's reshape function.
- Verify and print the new shape as a Python list.

```python
reshape3 = tf.reshape(matrix1, shape=[8, 1])
reshape3.shape.as_list()
```

#### Task 7:  Reshape the Matrix to 2x2x2

- Reshape the matrix into a tensor of size 2x2x2 using TensorFlow's reshape function.
- Verify and print the new shape as a Python list.

```python
reshape4 = tf.reshape(matrix1, shape=[2, 2, 2])
reshape4.shape.as_list()

```

#### Task 8: Verify Rank of the Reshaped Tensor

- Use TensorFlow's rank function to verify the rank of the reshaped tensor.

```python
tf.rank(reshape4).numpy()

```

#### Task 9: Transpose the Original Matrix (2x4)

- Transpose the matrix from 2x4 to 4x2:

- Verify and print the transposed tensor.

```python
transpose1 = tf.transpose(matrix1)
transpose1
```


#### Task 10: Compare Reshaping vs. Transposition

Compare the result of reshaping and transposing the matrix using equality


`transpose1 == reshape1
`

#### Task 11: Transpose the Reshaped Tensor (2x2x2)

Transpose the tensor reshaped in Task 7

```python
transpose2=tf.transpose(reshape4)
transpose2
```
In this exercise, you have successfully modified the shape of a tensor either through reshaping or transposition. You studied how the shape and rank of the tensor changes following the reshaping and transposition operation.
