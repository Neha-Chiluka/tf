# LAB - 1: Loading Tabular Data and Rescaling Numerical Fields

In this exercise, you will perform tensor multiplication in TensorFlow using TensorFlow's matmul function and the @ operator. In this exercise, you will use the example of data from a sandwich retailer representing the ingredients of various Pizzas and the costs of different ingredients. You will use matrix multiplication to determine the costs of each sandwich.



#### Task 1 - Open Google Collab

1. Open Google Colab:

- Go to Google Colab.
- If prompted, sign in with your Google account.

2. Create a New Notebook:
- Once you're logged in, click on File in the top left corner.
- Select New Notebook.

#### Task 2 - Import TensorFlow

- Import the TensorFlow library to perform matrix operations

```python
import tensorflow as tf
```

#### Task 3: Create a Matrix Representing Pizza Recipes

- Create a matrix to represent the number of ingredients needed for three types of pizzas, with rows representing different pizzas and columns representing five different ingredients. Use TensorFlow's Variable class.

```python
matrix1 = tf.Variable([[1.0, 0.0, 3.0, 1.0, 2.0], 
                       [0.0, 1.0, 1.0, 1.0, 1.0], 
                       [2.0, 1.0, 0.0, 2.0, 0.0]], 
                      tf.float32)
matrix1

```

#### Task 4: Verify the Shape of Matrix1. 

- Verify the shape of the pizza recipe matrix by calling its shape attribute as a Python list

```python
matrix1.shape.as_list()
```

#### Task 5: Create a Matrix Representing Ingredient Costs and Weights

- Create a second matrix representing the cost and weight of each of the five ingredients, with rows representing ingredients and columns representing cost and weight:


```python
matrix2 = tf.Variable([[0.49, 103], 
                       [0.18, 38], 
                       [0.24, 69], 
                       [1.02, 75], 
                       [0.68, 78]])
matrix2

```

#### Task 6: Perform Matrix Multiplication to Calculate Total Costs and Weights

Use TensorFlow's matmul function to multiply matrix1 and matrix2 to calculate the cost and weight for each pizza type

```python
matmul1 = tf.matmul(matrix1, matrix2)
matmul1

```

#### Task 7:  Create a Matrix Representing Sales Projections

Create a matrix to represent sales projections for five different stores, with rows representing stores and columns representing the number of each type of pizza sold:

```python
matrix3 = tf.Variable([[120.0, 100.0, 90.0], 
                       [30.0, 15.0, 20.0], 
                       [220.0, 240.0, 185.0], 
                       [145.0, 160.0, 155.0], 
                       [330.0, 295.0, 290.0]])


```

#### Task 8: Multiply Sales Projections by Pizza Costs and Weights

Multiply matrix3 by the result of the matrix multiplication of matrix1 and matrix2 to determine the total cost and weight for each store

```python
matmul3 = matrix3 @ matmul1
matmul3


```

#### Task 9: Analyze the Final Result

Examine the resulting tensor from the multiplication. It should contain the expected total cost and weight of ingredients for each store.

```python
matmul3
```


