# LAB - 3: Performing Tensor Addition in TensorFlow

The marks scored by three different students at three different universities in entrance Exam A and Exam B are as follows:



•	Store the total number of marks scored in University X in Exam A
•	Store the total number of marks scored in each University in Exam A
•	Store the total number of marks scored in each University in Both Exams

#### Task 1 - Open Google Collab

1. Open Google Colab:

- Go to Google Colab.
- If prompted, sign in with your Google account.

2. Create a New Notebook:
- Once you're logged in, click on File in the top left corner.
- Select New Notebook.

#### Task 2 - Import Tensorflow

1. Import TensorFlow:
- In the first cell of the notebook, enter the following code to import TensorFlow

`import tensorflow as tf`

#### Task 3: Create Scalar Variables for Marks of Students in University X for Exam A

3. Create Scalar Variables:

- Use tf.Variable to create three scalar variables representing the marks of three students at University X in Exam A.

```python
int1 = tf.Variable(41, tf.int32)
int2 = tf.Variable(38, tf.int32)
int3 = tf.Variable(51, tf.int32)

```

Task 4: Create Variable for Total Marks Scored by University X in Exam A

4. Create a Variable for Total Marks:

- Store the total marks scored by University X in Exam A by summing up the three students' marks.

```python
int_sum = int1+int2+int3
```

#### Task 5: Print the Result of the Sum as a NumPy Variable

5. Print the Result:

- Convert the result to a NumPy variable and print the total marks.

`int_sum.numpy()
`
#### Task 6: Create Vectors for Marks Scored in Exam A for Each University

6. Create Vectors for Marks:

- Create three vectors representing the marks scored by three different universities (University X, Y, and Z) in Exam A.

```python
vec1 = tf.Variable([41, 38, 51], tf.int32)
vec2 = tf.Variable([75, 67, 70], tf.int32)
vec3 = tf.Variable([62, 69, 65], tf.int32)
```

#### Task 7: Create a New Variable for Total Marks Scored in Each University in Exam A

7. Create a Variable for Total Marks in Each University:

Sum the marks for each university to get the total marks scored in Exam A.

`vec_sum = vec1 + vec2 + vec3`

#### Task 8: Print the Result of the Vector Addition as a NumPy Array

8. Print the Result of the Addition:

- Print the result of the sum of the three vectors as a NumPy array.

`vec_sum.numpy()`

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
