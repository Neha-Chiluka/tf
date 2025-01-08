**Activity 2: Performing Tensor Reshaping and Transposition in TensorFlow**

In this activity, you are required to simulate the grouping of 24 school children for class projects. The dimensions of each resulting reshaped or transposed tensor will represent the size of each group.


**Perform the following steps:**

1.	Import the TensorFlow library.

2.	Create a one-dimensional tensor with 24 monotonically increasing elements using the Variable class to represent the IDs of the school children. Verify the shape of the matrix.

You should get the following output as   **[24]**

3.	Reshape the matrix so that it has 12 rows and 2 columns using TensorFlow’s reshape function representing 12 pairs of school children. Verify the shape of the new matrix.

You should get the following output as **[12,2]**
 
4.	Reshape the original matrix so that it has a shape of 3x4x3 using TensorFlow’s reshape function representing 3 groups of 4 sets of pairs of school children. Verify the shape of the new tensor.

You should get the following output   **[3,4,2]**
 

5.	Verify that the rank of this new tensor is 3.

6.	Transpose the tensor created is step 3 to represent 2 groups of 12 students using TensorFlow’s transpose function. Verify the shape of the new tensor.

You should get the following output as  **[2,12]**
 

**Note:** This solution to this activity can be found via this link.

