
# **Activity 3: Applying Activation Functions**

In this activity, you will recall many of the concepts used throughout the chapter as well as apply activation functions to tensors. You will use example data of bike dealership sales, apply these concepts, show the sales records of various salespeople, and highlight those with net positive sales.

**Sales records:**
 ![](https://github.com/Neha-Chiluka/tf/blob/main/images/2.png?raw=true)


**Vehicle MSRPs:**

![](https://github.com/Neha-Chiluka/tf/blob/main/images/3.png?raw=true)

Fixed costs:
 
 ![](https://github.com/Neha-Chiluka/tf/blob/main/images/4.png?raw=true)


Perform the following steps:

1.	Import TensorFlow library

2.	Create a 3x4 tensor as an input with the values [[-0.013, 0.024, 0.06, 0.022], [0.001, -0.047, 0.039, 0.016], [0.018, 0.030, -0.021, -0.028]]. The rows in this tensor represent the sales of various sales representatives, the columns represent various vehicles available at the dealership, and values represent the average percentage difference from MSRP. The values are positive or negative depending on whether the salesperson was able to sell for more or less than the MSRP.

3.	Create a 4x1 weights tensor with the shape 4x1 with the values [[19995.95], [24995.50], [36745.50], [29995.95]] representing the MSRP of the cars.

4.	Create a bias tensor of size 3x1 with the values [[-2500.0], [-2500.0], [-2500.0]] representing the fixed costs associated with each salesperson.

5.	Matrix multiply the input by the weight to show the average deviation from the MSRP on all cars and add the bias to subtract the fixed costs of the salesperson. Print the result.

 

![](https://github.com/Neha-Chiluka/tf/blob/main/images/5.png?raw=true)

6.	Apply a ReLU activation function to highlight the net-positive salespeople and print the result.

You should get the following result:

![](https://github.com/Neha-Chiluka/tf/blob/main/images/6.png?raw=true)
