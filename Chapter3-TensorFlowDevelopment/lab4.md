# Chapter 3: LAB - 4: Using Google Colab to Visualize Data.

In this exercise, you will load a dataset from google drive that has temperature during world war II dataset.


You can find the Summary of Weather.csv file here:
To perform the exercise, you will have to navigate to https://colab.research.google.com/ and create a new notebook to work in. You will need to connect to a GPU-enabled environment to speed up TensorFlow operations such as tensor multiplication. Once the data has been loaded into the development environment, you will view the first five rows. Next, you'll drop the Date field since matrix multiplication requires numerical fields. Then, you will perform tensor multiplication of the dataset with a tensor or uniformly random variables.


#### Task 1: Import TensorFlow and Check Version.

Import TensorFlow and print its version to ensure compatibility.

```python
import tensorflow as tf
print('TF version:', tf.__version__)
```

#### Task 2: Check for GPU Availability.

Verify if TensorFlow detects a GPU for computations.

```python
tf.test.gpu_device_name()
```

#### Task 3: Import Dataset Using Pandas.  

Load a CSV file from a remote URL into a Pandas DataFrame for processing.


```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/Neha-Chiluka/tf/refs/heads/main/Chapter3-TensorFlowDevelopment/dataset/Summary_of_Weather.csv")

```

#### Task 4: Display the First Few Rows of the Dataset. 

Use the head function to preview the first 5 rows of the dataset.

```python
df.head()
```

#### Task 5: Drop the 'Date' Column.

Remove the Date column from the DataFrame since it’s not required for further processing.


```python
df.drop("Date", axis=1, inplace=True)
```

#### Task 6: Extract and Clean Specific Columns.

Select a subset of columns (WindGustSpd, MaxTemp, MinTemp, MeanTemp) and drop rows with missing values.

```python
df1 = df.iloc[:, 2:6].dropna()
```

#### Task 7: Display the First Few Rows of Cleaned Data

Preview the first 5 rows of the cleaned subset of the dataset.

```python
df1.head()
```

#### Task 8: Inspect the Dataset Information

Check the structure, column data types, and memory usage of the cleaned dataset

`df1.info()
`

#### Task 9: Convert DataFrame to Numpy Array

Convert the cleaned DataFrame to a NumPy array with a float32 data type for TensorFlow compatibility.

```python
import numpy as np
df1 = np.asarray(df1).astype(np.float32)

```
#### Task 10: Generate a Random Tensor

Create a random tensor with the shape matching the number of columns in the dataset.

```python
random_tensor = tf.random.normal((df1.shape[1], 1))

```

#### Task 11: Perform Matrix Multiplication

Multiply the cleaned dataset array with the random tensor using TensorFlow's matmul function.

`tf.matmul(df1, random_tensor)
`


In this exercise, you learned how to use Google Colab. You observed that Google Colab provides a convenient environment to build machine learning models and comes pre-loaded with many of the libraries that may be needed for any machine learning application. You can also see that the latest versions of the libraries are used. Unfortunately, the versions of TensorFlow cannot be modified, so using Google Colab in production environments may not be the most appropriate application. However, it is great for development environments.

