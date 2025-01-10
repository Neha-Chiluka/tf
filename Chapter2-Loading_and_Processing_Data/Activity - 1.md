#### Activity 2.01: Loading Tabular Data and Rescaling Numerical Fields with a MinMax Scaler


In this activity, you are required to load tabular data and rescale the data using a MinMax scaler. The dataset, Summary of Weather.csv, contains information related to the climate condition during world war II. The fields represent temperature measurements of the given date, the weather station at which the metrics were measured, model forecasts of weather-related metrics such as MinTemp, MaxTemp and MeanTemp. You are required to scale the columns so that the minimum value of each column is 0 and the maximum value is 1.

Perform the following steps to complete this activity:
1.	Open a new Jupyter notebook to implement this activity.
2.	Import pandas and the Summary of Weather.csv dataset.
3.	Read the dataset using the pandas read_csv function.
4.	Drop the date column of the DataFrame.
5.	Plot a histogram of the MaxTemp column.
6.	Import MinMaxScaler and fit it to and transform the feature DataFrame.
7.	Plot a histogram of the transformed MaxTemp column.
You should get an output similar to the following:
 
Figure 2.8: Expected output of Activity 2.01

One method of converting non-numerical fields such as categorical or date fields is to one-hot encode them. The one-hot encoding process creates a new column for each unique value in the provided column, while each row has a value of 0 except for the one that corresponds to the correct column. The column headers of the newly created dummy columns correspond to the unique values. One-hot encoding can be achieved by using the get_dummies function of the pandas library and passing in the column to be encoded. An optional argument is to provide a prefix feature that adds a prefix to the column headers. This can be useful for referencing the columns:
 
Note:

When using the get_dummies function, NaN values are converted into all zeros.

In the following exercise, you'll learn how to pre-process non-numerical fields. You will utilize the same dataset that you used in the previous exercise and activity, which describes the temperature during the world war II
