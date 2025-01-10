# Activity 2.01: Loading Tabular Data and Rescaling Numerical Fields with a MinMax Scaler


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
 
![](https://github.com/Neha-Chiluka/tf/blob/main/images/7.png?raw=true)

