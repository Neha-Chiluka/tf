# Chapter 2: LAB - 1: Loading Tabular Data and Rescaling Numerical Fields

The dataset, Weather Conditions in World War Two , contains information temperature during the world war II, The fields represent temperature measures of the given date, the weather station at which the metrics were measured, model forecasts of weather-related metrics such as WindGustSpd, MaxTemp, MinTemp, Meantemp and so on. You are required to pre-process the data to make all the columns normally distributed with a mean 0 and a standard deviation of 1. You will demonstrate the effects with the MaxTemp column, which represents the maximum temperature on the given data at a given weather station.


#### Task 1 - Open Google Collab

1. Open Google Colab:

- Go to Google Colab.
- If prompted, sign in with your Google account.

2. Create a New Notebook:
- Once you're logged in, click on File in the top left corner.
- Select New Notebook.

#### Task 2 - Import the Required Libraries

Import the pandas library to handle tabular data.

```python
import pandas as pd

```

#### Task 3: Load the Dataset into a Pandas DataFrame and Display the Data

Read the CSV file containing weather data into a DataFrame using pd.read_csv() and preview the first few rows of the dataset using the .head() method.

```python
url = "https://raw.githubusercontent.com/fenago/tf/main/Chapter4-Regression_and_Classification_Models/dataset/Summary_of_Weather.csv"
df = pd.read_csv(url)
df.head()


```

#### Task 4: Drop the Non-Numerical Date Column from the DataFrame.

- Drop the Date column as it is non-numerical, and preprocessing numerical data is the focus. Use drop() with axis=1 to specify a column, and inplace=True to modify the DataFrame directly.

```python
df.drop("Date", inplace=True, axis=1)
```

#### Task 5: Plot a Histogram of the MaxTemp Column

Create a histogram for the MaxTemp column to visualize the frequency distribution of maximum temperatures in the dataset.


```python
ax = df['MaxTemp'].hist(color='gray', bins=20)
ax.set_xlabel("MaxTemp")
ax.set_ylabel("Frequency")

```

#### Task 6: Import and Apply Standard Scaling to the Temperature Columns

Normalize the MaxTemp, MinTemp, and MeanTemp columns using StandardScaler to ensure the mean is 0 and the standard deviation is 1. 

Convert the scaled array back to a DataFrame for ease of use.

```python
from sklearn.preprocessing import StandardScaler
df1 = df.loc[:, ['MaxTemp', 'MinTemp', 'MeanTemp']]
scaler = StandardScaler() # Create an instance of StandardScaler
df2 = scaler.fit_transform(df1)
df2 = pd.DataFrame(df2, columns=['MaxTemp', 'MinTemp', 'MeanTemp'])
df2.head()

```

#### Task 7:  Plot a Histogram of the Transformed MaxTemp Column

Plot a histogram for the normalized MaxTemp column to check the distribution of values after scaling.

```python
ax = df2['MaxTemp'].hist(color='gray')
ax.set_xlabel("Normalized Temperature")
ax.set_ylabel("Frequency")
```
The resulting histogram shows that the temperature values range from around -3 to 3 degrees Celsius, as evidenced by the range on the x axis of the histogram. By using the standard scaler, the values will always have a mean of 0 and a standard deviation of 1. Having the features normalized can speed up the model training process.

In this exercise, you successfully imported tabular data using the pandas library and performed some preprocessing using the scikit-learn library. The preprocessing of data included dropping the date column and scaling the numerical fields so that they have a mean value of 0 and a standard deviation of 1.

