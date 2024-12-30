# Chapter 2: LAB - 2: Loading Tabular Data and Rescaling Numerical Fields

In this exercise, you will pre-process the date column by one-hot encoding the year and the month from the date column using the get_dummies function. You will join the one-hot-encoded columns with the original Data Frame and ensure that all the fields in the resultant Data Frame are numerical.

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
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
```

#### Task 5: Perform One-Hot Encoding for the Year

Extract the year from the Date column and apply one-hot encoding using get_dummies. This converts categorical values into binary columns.

```python
year_dummies = pd.get_dummies(df['Date'].dt.year, \
                                  prefix='year')
year_dummies

```

#### Task 6: Perform One-Hot Encoding for the Month

Extract the month from the Date column and apply one-hot encoding using get_dummies to convert months into binary columns..

```python
month_dummies = pd.get_dummies(df['Date'].dt.month, \
                                   prefix='month')
month_dummies

```

#### Task 7:  Concatenate the Dummy DataFrames with the Original DataFrame

Add the one-hot encoded columns for year and month to the original DataFrame using the concat function.

```python
df = pd.concat([df, month_dummies, year_dummies], \
                   axis=1)
```
#### Task 8: Drop the Original Date Column and Irrelevant Columns
Drop the redundant Date column and other columns with too many missing values or irrelevance to simplify the dataset.

```python
# Check the existing columns in your DataFrame
print(df.columns)

# Drop only the columns that exist in the DataFrame
columns_to_drop = ['Date', 'Precip','Snowfall', 'PoorWeather', 'PRCP',
        'TSHDSBRSGF', 'SNF']
existing_columns = [col for col in columns_to_drop if col in df.columns]

df.drop(existing_columns, axis=1, inplace=True)
```


#### Task 9: Verify the Data Types of All Columns

Use the dtypes attribute to confirm that all columns in the resultant DataFrame are numerical.

`df.dtypes`


Here, you can see that all the data types of the resultant DataFrame are numerical. This means they can now be passed into an ANN for modeling.
In this exercise, you successfully imported tabular data and pre-processed the date column using the pandas and scikit-learn libraries. You utilized the get_dummies function to convert categorical data into numerical data types.

