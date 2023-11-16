import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import jdatetime 

# Function to get the DataFrame and extract the features
def feature_extractor(df: pd.DataFrame)-> pd.DataFrame:
  """
  This function gets a DataFrame with 3 columns: being, Time,Power,
  and Temp. Then extracts feature from time

  Args:
      df: a pandas DataFrame, including a time, power, and temp columns

  Returns:
      A new pandas.DataFrame object with the new features
  """
  # make a copy of the data frame
  copy_df = df.copy()
  # addding mutiple columns using the "time" col
  copy_df["week_day"] = copy_df["time"].dt.weekday
  copy_df["day"] = copy_df["time"].dt.day
  copy_df['month'] = copy_df["time"].dt.month
  copy_df["hour"] = copy_df["time"].dt.hour

  # extracting features
  copy_df['up_of_month']=(copy_df['day']<=10).astype(int)
  copy_df['down_of_month']=(copy_df['day']>20).astype(int)
  copy_df['sin_day']=np.sin(2*np.pi*copy_df['day']/30)
  copy_df['cos_day']=np.cos(2*np.pi*copy_df['day']/30)

  copy_df['morning']=((copy_df['hour']>5)&(copy_df['hour']<=12)).astype(int)
  copy_df['afternoon']=((copy_df['hour']>12)&(copy_df['hour']<=19)).astype(int)
  copy_df['evening']=(1-copy_df['morning']-copy_df['afternoon']).astype(int)

  #  whether its thursday | friday or not
  copy_df['weekend'] = ((copy_df["week_day"] == 4) | (copy_df["week_day"] == 3)).astype(int)
  copy_df['time_slot'] = pd.cut(copy_df["hour"], bins=[0, 6, 12, 18, 24], labels=[1, 2, 3, 4], right=False)
  copy_df['peak_load'] = ((copy_df["hour"] >= 11) & (copy_df["hour"] <= 15)).astype(int)

  copy_df['spring']=((copy_df['month']>=3)&(copy_df['month']<=5)).astype(int)
  copy_df['summer']=((copy_df['month']>=6)&(copy_df['month']<=8)).astype(int)
  copy_df['fall']=((copy_df['month']>=9)&(copy_df['month']<=11)).astype(int)
  copy_df['winter']=((copy_df['month']==12)&(copy_df['month']<=2)).astype(int)

  # dropping non-informative columns
  copy_df.drop( ["day", "week_day", "month","hour"] , axis = 1, inplace = True)
  print(f"number of features: {len(copy_df.columns)}, including the 'time' column\ncolumns are :\n{df.columns.values}")

  return copy_df


def std_normalizer(df : pd.DataFrame,
                   col_name : str):
  """
  This function gets a df frame and a column name and does the-
  normalization operation.

  Args:
      df : its a panda.DataFrame object.
      col_name : its a string and the name of the desired column to noramlize.
  
  Returns: A pandas.DataFrame as same as the input df frame, but with normalized column and 
  the std and mean values as a list in case, there is an unromalization.
  """
  # make a copy of the data frame
  copy_df = df.copy()

  df_mean = copy_df[col_name].mean()
  df_std = copy_df[col_name].std()
  copy_df.loc[:, col_name] = (copy_df[col_name] - df_mean) / df_std
  print(f"\nNormalization is done for column : {col_name}")
  return copy_df , (df_mean , df_std)


def hourly_rows_data(df : pd.DataFrame)-> pd.DataFrame:
  """
  This function turns hour original dataset into hourly shaped, which is
  each row followed by its repective value

  Args:
    df : it is the original dataframe
  
  Returns:
    new_df : it is a new hourly data frame
  """
  # check if the "time" column is in the data
  if "date" not in df.columns:
    raise ValueError("'date' column was not found in the dataset")
  # adding another column "time" which is the dates in Gregorian format and then removing the "date" column
  df['time'] = df['date'].apply(lambda x: jdatetime.datetime.strptime(x, '%Y/%m/%d').togregorian().strftime('%Y-%m-%d'))
  # then drop the "date"column
  df.drop(columns="date",axis = 1, inplace=True)

  # making the dataframe into a column vector of time and power consumption
  # we make to lists of hours and their repective values and then concatenate them
  time_hourly = []
  values_hourly = []
  list_of_hours = np.arange(1,25)

  for i in range(len(df["time"])):
    time = pd.to_datetime(df['time'][i], format='%Y-%m-%d') + pd.to_timedelta(list_of_hours, unit='h').values
    values = df.loc[i][:-1].values
    time_hourly.append(time)
    values_hourly.append(values)

  time_concatenated = np.concatenate(time_hourly)
  values_concatenated = np.concatenate(values_hourly)

  # and now we make the new df
  new_df = pd.DataFrame({"time": time_concatenated, "power": values_concatenated})
  print(f"\nchanged the data frame's shape from {df.shape} to {new_df.shape}\n")

  return new_df



def create_historical_dataset(df: pd.DataFrame,
                              window_size : int,
                              exclude_columns: list =[])-> pd.DataFrame:
  """
  This function recieves a pd.DataFrame object, with multiple features and a window size
  then creates a new df frame with each row representing window-sized previous values , and a
  respective next target value.

  Args:
    df : Its the desired df frame.
    window_size : Its a int number for creating the windowed data frame

  Returns:
    A new pd.DataFrame object with windowed features 
  """
  # make a copy of the data frame
  copy_df = df.copy()
  
  # Ensure the input DataFrame has the required columns
  if 'power' not in copy_df.columns:
      raise ValueError("Input DataFrame must have a 'power' column.")

  # Create empty DataFrame to store historical df
  historical_df = pd.DataFrame()

  # Include original features
  for col in copy_df.columns:
    historical_df[col] = copy_df[col]

  # exclude the columns that are not going to be shiftted
  copy_df.drop(columns=exclude_columns, axis=1, inplace=True)
  
  # Create lag features for the specified window size
  for i in range(1, window_size + 1):
    for col in copy_df.columns:
      historical_df[f'{col}_lag_{i}'] = copy_df[col].shift(i)

  # Add the target variable (power) shifted by the window size
  historical_df['target_power'] = copy_df['power'].shift(-1)

  # Drop rows with NaN values introduced by shifting
  historical_df = historical_df.dropna().reset_index(drop=True)

  return historical_df
