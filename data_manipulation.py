import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# Function to get the DataFrame and extract the features
def feature_extractor(df: pd.DataFrame)-> pd.DataFrame:
    """
    This function gets a DataFrame with 3 columns: being, Time,Power,
    and Temp. Then extracts required features

    Args:
        df: a pandas DataFrame, including a time, power, and temp columns

    Returns:
        A new pandas.DataFrame object with the new features
    """
    # addding mutiple columns using the "time" col
    df["week_day"] = df["time"].dt.weekday
    df["day"] = df["time"].dt.day
    df['month'] = df["time"].dt.month
    df["hour"] = df["time"].dt.hour

    # extracting features
    df['up_of_month']=(df['day']<=10).astype(int)
    df['down_of_month']=(df['day']>20).astype(int)
    df['sin_day']=np.sin(2*np.pi*df['day']/30)
    df['cos_day']=np.cos(2*np.pi*df['day']/30)

    df['morning']=((df['hour']>5)&(df['hour']<=12)).astype(int)
    df['afternoon']=((df['hour']>12)&(df['hour']<=19)).astype(int)
    df['evening']=(1-df['morning']-df['afternoon']).astype(int)

    #  whether its thursday | friday or not
    df['weekend'] = ((df["week_day"] == 4) | (df["week_day"] == 3)).astype(int)
    df['time_slot'] = pd.cut(df["hour"], bins=[0, 6, 12, 18, 24], labels=[1, 2, 3, 4], right=False)
    df['peak_load'] = ((df["hour"] >= 11) & (df["hour"] <= 15)).astype(int)

    df['spring']=((df['month']>=3)&(df['month']<=5)).astype(int)
    df['summer']=((df['month']>=6)&(df['month']<=8)).astype(int)
    df['fall']=((df['month']>=9)&(df['month']<=11)).astype(int)
    df['winter']=((df['month']==12)&(df['month']<=2)).astype(int)

    # dropping non-informative columns
    df.drop( ["day", "week_day", "month","hour"] , axis = 1, inplace = True)
    print(f"number of features: {len(df.columns)}, including the 'time' column\ncolumns are :\n{df.columns.values}")

    return df

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
    df_mean = df[col_name].mean()
    df_std = df[col_name].std()
    df.loc[:, col_name] = (df[col_name] - df_mean) / df_std
    print(f"\nNormalization is done for column : {col_name}")
    return df , [df_mean , df_std]


def create_historical_dataset(df : pd.DataFrame, window_size : int,
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
  # Ensure the input DataFrame has the required columns
  if 'power' not in df.columns:
      raise ValueError("Input DataFrame must have a 'power' column.")

  # Create empty DataFrame to store historical df
  historical_df = pd.DataFrame()

  # Include original features
  for col in df.columns:
    historical_df[col] = df[col]

  # exclude the columns that are not going to be shiftted
  df.drop(columns=exclude_columns, axis=1, inplace=True)
  
  # Create lag features for the specified window size
  for i in range(1, window_size + 1):
    for col in df.columns:
      historical_df[f'{col}_lag_{i}'] = df[col].shift(i)

  # Add the target variable (power) shifted by the window size
  historical_df['target_power'] = df['power'].shift(-1)

  # Drop rows with NaN values introduced by shifting
  historical_df = historical_df.dropna().reset_index(drop=True)

  return historical_df
