import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# Function to get the DataFrame and extract the features
def feature_extractor(df: pd.DataFrame)-> pd.DataFrame:
    """
    This function gets a dataframe with 3 columns: being, Time,Power,
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
    This function gets a data frame and a column name and does the-
    normalization operation.

    Args:
        df : its a panda.DataFrame object.
        col_name : its a string and the name of the desired column to noramlize.
    
    Returns: A pandas.DataFrame as same as the input data frame, but with normalized column and 
    the std and mean values as a list in case, there is an unromalization.
    """
    df_mean = df[col_name].mean()
    df_std = df[col_name].std()
    df.loc[:, col_name] = (df[col_name] - df_mean) / df_std
    print(f"\nNormalization is done for column : {col_name}")
    return df , [df_mean , df_std]

