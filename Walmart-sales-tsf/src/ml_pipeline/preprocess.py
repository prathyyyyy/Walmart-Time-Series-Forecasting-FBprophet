import numpy as np
import pandas as pd


# Function to replace outliers

def replace_outliers(df_col_name,outlier_bound,replace_by):
    df_col_name = np.where(df_col_name > outlier_bound, replace_by, df_col_name)
    return  df_col_name

def separate_date_col(df,date_col,new_col_name):
    for col in new_col_name:
        if col in df.columns:
            raise KeyError(f"{col} - Column already exists, enter a different value")

        df[date_col] = pd.to_datetime(df[date_col])
        df[new_col_name[0]] = df[date_col].dt.year
        df[new_col_name[1]] = df[date_col].dt.month
        df[new_col_name[2]] = df[date_col].dt.day
        df[new_col_name[3]] = df[date_col].dt.dayofweek

        return df

def map(df,col,mapping):
    if col in df.columns:
        df[col] = df[col].map(mapping)
    else:
        raise KeyError(f"{col} - Column doesnt exist in dataframe ")
    return  df

def drop_col(df,col_list):
    for col in col_list:
        if col not in df.columns:
            raise KeyError(f"{col} - Column Doesnt exist in dataframe")
    df = df.drop(col_list,axis = 1)
    return df

def rename_column(df, rename_col):
    df = df.rename(columns = rename_col)
    return df


def select_features(df, features):
    df = df[features]
    return features

def sort_data(df, by_col):
    if by_col in df.columns:
        df = df.loc[:, :]
        df = df.sort_values(by=[by_col])
    else:
        raise KeyError(f"{by_col} column does not exist")
    return df

def change_type(df,col, data_type):
    if col in df.columns:
        df[col] = df[col].astype(data_type)
    else:
        raise KeyError(f"{col} - Column doesnt exist")

    return df

def impute(df, value= 0):
    df = df.fillna(value)
    return  df

def group_data(df,group_col,agg_col):
    if group_col in df.columns:
        if type(agg_col) is dict:
            df = df.groupby([group_col]).agg(agg_col).reset_index()
        else:
            print("agg_col is a dict")
    else:
        raise KeyError(f"{group_col} column doest not exist")

    return df