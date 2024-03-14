import pickle
import pandas as pd

def save_model(model,path):
    with open(path, "wb") as output:
        pickle.dump(model,output, pickle.HIGHEST_PROTOCOL)

def load_model(path):
    with open(path, "rb") as model:
        ml_model = pickle.load(model)
        return ml_model


def read_data(file_path, **kwargs):
    raw_data = pd.read_csv(file_path,**kwargs)
    return raw_data

def merge_dataframes(df1,df2):
    combined_data = pd.merge(df1,df2)
    return combined_data