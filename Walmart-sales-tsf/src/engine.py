from ml_pipeline import preprocess, train_model, Utils,deploy

import pandas as pd



# Input 0 to train the model and 1 to deploy the model
val = int(input("Train - 0\nDeploy - 1\nEnter your value: "))
if val == 0:
    # Loading the data
    train = Utils.read_data('../data/train.csv').drop_duplicates()
    feature = Utils.read_data('../data/features.csv')
    stores = Utils.read_data('../data/stores.csv')
    test = Utils.read_data('../data/test.csv')
    print("Data is loaded")

    print("Data preprocessing started")

    agg_train_col = {"Weekly_Sales": sum, "IsHoliday": "first",
                     "Type": "first", "Size": "first", "Temperature": "first",
                     "Fuel_Price": "first", "MarkDown1": "first", "MarkDown2": "first",
                     "MarkDown3": "first", "MarkDown4": "first", "MarkDown5": "first",
                     "CPI": "first", "Unemployment": "first"}

    agg_test_col = {"IsHoliday": "first",
                    "Type": "first", "Size": "first", "Temperature": "first",
                    "Fuel_Price": "first", "MarkDown1": "first", "MarkDown2": "first",
                    "MarkDown3": "first", "MarkDown4": "first", "MarkDown5": "first",
                    "CPI": "first", "Unemployment": "first"}
    # Merging the data frames
    train_m1 = Utils.merge_dataframes(train, stores)
    train_data = Utils.merge_dataframes(train_m1, feature)


    # Grouping the dataframe by date
    train_data = preprocess.group_data(train_data, "Date", agg_train_col)

    # Imputing missing value by 0
    train_data = preprocess.impute(train_data)

    # Replacing the outliers in the target variable
    train_data.Weekly_Sales = preprocess.replace_outliers(train_data.Weekly_Sales, 2000000, 2000000)

    # Adding new columns in data for year, month, and day
    new_col = ['Date_year', 'Date_month', 'Date_day', 'Date_dayofweek']
    date_col = 'Date'
    train_data = preprocess.separate_date_col(train_data, date_col, new_col)

    # Mapping
    type_mapping = {"A": 1, "B": 2, "C": 3}
    train_data = preprocess.map(train_data, 'Type', type_mapping)

    holiday_type_mapping = {False: 0, True: 1}
    train_data = preprocess.map(train_data, 'IsHoliday', holiday_type_mapping)

    # Dropping the features
    features_drop = ['Unemployment', 'CPI', 'MarkDown5']
    train_data = preprocess.drop_col(train_data, features_drop)

    # Changing the type of date column
    train_data = preprocess.change_type(train_data, 'Date', 'datetime64[ns]')

    # Renaming the columns
    rename_col = {'Date': 'ds', 'Weekly_Sales': 'y'}
    train_data = preprocess.rename_column(train_data, rename_col)

    # Selecting specific features in the dataset
    select_col = ['ds', 'Temperature', 'Fuel_Price', 'IsHoliday', 'y']
    final_train_data = preprocess.select_features(train_data, select_col)

    # Sorting the data for model training
    final_data = preprocess.sort_data(final_train_data, 'ds')
    print("Data preprocessing ended")

    # Model Training
    print("Model training has started")

     # Neural prophet model training
    future_regressors = ['Temperature', 'Fuel_Price', 'IsHoliday']
    prophet_model = train_model.train_neural_prophet(final_data, future_regressors)
    p_path = '../output/prophet_model.pkl'
    Utils.save_model(prophet_model, p_path)
    print('Neural Prophet model is saved as a pkl file in ' + str('../output/prophet_model.pkl'))

else:
    p_path = '../output/prophet_model.pkl'
    deploy.init(p_path)