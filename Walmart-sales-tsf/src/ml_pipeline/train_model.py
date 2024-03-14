import pandas as pd
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_percentage_error

def train_neural_prophet(data, future_reg):
    test_length = 20
    df_train = data.iloc[:-test_length]
    df_test = data.iloc[-test_length]

    # Create a NeuralProphet model
    model = NeuralProphet(loss_func='MSE', n_changepoints=2, seasonality_mode='additive')

    # Add future regressors
    for col in future_reg:
        model.add_future_regressor(col)

    # Fit the model
    metrics = model.fit(df_train, freq="W")

    # Make future predictions
    future_df = model.make_future_dataframe(df_test, periods=test_length, n_historic_predictions=len(df_test),
                                            regressors_df=df_test)
    forecast = model.predict(future_df)

    # Calculate Mean Absolute Percentage Error for evaluation
    mape = mean_absolute_percentage_error(df_test['y'], forecast.iloc[-test_length:]['yhat1'])
    print(f"Mean absolute percentage error for the fitted NeuralProphet model is {mape:.4f}")
    return model