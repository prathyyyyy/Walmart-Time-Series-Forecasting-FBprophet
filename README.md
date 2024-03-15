# Walmart Time Series Forecasting Project using Meta's Neural Prophet

## 1. Introduction: 
This project aims to predict future demand/sales using historical data and related features from Walmart store sales. Time series forecasting techniques are applied to analyze the data and forecast future sales using Neural Prophet, which is utilized for forecasting demand.

![newplot (5)](https://github.com/prathyyyyy/Walmart-Time-Series-Forecasting-FBprophet/assets/97932221/b6c2f286-590a-4627-b44e-ee8660d850b1)

## 2. Aim:
To predict future demand/sales using historical data and other related features.

## 3. Tech Stack:
Language: Python
Libraries: Greykite, Neural Prophet, Sci-kit Learn, Pandas, Sweetviz, PytimeTk, Datetime, Plotly, NumPy

## 4. Approach:

1. Exploratory Data Analysis (EDA)
2. Inference about features
3. Data visualization using Pandas Profiling
4. Data cleaning (outlier/missing values)
5. Missing value imputation
6. Outlier detection
7. Feature Engineering
  - Extracting day, month, and year from date
  - Mapping
8. Time series component analysis
  - Trend
  ![newplot (6)](https://github.com/prathyyyyy/Walmart-Time-Series-Forecasting-FBprophet/assets/97932221/728dde25-6f79-4904-a4ba-e24ffe64a58f)

  - Seasonality
  ![newplot (7)](https://github.com/prathyyyyy/Walmart-Time-Series-Forecasting-FBprophet/assets/97932221/ac5a94ee-1ff7-4533-9967-716a23e76bdd)

9. Model building on training data
  - Neural Prophet
10. Model validation
  - Mean Absolute Percent Error
  - Root Mean Squared Error (RMSE)
11. Forecasting using trained models

## 5. Model Validation Metrics

After training and validating the forecasting models, the following evaluation metrics were obtained:

- #### Mean Absolute Percent Error (MAPE): 0.039
- #### Root Mean Squared Error (RMSE): 80742.23

![newplot (8)](https://github.com/prathyyyyy/Walmart-Time-Series-Forecasting-FBprophet/assets/97932221/198bd78c-863b-4ecb-960b-7c5c0b6f21e7)

These metrics provide insights into the accuracy of the forecasting models in predicting future demand/sales based on historical data. Lower values of MAPE and RMSE indicate better performance of the models in capturing the underlying patterns and trends in the data.
