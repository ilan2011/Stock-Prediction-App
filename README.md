# Stock-Prediction-App

This repository contains the implementation of a simple stock price predictor using Streamlit, Keras for building a LSTM model, and yfinance for fetching historical stock data.

The script uses Long Short Term Memory (LSTM), a type of Recurrent Neural Network (RNN), to predict the closing price of a chosen stock using past 60 days of stock data.

How to use
Input a valid stock symbol in the text box.
The script downloads the historical data for that stock, performs preprocessing and trains a LSTM model.
The script then makes predictions on the testing data and displays the Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE) of the predictions.
The actual and predicted closing prices are also displayed in a graph.
Finally, the model predicts the price for the next trading day and displays it alongside the past 5 days' prices.
