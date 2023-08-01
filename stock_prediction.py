import streamlit as st
import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


st.title('Stock Price Predictor by Ilan Breines')

st.write('''
**Disclaimer**: For educational use only. Always consult with a certified financial advisor before making investment decisions. 
''')

# Let the user input the stock name
stock = st.text_input('Enter a stock name:')


if stock:  # Proceed only if stock is not empty
    df = yf.download(stock, start = '2015-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))

    if df.empty:
        st.write(f'Could not find any data for the stock symbol: {stock}. Please enter a valid stock symbol.')

    else:
        stock_info = yf.Ticker(stock)
        stock_name = stock_info.info['longName']
        stock_symbol = stock_info.info['symbol']
        st.subheader(f"Selected Stock: {stock_name} ({stock_symbol})")

        #Show the data
        st.write(df)

        #Plotting
        st.subheader('Closing Price History')
        fig, ax = plt.subplots(figsize=(16,8))

        # Get the last month's data
        last_month_data = df.loc[pd.to_datetime('today') - pd.DateOffset(days=60):]

        ax.plot(last_month_data['Close'], label='Close Price history')
        st.pyplot(fig)

        

        #Create the new dataframe with only the close column
        data=df.filter(['Close'])
        #Convert the dataframe to a numpy array
        dataset=data.values
        #Get the number of rows to train the model on
        training_data_len=math.ceil(len(dataset)*0.8)

        #Scale the data
        scaler=MinMaxScaler(feature_range=(0,1))
        scaled_data=scaler.fit_transform(dataset)

        #Create the taining data set
        #Create the scaled training data set
        train_data=scaled_data[0:training_data_len,:]
        #Split the data into x_train and y_train datasets
        x_train=[]
        y_train=[]

        for i in range(60,len(train_data)):
            x_train.append(train_data[i-60:i,0])
            y_train.append(train_data[i,0])
            if i<=60:
                print(x_train)
                print(y_train)
                print()
        
        #Convert the x_train and y_train to numpy arrays
        x_train, y_train=np.array(x_train),np.array(y_train)
        
        x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        x_train.shape

        #Build the LSTM model
        model=Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
        model.add(LSTM(50,return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        #Compile the model
        model.compile(optimizer='adam',loss='mean_squared_error')

        #########################

        st.subheader('Model Training...')
        model.fit(x_train,y_train,batch_size=1,epochs=1)

        #Create the testing dataset
        #Create a new array containing scaled values from the index 1543 to 2003
        test_data=scaled_data[training_data_len-60:,:]
        #Create the dataset x_test and y_test
        x_test=[]
        y_test=dataset[training_data_len: ,:]
        for i in range(60,len(test_data)):
            x_test.append(test_data[i-60:i,0])
        
        #Convert the data to a numpy array
        x_test=np.array(x_test)
        #Reshape the data
        x_test=np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

        #Get the model predicted price values
        predictions=model.predict(x_test)
        predictions=scaler.inverse_transform(predictions)

        #Get the root mean square error(RMSE)
        rmse=np.sqrt(np.mean(predictions-y_test)**2)
        st.subheader(f'Root Mean Squared Error: {rmse}')

        mae = mean_absolute_error(y_test, predictions)
        st.subheader(f'Mean Absolute Error: {mae}')

        def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        mape = mean_absolute_percentage_error(y_test, predictions)
        mape_rounded = round(mape, 2)
        st.subheader(f'Mean Absolute Percentage Error: {mape_rounded}%')

        # Visualize predictions
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        st.subheader('Model Predictions')
        fig2, ax2 = plt.subplots(figsize=(16,8))
        ax2.plot(train['Close'])
        ax2.plot(valid[['Close','Predictions']])
        st.pyplot(fig2)

        # Show the valid and predicted prices
        st.subheader('Validation Results')
        st.write(valid)

        # Predict the future price
        apple_quote = yf.download(stock, start = '2015-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))
        new_df = apple_quote.filter(['Close'])
        last_60_days = new_df[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)

        pred_price_str = "{:.2f}".format(pred_price[0][0])
        st.subheader('Predicted Future Price (next trading day): $' + pred_price_str)


        

        # Get the latest prices
        end_date = pd.to_datetime('today')
        start_date = end_date - pd.DateOffset(days=6)  # Get data from 5 days ago
        latest_data = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        st.subheader('Latest Prices (past 5 days)')
        st.write(latest_data['Close'])

        # Create a new pandas Series with the predicted price
        predicted_price_series = pd.Series(pred_price[0][0], index=[end_date + pd.DateOffset(days=1)])

        # Concatenate the latest prices with the predicted price
        latest_prices_with_prediction = pd.concat([latest_data['Close'], predicted_price_series])

        # Plot the latest prices and the predicted price
        st.subheader('Latest Prices and Predicted Future Price')
        fig3, ax3 = plt.subplots(figsize=(16,8))
        ax3.plot(latest_data['Close'].index, latest_data['Close'], color='blue', label='Latest Prices')
        ax3.plot(latest_prices_with_prediction.index[-2:], latest_prices_with_prediction[-2:], color='red', label='Predicted Price')
        ax3.legend()
        st.pyplot(fig3)


