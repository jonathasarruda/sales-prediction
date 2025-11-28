import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

def train_arima(df, order=(1,1,1), output_path="models/arima_model.pkl"):
    """
    Train ARIMA model on sales data.
    Treina modelo ARIMA nos dados de vendas.
    """
    model = ARIMA(df['sales'], order=order)
    fitted = model.fit()
    joblib.dump(fitted, output_path)
    return fitted

def train_prophet(df, output_path="models/prophet_model.pkl"):
    """
    Train Prophet model on sales data.
    Treina modelo Prophet nos dados de vendas.
    """
    prophet_df = df[['date','sales']].rename(columns={'date':'ds','sales':'y'})
    model = Prophet()
    model.fit(prophet_df)
    joblib.dump(model, output_path)
    return model

def train_lstm(df, output_path="models/lstm_model.h5"):
    """
    Train LSTM model on sales data.
    Treina modelo LSTM nos dados de vendas.
    """
    # Converte série em sequências para treino
    series = df['sales'].values.reshape(-1,1)
    X, y = [], []
    for i in range(len(series)-10):
        X.append(series[i:i+10])
        y.append(series[i+10])
    X, y = np.array(X), np.array(y)

    # Ajusta formato para LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Define arquitetura da rede
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Treina modelo
    model.fit(X, y, epochs=10, verbose=0)
    model.save(output_path)
    return model
