import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def run_lstm_model(df2):
    kolom = ['Joint', 'OutputCase', 'CaseType', 'StepType', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3']
    valid_columns = [col for col in kolom if col in df2.columns]
    features = df2[valid_columns].copy()

    # Label encoding untuk kolom kategori
    label_encoders = {}
    kategori_kolom = features.select_dtypes(include=['object']).columns

    for col in kategori_kolom:
        features[col] = features[col].astype(str)
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])
        label_encoders[col] = le

    # Scaling
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # LSTM preparation
    n_past = 5
    X, y = [], []
    for i in range(n_past, len(features_scaled)):
        X.append(features_scaled[i - n_past:i])
        y.append(features_scaled[i])
    X, y = np.array(X), np.array(y)

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=2)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(X.shape[2])
        ])
        model.compile(optimizer='adam', loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test)

    # Thresholding untuk deteksi noise
    errors = np.abs(predictions_rescaled - y_test_rescaled)
    threshold = np.percentile(np.max(errors, axis=1), 95)

    true_labels = np.zeros(len(y_test))
    predicted_labels = np.zeros(len(y_test))
    noise_idx = np.where(np.max(errors, axis=1) > threshold)[0]
    true_labels[noise_idx] = 1
    predicted_labels[noise_idx] = 1

    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision": float(precision_score(true_labels, predicted_labels, zero_division=1)),
        "recall": float(recall_score(true_labels, predicted_labels, zero_division=1)),
        "f1_score": float(f1_score(true_labels, predicted_labels, zero_division=1))
    }
