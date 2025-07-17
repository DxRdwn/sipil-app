# dnn_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

kategori = {
    "Bukan Pondasi": [
        "1,2D+1,6L", "1,4D", "1.2D+Ev+Eh+L(x)rs", "0.9D-Ev+Eh(x)rs", "1.2D+Ev+Eh+L(y)rs", "0.9D-Ev+Eh(y)rs",
        "1.2D+Ev+Emh+L(x)rs", "1.2D+Ev+Emh+L(y)rs", "0.9D-Ev+Emh(x)rs", "0.9D-Ev+Emh(y)rs"
    ],
    "Layan": ["D+L"],
    "Nominal": [
        "1.0D+0.7Ev+0.7Eh(x)rs", "1.0D+0.7Ev+0.7Eh(y)rs", "1.0D+0.525Ev+0.525Eh+0.75L(x)rs",
        "1.0D+0.525Ev+0.525Eh+0.75L(y)rs", "0.6D-0.7Ev+0.7Eh(x)rs", "0.6D-0.7Ev+0.7Eh(y)rs"
    ],
    "Kuat": [
        "1.0D+0.7Ev+0.7Emh(y)rs", "1.0D+0.7Ev+0.7Emh(x)rs", "1.0D+0.525Ev+0.525Emh+0.75L(x)rs",
        "1.0D+0.525Ev+0.525Emh+0.75L(y)rs", "0.6D-0.7Ev+0.7Emh(x)rs", "0.6D-0.7Ev+0.7Emh(y)rs"
    ],
}


def klasifikasikan_kasus(kasus):
    for key, values in kategori.items():
        if kasus in values:
            return key
    return "Tidak Diketahui"

def run_dnn_model(file_path):
    df = pd.read_excel(file_path, sheet_name='Joint Reactions', header=None)
    header = df.iloc[1]
    data = df.iloc[3:]
    data.columns = header
    data.reset_index(drop=True, inplace=True)

    df2 = data[['Joint', 'OutputCase', 'CaseType', 'StepType', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3']].copy()
    df2['OutputCase'] = df2['OutputCase'].str.replace(' ', '', regex=False)
    df2['Kategori'] = df2['OutputCase'].apply(klasifikasikan_kasus)

    df2['Joint'] = pd.to_numeric(df2['Joint'], errors='coerce')
    for col in ['F1', 'F2', 'F3', 'M1', 'M2', 'M3']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')

    df2['F3_Absolut'] = df2['F3'].abs()
    df2['M1_Absolut'] = df2['M1'].abs()
    df2['M2_Absolut'] = df2['M2'].abs()
    df2['F3_Max_Per_Category'] = df2.groupby(['Joint', 'Kategori'])['F3_Absolut'].transform('max')
    df2 = df2.sort_values(by=['Joint', 'Kategori', 'F3'], ascending=[True, True, False])
    df2['M1_divider'] = df2.groupby(['Joint', 'Kategori'])['M1_Absolut'].transform('first')
    df2['M2_divider'] = df2.groupby(['Joint', 'Kategori'])['M2_Absolut'].transform('first')
    df2['F3_ratio'] = ((df2['F3_Absolut'] / df2['F3_Max_Per_Category']) * 100).round(2)
    df2['M1_ratio'] = ((df2['M1_Absolut'] / df2['M1_divider']) * 100).round(2)
    df2['M2_ratio'] = ((df2['M2_Absolut'] / df2['M2_divider']) * 100).round(2)

    df = df2.dropna(subset=['F3_ratio', 'F1', 'F2', 'M1', 'M2', 'M3', 'M1_ratio', 'M2_ratio'])
    numerik_cols = ['F1', 'F2', 'M1', 'M2', 'M3', 'M1_ratio', 'M2_ratio', 'F3_ratio']
    df_encoded = pd.get_dummies(df[['OutputCase', 'CaseType', 'StepType', 'Kategori']], drop_first=True)

    X = pd.concat([df[numerik_cols], df_encoded], axis=1)
    y = df['F3_ratio']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=0)

    y_pred = model.predict(X_test).flatten()
    mae = np.mean(np.abs(y_pred - y_test))
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    akurasi = np.mean(np.abs(y_pred - y_test) <= 5) * 100

    return {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4),
        "akurasi_5%": round(akurasi, 2)
    }
