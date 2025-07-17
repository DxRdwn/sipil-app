import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def run_cnn_model(file_path):
    # Load dan persiapkan data
    data = pd.read_excel(file_path, sheet_name='Joint Reactions', header=None)
    new_header = data.iloc[1]
    data_new = data[2:].copy()
    data_new.columns = new_header
    data_new = data_new.drop(2)

    kolom = ['Joint', 'OutputCase', 'CaseType', 'StepType', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3']
    valid_columns = [col for col in kolom if col in data_new.columns]
    df_filtered = data_new[valid_columns].copy()

    if 'StepType' in df_filtered.columns:
        df_filtered['StepType'] = df_filtered['StepType'].fillna('Beban layan')

    le = LabelEncoder()
    for col in ['CaseType', 'StepType', 'OutputCase', 'Joint']:
        if col in df_filtered.columns:
            df_filtered[col] = le.fit_transform(df_filtered[col])

    X = df_filtered.drop(columns=['StepType'])
    y = df_filtered['StepType']

    X = X.astype(float)
    y = np.array(y, dtype=int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Model CNN
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(np.unique(y)), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
              verbose=0, callbacks=[early_stopping])

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return {
        "accuracy": round(float(accuracy) * 100, 2),
        "loss": round(float(loss), 4),
        "status": "CNN model executed successfully"
    }
