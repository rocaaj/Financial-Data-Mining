import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf

# =======================
# data preprocessing
# =======================
def load_and_preprocess_data(filepath, time_steps=12):
    df = pd.read_csv(filepath)

    # Fix: Remove any rows where 'datetime' is literally the string 'datetime'
    df = df[df["datetime"] != "datetime"]

    # Try parsing datetime, drop if unparseable
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", format="%Y-%m-%d")
    df.dropna(subset=["datetime"], inplace=True)

    df = df.sort_values(["stockCodeCompany", "datetime"])

    df_pivot = df.pivot(index="datetime", columns="stockCodeCompany", values="closeValueStock")
    df_pivot.fillna(method="ffill", inplace=True)
    df_pivot.dropna(axis=1, inplace=True)

    all_X_train, all_y_train, all_X_test, all_y_test = [], [], [], []
    scaler = MinMaxScaler()

    for col in df_pivot.columns:
        series = df_pivot[col].values.reshape(-1, 1)
        series_scaled = scaler.fit_transform(series)
        X_seq, y_seq = create_sequences(series_scaled, time_steps)

        split = int(len(X_seq) * 0.8)
        all_X_train.append(X_seq[:split])
        all_y_train.append(y_seq[:split])
        all_X_test.append(X_seq[split:])
        all_y_test.append(y_seq[split:])

    X_train = np.vstack(all_X_train)
    y_train = np.vstack(all_y_train)
    X_test = np.vstack(all_X_test)
    y_test = np.vstack(all_y_test)

    return X_train, y_train, X_test, y_test, scaler
# Create time-series sequences
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# =======================
# model definition
# =======================
def build_bilstm_model(time_steps, num_features=1):
    model = Sequential()
    model.add(Bidirectional(LSTM(32), input_shape=(time_steps, num_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# =======================
# evaluation
# =======================
def evaluate_model(model, X_test, y_test, scaler):
    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)

    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.3f}")

    output_dir = '/Users/anthonyroca/csc_373/extra_credit/output/StockForecasting'
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"R^2: {r2:.3f}\n")
    print(f"[INFO] Evaluation metrics saved to {metrics_path}")

    return y_test_inv, y_pred_inv

# =======================
# prediction plot
# =======================
def plot_predictions(y_test_inv, y_pred_inv):
    output_dir = '/Users/anthonyroca/csc_373/extra_credit/output/StockForecasting'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual')
    plt.plot(y_pred_inv, label='Predicted')
    plt.title('BiLSTM Stock Price Forecast')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'predictions.png')
    plt.savefig(plot_path)
    print(f"[INFO] Predictions plot saved to {plot_path}")

# =======================
# residuals
# =======================
def plot_residuals(y_test_inv, y_pred_inv):
    residuals = y_test_inv.flatten() - y_pred_inv.flatten()
    output_dir = '/Users/anthonyroca/csc_373/extra_credit/output/StockForecasting'
    plt.figure(figsize=(10, 4))
    plt.plot(residuals, color='red')
    plt.title('Residuals (Actual - Predicted)')
    plt.xlabel('Time')
    plt.ylabel('Residual Price')
    plt.tight_layout()
    residual_path = os.path.join(output_dir, 'residuals.png')
    plt.savefig(residual_path)
    print(f"[INFO] Residuals plot saved to {residual_path}")

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Main
TIME_STEPS = 12
if __name__ == '__main__':
    filepath = '/Users/anthonyroca/csc_373/extra_credit/data/csvs/ml_stocks.csv'
    print("[INFO] Loading and preprocessing data...")

    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(filepath, time_steps=TIME_STEPS)
    print(f"[INFO] Data loaded. Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    print("[INFO] Building BiLSTM model...")
    model = build_bilstm_model(time_steps=TIME_STEPS)
    print("[INFO] Model built.")

    print("[INFO] Starting training...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1, verbose=1, callbacks=[early_stop])
    print("[INFO] Training complete.")

    output_dir = '/Users/anthonyroca/csc_373/extra_credit/output/StockForecasting'
    model_path = os.path.join(output_dir, 'bilstm_model.h5')
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

    print("[INFO] Evaluating model...")
    y_test_inv, y_pred_inv = evaluate_model(model, X_test, y_test, scaler)
    plot_predictions(y_test_inv, y_pred_inv)

    print("[INFO] Plotting residuals...")
    plot_residuals(y_test_inv, y_pred_inv)
