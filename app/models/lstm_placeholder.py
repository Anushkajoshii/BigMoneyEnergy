# app/models/lstm_placeholder.py
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    tf = None

def train_lstm(timeseries_df):
    """
    Placeholder: Train an LSTM on user's timeseries of monthly savings.
    timeseries_df should be a DataFrame with column 'monthly_savings' and a time index.
    This function is optional (requires tensorflow).
    """
    if tf is None:
        raise RuntimeError("TensorFlow not installed. Install tensorflow to use LSTM path.")
    # Very minimal placeholder: create simple model (real training requires scaling, windows, etc.)
    model = models.Sequential([
        layers.Input(shape=(12, 1)),
        layers.LSTM(32, activation="tanh"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
