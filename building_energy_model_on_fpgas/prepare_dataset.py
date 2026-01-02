# Dataset preprocessing logic will be added here
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_energy_dataset(csv_path):
    """
    Prepare building energy dataset for anomaly detection.
    Anomaly = unusually high energy usage.
    """

    # Load dataset
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month

    # Target energy column (adjust name if needed)
    energy_col = 'energy_usage'

    # Create anomaly labels
    mean_energy = df[energy_col].mean()
    std_energy = df[energy_col].std()

    df['anomaly'] = (df[energy_col] > (mean_energy + 2 * std_energy)).astype(int)

    # Features used by FPGA model
    features = [
        energy_col,
        'hour',
        'day',
        'month'
    ]

    X = df[features].values
    y = df['anomaly'].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Save for FPGA inference
    np.save('package/X_test.npy', X_test)
    np.save('package/y_test.npy', y_test)

    return X_train, X_test, y_train, y_test
