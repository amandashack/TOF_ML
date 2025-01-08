import numpy as np
import pandas as pd


def add_interaction_terms(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Example: log2(x_col), keep y_col as-is, then add x*y, x^2, y^2
    Returns a modified DataFrame with new columns appended:
      'log_tof', 'retardation', 'tof_times_ret', 'tof_squared', 'ret_squared'
    Adjust names & logic as needed.
    """
    # We'll rename x_col -> e.g., "tof", y_col -> "retardation"
    df[f"log_{x_col}"] = np.log2(df[x_col])  # log2 of the first col
    df[f"{x_col}_times_{y_col}"] = df[x_col] * df[y_col]
    df[f"{x_col}^2"] = df[x_col] ** 2
    df[f"{y_col}^2"] = df[y_col] ** 2
    return df


def preprocess_data_array(data: np.ndarray):
    """
    Example from your snippet: data shape => (N, 3) or more
    Indices: [0] = tof, [1] = retardation, [2] = KE, ...
    We'll log2(tof), keep ret, add interactions, log(KE).

    Returns X_scaled, y_scaled, scaler_X, scaler_y
    """
    from sklearn.preprocessing import StandardScaler

    # 1. Extract columns
    tof = data[:, 0]  # e.g. data[:,0] = 'tof'
    ret = data[:, 1]  # e.g. data[:,1] = 'retardation'

    # 2. log2(tof)
    x1 = np.log2(tof)
    x2 = ret

    # 3. Interaction terms
    #    x1*x2, x1^2, x2^2
    inter = np.column_stack([
        x1 * x2,
        x1 ** 2,
        x2 ** 2
    ])

    # Combine into one input array
    # final shape => (N, 5): [x1, x2, x1*x2, x1^2, x2^2]
    X = np.column_stack([x1, x2, inter])

    # 4. y => log( KE ) if data[:,2] is KE
    KE = data[:, 2]
    y = np.log(KE).reshape(-1, 1)

    # 5. Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y).flatten()

    return X_scaled, y_scaled, scaler_X, scaler_y
