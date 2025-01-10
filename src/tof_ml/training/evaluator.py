# src/training/evaluator.py
from src.utils.metrics import calculate_mse, calculate_mae, calculate_r2

class Evaluator:
    def evaluate(self, y_true, y_pred):
        return {
            "mse": calculate_mse(y_true, y_pred),
            "mae": calculate_mae(y_true, y_pred),
            "r2": calculate_r2(y_true, y_pred)
        }
