# src/tof_ml/transforms/common_transforms.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from .base_transform import BaseTransform
from .transform_pipeline import register_transform


@register_transform
class ColumnSelector(BaseTransform):
    """Select specific columns from the data"""

    def __init__(self, columns: List[Union[int, str]], name: str = None):
        super().__init__(name=name, columns=columns)
        self.column_indices = None

    def fit(self, data: np.ndarray, **kwargs) -> 'ColumnSelector':
        """Determine column indices if column names were provided"""
        if all(isinstance(col, int) for col in self.params['columns']):
            self.column_indices = self.params['columns']
        else:
            # Assume column_mapping is provided in kwargs
            column_mapping = kwargs.get('column_mapping', {})

            if not column_mapping:
                raise ValueError("column_mapping must be provided when using column names")

            # Resolve column names to indices, with validation
            self.column_indices = []
            for col in self.params['columns']:
                if isinstance(col, int):
                    if col < data.shape[1]:
                        self.column_indices.append(col)
                    else:
                        raise ValueError(f"Column index {col} out of range for data with {data.shape[1]} columns")
                elif col in column_mapping:
                    self.column_indices.append(column_mapping[col])
                else:
                    raise ValueError(
                        f"Column '{col}' not found in column_mapping. Available columns: {list(column_mapping.keys())}")

        # Validate indices against data shape
        for idx in self.column_indices:
            if idx >= data.shape[1]:
                raise ValueError(f"Column index {idx} out of range for data with {data.shape[1]} columns")

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Select the specified columns"""
        if not self._is_fitted:
            raise ValueError("Transform not fitted. Call fit() first.")
        return data[:, self.column_indices]


@register_transform
class LogTransform(BaseTransform):
    """Apply log transformation to specified columns"""

    def __init__(self, columns: List[Union[int, str]], base: float = 2.0,
                 offset: float = 1e-10, name: str = None):
        super().__init__(name=name, columns=columns, base=base, offset=offset)
        self.column_indices = None

    def fit(self, data: np.ndarray, **kwargs) -> 'LogTransform':
        """Determine column indices if column names were provided"""
        if all(isinstance(col, int) for col in self.params['columns']):
            self.column_indices = self.params['columns']
        else:
            # Assume column_mapping is provided in kwargs
            column_mapping = kwargs.get('column_mapping', {})

            if not column_mapping:
                raise ValueError("column_mapping must be provided when using column names")

            # Resolve column names to indices, with validation
            self.column_indices = []
            for col in self.params['columns']:
                if isinstance(col, int):
                    if col < data.shape[1]:
                        self.column_indices.append(col)
                    else:
                        raise ValueError(f"Column index {col} out of range for data with {data.shape[1]} columns")
                elif col in column_mapping:
                    self.column_indices.append(column_mapping[col])
                else:
                    raise ValueError(
                        f"Column '{col}' not found in column_mapping. Available columns: {list(column_mapping.keys())}")

        # Validate indices against data shape
        for idx in self.column_indices:
            if idx >= data.shape[1]:
                raise ValueError(f"Column index {idx} out of range for data with {data.shape[1]} columns")

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply log transformation to the specified columns"""
        if not self._is_fitted:
            raise ValueError("Transform not fitted. Call fit() first.")

        result = data.copy()
        base = self.params['base']
        offset = self.params['offset']

        for col_idx in self.column_indices:
            # Add a safer approach with explicit handling of invalid values
            values = result[:, col_idx]
            valid_mask = values + offset > 0  # Only apply log to positive values

            # Apply log transform only to valid values
            result[valid_mask, col_idx] = np.log(values[valid_mask] + offset) / np.log(base)

            # For invalid values, set to a minimum value (log of smallest valid value)
            if not np.all(valid_mask):
                min_valid = np.min(values[valid_mask]) if np.any(valid_mask) else offset
                min_log_value = np.log(min_valid + offset) / np.log(base)
                result[~valid_mask, col_idx] = min_log_value

                # Log warning
                n_invalid = np.sum(~valid_mask)
                self._metadata['n_invalid_values'] = int(n_invalid)
                self._metadata['invalid_replacement_value'] = float(min_log_value)
                import logging
                logger = logging.getLogger("transform")
                logger.warning(
                    f"LogTransform: {n_invalid} negative or zero values replaced with log({min_valid + offset})")

        return result


@register_transform
class StandardScaler(BaseTransform):
    """Standardize features by removing the mean and scaling to unit variance"""

    def __init__(self, columns: List[Union[int, str]] = None, name: str = None):
        super().__init__(name=name, columns=columns)
        self.column_indices = None
        self.scalers = {}

    def fit(self, data: np.ndarray, **kwargs) -> 'StandardScaler':
        """Fit the scaler to the data"""
        # Determine which columns to scale
        if self.params['columns'] is None:
            self.column_indices = list(range(data.shape[1]))
        elif all(isinstance(col, int) for col in self.params['columns']):
            self.column_indices = self.params['columns']
        else:
            # Assume column_mapping is provided in kwargs
            column_mapping = kwargs.get('column_mapping', {})

            if not column_mapping:
                raise ValueError("column_mapping must be provided when using column names")

            # Resolve column names to indices, with validation
            self.column_indices = []
            for col in self.params['columns']:
                if isinstance(col, int):
                    if col < data.shape[1]:
                        self.column_indices.append(col)
                    else:
                        raise ValueError(f"Column index {col} out of range for data with {data.shape[1]} columns")
                elif col in column_mapping:
                    self.column_indices.append(column_mapping[col])
                else:
                    raise ValueError(
                        f"Column '{col}' not found in column_mapping. Available columns: {list(column_mapping.keys())}")

        # Validate indices against data shape
        for idx in self.column_indices:
            if idx >= data.shape[1]:
                raise ValueError(f"Column index {idx} out of range for data with {data.shape[1]} columns")

        # Fit a scaler for each column
        for col_idx in self.column_indices:
            # Ensure the column data is reshaped properly for scikit-learn
            col_data = data[:, col_idx].reshape(-1, 1)
            scaler = SklearnStandardScaler()
            scaler.fit(col_data)
            self.scalers[col_idx] = scaler

            # Store stats in metadata
            self._metadata[f'mean_{col_idx}'] = float(scaler.mean_[0])
            self._metadata[f'scale_{col_idx}'] = float(scaler.scale_[0])

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply standardization to the data"""
        if not self._is_fitted:
            raise ValueError("Transform not fitted. Call fit() first.")

        result = data.copy()

        for col_idx, scaler in self.scalers.items():
            # Ensure data is correctly shaped for scikit-learn
            col_data = data[:, col_idx].reshape(-1, 1)
            result[:, col_idx] = scaler.transform(col_data).flatten()

        return result


@register_transform
class MinMaxScaler(BaseTransform):
    """Scale features to a given range"""

    def __init__(self, columns: List[Union[int, str]] = None,
                 feature_range: Tuple[float, float] = (0, 1), name: str = None):
        super().__init__(name=name, columns=columns, feature_range=feature_range)
        self.column_indices = None
        self.scalers = {}

    def fit(self, data: np.ndarray, **kwargs) -> 'MinMaxScaler':
        """Fit the scaler to the data"""
        # Determine which columns to scale
        if self.params['columns'] is None:
            self.column_indices = list(range(data.shape[1]))
        elif all(isinstance(col, int) for col in self.params['columns']):
            self.column_indices = self.params['columns']
        else:
            # Assume column_mapping is provided in kwargs
            column_mapping = kwargs.get('column_mapping', {})

            if not column_mapping:
                raise ValueError("column_mapping must be provided when using column names")

            # Resolve column names to indices, with validation
            self.column_indices = []
            for col in self.params['columns']:
                if isinstance(col, int):
                    if col < data.shape[1]:
                        self.column_indices.append(col)
                    else:
                        raise ValueError(f"Column index {col} out of range for data with {data.shape[1]} columns")
                elif col in column_mapping:
                    self.column_indices.append(column_mapping[col])
                else:
                    raise ValueError(
                        f"Column '{col}' not found in column_mapping. Available columns: {list(column_mapping.keys())}")

        # Validate indices against data shape
        for idx in self.column_indices:
            if idx >= data.shape[1]:
                raise ValueError(f"Column index {idx} out of range for data with {data.shape[1]} columns")

        # Fit a scaler for each column
        for col_idx in self.column_indices:
            # Ensure the column data is reshaped properly for scikit-learn
            col_data = data[:, col_idx].reshape(-1, 1)
            scaler = SklearnMinMaxScaler(feature_range=self.params['feature_range'])
            scaler.fit(col_data)
            self.scalers[col_idx] = scaler

            # Store stats in metadata
            self._metadata[f'min_{col_idx}'] = float(scaler.data_min_[0])
            self._metadata[f'max_{col_idx}'] = float(scaler.data_max_[0])

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply min-max scaling to the data"""
        if not self._is_fitted:
            raise ValueError("Transform not fitted. Call fit() first.")

        result = data.copy()

        for col_idx, scaler in self.scalers.items():
            # Ensure data is correctly shaped for scikit-learn
            col_data = data[:, col_idx].reshape(-1, 1)
            result[:, col_idx] = scaler.transform(col_data).flatten()

        return result


@register_transform
class FilterRows(BaseTransform):
    """Filter rows based on column values"""

    def __init__(self, filters: Dict[Union[int, str], Tuple[float, float]],
                 combine_with: str = 'and', name: str = None):
        super().__init__(name=name, filters=filters, combine_with=combine_with)
        self.column_filters = None

    def fit(self, data: np.ndarray, **kwargs) -> 'FilterRows':
        """Convert column names to indices if necessary"""
        column_mapping = kwargs.get('column_mapping', {})
        self.column_filters = {}

        for col, (min_val, max_val) in self.params['filters'].items():
            if isinstance(col, str):
                if not column_mapping:
                    raise ValueError("column_mapping must be provided when using column names")
                if col not in column_mapping:
                    raise ValueError(
                        f"Column '{col}' not found in column_mapping. Available columns: {list(column_mapping.keys())}")
                col_idx = column_mapping[col]
            else:
                col_idx = col

            # Validate index against data shape
            if col_idx >= data.shape[1]:
                raise ValueError(f"Column index {col_idx} out of range for data with {data.shape[1]} columns")

            self.column_filters[col_idx] = (min_val, max_val)

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Filter rows based on column values"""
        if not self._is_fitted:
            raise ValueError("Transform not fitted. Call fit() first.")

        # Build mask based on filters
        if self.params['combine_with'] == 'and':
            mask = np.ones(data.shape[0], dtype=bool)
            for col_idx, (min_val, max_val) in self.column_filters.items():
                mask &= (data[:, col_idx] >= min_val) & (data[:, col_idx] <= max_val)
        else:  # 'or'
            mask = np.zeros(data.shape[0], dtype=bool)
            for col_idx, (min_val, max_val) in self.column_filters.items():
                mask |= (data[:, col_idx] >= min_val) & (data[:, col_idx] <= max_val)

        # Store how many rows were filtered in metadata
        self._metadata['input_rows'] = data.shape[0]
        self._metadata['output_rows'] = np.sum(mask)
        self._metadata['rows_filtered'] = data.shape[0] - np.sum(mask)

        return data[mask]


@register_transform
class PolynomialFeatures(BaseTransform):
    """Generate polynomial features"""

    def __init__(self, columns: List[Union[int, str]], degree: int = 2,
                 interaction_only: bool = False, name: str = None):
        super().__init__(name=name, columns=columns, degree=degree,
                         interaction_only=interaction_only)
        self.column_indices = None
        self.n_output_features = None
        self.combinations = None

    def fit(self, data: np.ndarray, **kwargs) -> 'PolynomialFeatures':
        """Determine column indices and prepare feature combinations"""
        # Determine which columns to use
        if all(isinstance(col, int) for col in self.params['columns']):
            self.column_indices = self.params['columns']
        else:
            # Assume column_mapping is provided in kwargs
            column_mapping = kwargs.get('column_mapping', {})

            if not column_mapping:
                raise ValueError("column_mapping must be provided when using column names")

            # Resolve column names to indices, with validation
            self.column_indices = []
            for col in self.params['columns']:
                if isinstance(col, int):
                    if col < data.shape[1]:
                        self.column_indices.append(col)
                    else:
                        raise ValueError(f"Column index {col} out of range for data with {data.shape[1]} columns")
                elif col in column_mapping:
                    self.column_indices.append(column_mapping[col])
                else:
                    raise ValueError(
                        f"Column '{col}' not found in column_mapping. Available columns: {list(column_mapping.keys())}")

        # Validate indices against data shape
        for idx in self.column_indices:
            if idx >= data.shape[1]:
                raise ValueError(f"Column index {idx} out of range for data with {data.shape[1]} columns")

        degree = self.params['degree']
        interaction_only = self.params['interaction_only']

        # Generate all combinations of features
        from itertools import combinations_with_replacement, combinations

        if interaction_only:
            # Only interactions, no powers
            self.combinations = []
            for d in range(2, degree + 1):
                self.combinations.extend(combinations(self.column_indices, d))
        else:
            # All combinations including powers
            self.combinations = []
            for d in range(2, degree + 1):
                self.combinations.extend(combinations_with_replacement(self.column_indices, d))

        self.n_output_features = len(self.column_indices) + len(self.combinations)
        self._metadata['n_input_features'] = len(self.column_indices)
        self._metadata['n_output_features'] = self.n_output_features

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Generate polynomial features"""
        if not self._is_fitted:
            raise ValueError("Transform not fitted. Call fit() first.")

        # Start with the original features
        result = data[:, self.column_indices].copy()

        # Add polynomial features
        for combo in self.combinations:
            # Multiply the specified columns together
            new_feature = np.ones(data.shape[0])
            for col_idx in combo:
                new_feature *= data[:, col_idx]

            result = np.column_stack((result, new_feature))

        return result


@register_transform
class AddConstantColumn(BaseTransform):
    """Add a constant column (intercept) to the data"""

    def __init__(self, value: float = 1.0, name: str = None):
        super().__init__(name=name, value=value)

    def fit(self, data: np.ndarray, **kwargs) -> 'AddConstantColumn':
        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        constant_col = np.full((data.shape[0], 1), self.params['value'])
        return np.hstack((data, constant_col))


@register_transform
class RandomSampler(BaseTransform):
    """Sample rows randomly"""

    def __init__(self, n_samples: int = None, fraction: float = None,
                 random_state: int = 42, name: str = None):
        super().__init__(name=name, n_samples=n_samples, fraction=fraction,
                         random_state=random_state)
        if n_samples is None and fraction is None:
            raise ValueError("Either n_samples or fraction must be specified")
        if n_samples is not None and fraction is not None:
            raise ValueError("Only one of n_samples or fraction can be specified")

    def fit(self, data: np.ndarray, **kwargs) -> 'RandomSampler':
        """Determine the number of samples to draw"""
        if self.params.get('fraction') is not None:
            self.params['n_samples'] = int(data.shape[0] * self.params['fraction'])

        self._metadata['input_rows'] = data.shape[0]
        self._metadata['output_rows'] = min(self.params['n_samples'], data.shape[0])

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Sample rows randomly"""
        if not self._is_fitted:
            raise ValueError("Transform not fitted. Call fit() first.")

        rng = np.random.RandomState(self.params['random_state'])
        n_samples = min(self.params['n_samples'], data.shape[0])

        if n_samples == data.shape[0]:
            return data

        indices = rng.choice(data.shape[0], n_samples, replace=False)
        return data[indices]