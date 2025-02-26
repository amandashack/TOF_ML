from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Dict, List


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    """

    def __init__(
            self,
            directory: Optional[str] = None,
            plot_live: bool = False,
            mid1_range: Optional[List] = None,
            mid2_range: Optional[List] = None,
            retardation: Optional[List] = None,
            n_samples: Optional[int] = None,
            mask_col: Optional[str] = None,
            column_mapping: Optional[Dict] = None,
            feature_columns: Optional[List] = None,
            output_columns: Optional[List] = None,
            meta_data: Optional[Dict] = None,
            config: Optional[Dict] = None,
            **kwargs
    ):
        """
        If `config` is provided, populate the attributes from that.
        Then override with any explicitly passed parameters (non-None).
        """

        # 1) Set some defaults (safe initial values).
        self.directory = None
        self.plot_live = False
        self.mid1_range = None
        self.mid2_range = None
        self.retardation = None
        self.n_samples = None
        self.mask_col = None
        self.column_mapping = {}
        self.feature_columns = []
        self.output_columns = []
        self.meta_data = {}
        self.pass_energy = False

        # 2) If a config is provided, load from config.
        #    (Child classes can override `_init_from_config` if needed.)
        if config is not None:
            self._init_from_config(config, **kwargs)

        # 3) Now override with direct parameters if they are not None
        #    (i.e., direct parameters take precedence over config).
        if directory is not None:
            self.directory = directory
        if plot_live is not None:
            self.plot_live = plot_live
        if mid1_range is not None:
            self.mid1_range = mid1_range
        if mid2_range is not None:
            self.mid2_range = mid2_range
        if retardation is not None:
            self.retardation = retardation
        if n_samples is not None:
            self.n_samples = n_samples
        if mask_col is not None:
            self.mask_col = mask_col
        if column_mapping is not None:
            self.column_mapping = column_mapping
        if feature_columns is not None:
            self.feature_columns = feature_columns
        if output_columns is not None:
            self.output_columns = output_columns
        if meta_data is not None:
            self.meta_data = meta_data
        if 'pass_energy' in kwargs:
            self.pass_energy = kwargs['pass_energy']

    def _init_from_config(self, config: Dict, **kwargs):
        """
        A helper method to initialize/override instance variables from a config dict.
        Child classes can override or extend this if they have extra fields.
        """
        # Example: read the "data" section of the config, if that's how you nest it.
        # Or read directly from top-level if your config is flat.

        self.directory = config.get("directory", self.directory)
        self.plot_live = config.get("plot_live", self.plot_live)

        self.mid1_range = config.get("mid1_range", self.mid1_range)
        self.mid2_range = config.get("mid2_range", self.mid2_range)
        self.retardation = config.get("retardation_range", self.retardation)

        self.n_samples = config.get("n_samples", self.n_samples)
        self.mask_col = config.get("mask_data", self.mask_col)

        self.column_mapping = config.get("column_mapping", self.column_mapping)
        self.feature_columns = config.get("feature_columns", self.feature_columns)
        self.output_columns = config.get("output_columns", self.output_columns)

        self.meta_data = config.get("meta_data", self.meta_data)
        self.pass_energy = config.get("pass_energy", self.pass_energy)

    @abstractmethod
    def load_data(self) -> np.ndarray:
        """
        Abstract: Must return data of shape (N, X) (depending on your use case).
        """
        pass

