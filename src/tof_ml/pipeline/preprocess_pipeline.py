# preprocess_pipeline.py
import os
import logging
import numpy as np
import pandas as pd
import time

from src.tof_ml.data.h5_data_loader import H5DataLoader
from src.tof_ml.data.data_provenance import DataProvenance
from src.tof_ml.data.data_preprocessor import DataPreprocessor
from src.tof_ml.data.data_filtering import filter_data
from src.tof_ml.utils.config_utils import load_config
from src.tof_ml.data.column_mapping import COLUMN_MAPPING
from src.tof_ml.utils.plot_factory import PlotFactory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
# Get the matplotlib logger
pltlog = logging.getLogger('matplotlib')

# Set the logging level to a higher level like 'CRITICAL'
pltlog.setLevel(logging.CRITICAL)

def build_df_for_plot(dp: DataProvenance, feats: list, outs: list) -> pd.DataFrame:
    """
    Helper to create a DataFrame from dp.raw_data, using feats + outs
    as columns. This is used by PlotFactory for "plot_complex_figure".
    """
    logger.debug("Starting making a dataframe for plotting...")

    try:
        combined_cols = feats + outs
    except TypeError:
        combined_cols = feats + [outs]
    if not combined_cols:
        logger.warning("No combined_cols found, returning empty df.")
        return pd.DataFrame()

    # Truncate if dp.raw_data has fewer columns than expected
    if dp.raw_data.shape[1] < len(combined_cols):
        logger.warning(
            "dp.raw_data has fewer columns than combined_cols. Possibly a mismatch."
        )
        return pd.DataFrame(dp.raw_data)

    df = pd.DataFrame(dp.raw_data[:, :len(combined_cols)], columns=combined_cols)
    return df


class DataPipeline:
    """
    Orchestrates the loading of data, application of transformations,
    and optional generation of plots after each transformation.
    """

    def __init__(self, base_config_path: str = "config/base_config.yaml"):
        self.base_config = load_config(base_config_path)
        self.data_cfg = self.base_config.get("data", {})
        self.plot_enabled = self.data_cfg.get("live_plot", False)
        self.output_dir = self.data_cfg.get("directory", ".")

        self.dp = None            # Will hold an instance of DataProvenance
        self.preprocessor = None  # Will hold an instance of DataPreprocessor

    def run(self):
        logger.info("=== Starting DataPipeline.run() ===")
        # 1) Load data
        raw_data, loader_meta = self._load_data()
        if raw_data.size == 0:
            logger.error("No data loaded. Exiting pipeline.")
            return

        # 2) Optional filtering
        raw_data = self._filter_data(raw_data)

        # 3) Initialize DataProvenance
        self.dp = DataProvenance(
            raw_data=raw_data,
            column_mapping=COLUMN_MAPPING,
            config=self.data_cfg,
            loader_metadata=loader_meta
        )

        logger.info(f"DataProvenance created. raw_data shape={self.dp.raw_data.shape}")

        # 4) Plot raw data (complex figure)
        self._plot_data(step_name="raw_data", title="Raw Data")

        # 5) Initialize the DataPreprocessor
        #    Note: If you want a “standalone” preprocessor, do NOT pass dp directly here.
        #    Instead, you can pass config + column_mapping. But here we’ll assume we can
        #    pass dp in the constructor if you prefer that pattern.
        self.preprocessor = DataPreprocessor(config=self.base_config, column_mapping=self.dp.column_mapping)
        logger.debug("DataPreprocessor initialized.")
        # 5A) Mask data
        masked_data, mask_info = self.preprocessor.mask_data(self.dp.raw_data)
        self.dp.raw_data = masked_data
        self.dp.update_from_step(mask_info)
        logger.debug(f"Masking done. {mask_info}")
        self._plot_data(step_name="masked_data", title="Masked Data")

        # 5B) Apply pass-energy
        pass_energy_done = self.dp.metadata.get("pass_energy_done", False)
        pass_energy_config = self.dp.metadata.get("pass_energy_config", False)
        pe_data, pe_info = self.preprocessor.apply_pass_energy(
            self.dp.raw_data,
            pass_energy_config=pass_energy_config,
            pass_energy_done=pass_energy_done
        )
        self.dp.raw_data = pe_data  # should not be necessary but good for clarity
        self.dp.update_from_step(pe_info)
        logger.debug(f"Pass-energy step info: {pe_info}")

        # --- RENAME kinetc_energy -> pass_energy in your provenance ---
        if not pe_info.get("skipped"):
            # 1) fix column_mapping
            old_idx = self.dp.column_mapping.pop("kinetic_energy")
            self.dp.column_mapping["pass_energy"] = old_idx

            # 2) fix output_columns (if your pass energy is considered the output)
            outs = self.dp.metadata.get("output_columns", [])
            new_outs = [("pass_energy" if c == "kinetic_energy" else c) for c in outs]
            self.dp.metadata["output_columns"] = new_outs

            # 3) fix feature_columns too, if it was there
            feats = self.dp.metadata.get("feature_columns", [])
            new_feats = [("pass_energy" if c == "kinetic_energy" else c) for c in feats]
            self.dp.metadata["feature_columns"] = new_feats

        # 5C) Separate X,y so we can do log-transform + scaling
        feats = self.dp.metadata.get("feature_columns", [])
        outs  = self.dp.metadata.get("output_columns", [])
        if not feats or not outs:
            logger.error("No feature/output columns found. Exiting.")
            return
        logger.debug(f"feature_columns={feats}, output_columns={outs}")

        feat_idx = [self.dp.column_mapping[f] for f in feats]
        out_idx  = [self.dp.column_mapping[o] for o in outs]
        X = self.dp.raw_data[:, feat_idx] # this is a numpy view
        y = self.dp.raw_data[:, out_idx] # this is a numpy view
        if y.shape[1] == 1:
            y = y.ravel()

        # 5D) Log transform
        # Here we read if "log_transform" is set in self.base_config["scaler"]["log_transform"]
        log_transform = self.base_config.get("scaler", {}).get("log_transform", False)
        if log_transform:
            X, y, log_info = self.preprocessor.apply_log2(X, y)
            self.dp.update_from_step(log_info)
            if y.ndim == 1:
                y = y.reshape(-1,1)
            self.dp.raw_data = np.hstack([X, y])
            self._plot_data(step_name="log_data", title="Log-Transformed Data")

        # 5E) Fit-transform scaler
        scaling_done = self.dp.metadata.get("scaling_done", False)
        force_rescale = False
        X_scaled, y_scaled, scale_info = self.preprocessor.fit_transform_scaler(
            X, y, scaling_already_done=scaling_done, force_rescale=force_rescale
        )
        self.dp.update_from_step(scale_info)
        logger.debug(f"Scaling step. {scale_info}")

        if y_scaled is not None and y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1,1)

        # store final scaled data
        if y_scaled is not None:
            final_data = np.hstack([X_scaled, y_scaled])
        else:
            final_data = X_scaled
        self.dp.raw_data = final_data

        # plot scaled data
        self._plot_data(step_name="scaled_data", title="Scaled Data")

        # 6) Write final data + essential provenance to HDF5
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        h5_filename = f"final_processed_data_{timestamp}.h5"
        h5_path = os.path.join(self.output_dir, h5_filename)

        self.dp.write_to_h5(h5_path)
        logger.info(f"Wrote final processed data to HDF5: {h5_path}")

        # 7) Write operations log
        json_filename = f"pipeline_operations_{timestamp}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        self.dp.write_operations_log_to_file(json_path)
        logger.info(f"Wrote pipeline operations log to: {json_path}")

    def _load_data(self):
        loader = H5DataLoader(
            directory=self.data_cfg.get("directory", ""),
            mid1_range=self.data_cfg.get("mid1"),
            mid2_range=self.data_cfg.get("mid2"),
            retardation=self.data_cfg.get("retardation_range"),
            column_mapping=COLUMN_MAPPING,
            config=self.data_cfg
        )
        raw_data, loader_meta = loader.load_data()
        logger.info(f"Loaded data shape: {raw_data.shape}")
        return raw_data, loader_meta

    def _filter_data(self, raw_data: np.ndarray) -> np.ndarray:
        n_samples = self.data_cfg.get("n_samples")
        if n_samples:
            raw_data = filter_data(raw_data, number_of_samples=n_samples)
            logger.info(f"After filtering: {raw_data.shape}")
        return raw_data

    def _plot_data(self, step_name: str, title: str = ""):
        """
        Helper to produce a "complex figure" for the current dp.raw_data,
        store the figure path in dp, or show it live if plot_enabled = True.
        """
        if not self.dp:
            logger.warning("No DataProvenance (dp) to plot from. Skipping.")
            return

        feats = self.dp.metadata.get("feature_columns", [])
        outs  = self.dp.metadata.get("output_columns", [])
        df    = build_df_for_plot(self.dp, feats, outs)
        if df.empty:
            logger.warning(f"No data frame to plot at step {step_name}")
            return

        # If plot_enabled=False, we save to disk. Otherwise, we show live.
        out_path = ""
        if not self.plot_enabled:
            plots_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # Make a unique name for the file, e.g. raw_data_complex_20230330_121550.png
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{step_name}_complex_{timestamp}.png"
            out_path = os.path.join(plots_dir, filename)

        fig_path = PlotFactory.plot_complex_figure(
            df=df,
            title=title,
            out_path=out_path,   # empty => show on screen
            scatter_symlog=True
        )

        logger.debug("Finished making a plot...")

        if fig_path:
            # If we actually saved a file, store it in dp
            self.dp.metadata.setdefault("plots", {})
            self.dp.metadata["plots"][step_name] = fig_path

    def _print_ops_log(self):
        # For demonstration, we can see how many transformations were recorded
        if hasattr(self.dp, "operations_log"):
            logger.info("=== DataProvenance Operations Log ===")
            for idx, op in enumerate(self.dp.operations_log):
                logger.info(f"Step {idx+1}: {op}")
        else:
            logger.info("No operations log found in DataProvenance.")


def main():
    pipeline = DataPipeline("config/base_config.yaml")
    pipeline.run()


if __name__ == "__main__":
    main()
