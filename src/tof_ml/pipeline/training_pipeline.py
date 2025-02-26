import os
import logging
import importlib
import datetime
import numpy as np
import pandas as pd
from src.tof_ml.database.api import DBApi
from src.tof_ml.logging.logging_utils import setup_logger
from src.tof_ml.training.trainer import Trainer
from src.tof_ml.database.report_generator import ReportGenerator
from src.tof_ml.utils.config_utils import load_config
from src.tof_ml.data.preprocessor import DataPreprocessor
from src.tof_ml.data.column_mapping import COLUMN_MAPPING
from src.tof_ml.data.data_filtering import filter_data

logger = logging.getLogger(__name__)


def main():
    logger = setup_logger('trainer')
    logger.info("Setting up the database connection.")
    db_api = DBApi(config_path="config/database_config.yaml")

    # 1. Load base config
    base_config = load_config("config/base_config.yaml")
    metadata = base_config
    base_config_not_reportable = load_config("config/base_config_not_report.yaml")

    # 2. class_config
    class_mapping = load_config("config/class_mapping_config.yaml")
    loader_mapping = class_mapping.get("Loader", {})
    splitter_mapping = class_mapping.get("Splitter", {})

    # 3. Dynamic loader
    loader_key = base_config["data"]["loader_config_key"]
    loader_info = loader_mapping[loader_key]
    loader_class_str = loader_info["loader_class"]
    module_name, class_name = loader_class_str.rsplit('.', 1)
    loader_module = importlib.import_module(module_name)
    LoaderClass = getattr(loader_module, class_name)

    # 4. Instantiate data loader
    loader_params = base_config.get("data", {})
    data_loader = LoaderClass(
        config=loader_params,
        mid1=[0.11248, 0.11248],
        mid2=[0.1354, 0.1354],
        column_mapping=COLUMN_MAPPING
    )

    # 5. Load data
    df = data_loader.load_data()
    df = filter_data(df, number_of_samples=base_config["data"].get("n_samples"))

    # 6. Split
    splitting_config = base_config.get("data_splitting", {})
    splitting_config["feature_columns"] = base_config["data"].get("feature_columns")
    splitting_config["output_columns"]  = base_config["data"].get("output_columns")
    splitter_type = splitting_config.get("type", "UniformSplitter")
    splitter_class_str = splitter_mapping[splitter_type]
    mod_name, cls_name = splitter_class_str.rsplit('.', 1)
    splitter_module = importlib.import_module(mod_name)
    SplitterClass = getattr(splitter_module, cls_name)

    splitter = SplitterClass(
        config=splitting_config,
        column_mapping=data_loader.column_mapping
    )
    X_train, y_train, X_val, y_val, X_test, y_test = splitter.split(df)
    metadata["original_data_shape"] = splitter.meta_data["original_data_shape"]
    metadata["X_train_shape"] = splitter.meta_data["X_train_shape"]
    metadata["X_val_shape"] = splitter.meta_data["X_val_shape"]
    metadata["X_test_shape"] = splitter.meta_data["X_test_shape"]

    # 7. Preprocess
    preprocessor = DataPreprocessor(config=base_config, local_col_mapping=data_loader.column_mapping)
    X_train_proc, y_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_val_proc, y_val_proc = preprocessor.transform(X_val, y_val, dataset_name="Val")
    X_test_proc, y_test_proc = preprocessor.transform(X_test, y_test, dataset_name="Test")


    # 8. Output path
    base_output_dir = base_config.get("model_output_dir", "./artifacts")
    timestamp_str = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    unique_subdir = f"run_{timestamp_str}"
    full_output_path = os.path.join(base_output_dir, unique_subdir)
    os.makedirs(full_output_path, exist_ok=True)

    # 9. Train
    model_config = base_config["model"]
    trainer = Trainer(
        config=base_config,
        model_config=model_config,
        X_train=X_train_proc,
        y_train=y_train_proc,
        X_val=X_val_proc,
        y_val=y_val_proc,
        X_test=X_test_proc,
        y_test=y_test_proc,
        output_path=full_output_path
    )
    trainer.run_training()
    y_test_pred, y_test_true, residuals = trainer.evaluate_model()

    training_results = {
        "val_mse": trainer.best_val_mse,
        "test_mse": trainer.test_mse
    }

    metadata = metadata | training_results

    # 10. Build data for plotting
    # E.g. we can keep them as NumPy arrays or create DataFrames with meaningful column names.
    # If you want to do stacked categories in histograms, you need e.g. "retardation" col, "R" col, etc.
    columns = data_loader.column_mapping.keys()
    # data_loader stage => masked data
    dl_df = pd.DataFrame(df, columns=columns)
    # for example, rename them or keep them consistent with your config references

    # splitter stage => train/val/test
    train_df = pd.DataFrame(np.column_stack([X_train, y_train]), columns=columns)
    train_df["split"] = "train"

    val_df = pd.DataFrame(np.column_stack([X_val, y_val]), columns=columns)
    val_df["split"] = "val"

    test_df = pd.DataFrame(np.column_stack([X_test, y_test]), columns=columns)
    test_df["split"] = "test"

    # preprocessor => scaled data
    train_scaled_df = pd.DataFrame(np.column_stack([X_train_proc, y_train_proc]), columns=columns)
    val_scaled_df   = pd.DataFrame(np.column_stack([X_val_proc, y_val_proc]), columns=columns)
    test_scaled_df  = pd.DataFrame(np.column_stack([X_test_proc, y_test_proc]), columns=columns)

    # trainer => e.g. combine y_test_true, y_test_pred, residuals into a DF
    retardation_column = data_loader.column_mapping['retardation']
    trainer_eval_df = pd.DataFrame({
        "true_energy":    y_test_true,
        "predicted_energy": y_test_pred,
        "residuals":        residuals,
        "retardation": X_test[:, retardation_column]
    })

    # 11. Build plot_data_dict
    plot_data_dict = {
        "data_loader": {
            "df": dl_df
        },
        "splitter": {
            "train": train_df,
            "val":   val_df,
            "test":  test_df
        },
        "preprocessor": {
            "train_scaled": train_scaled_df,
            "val_scaled":   val_scaled_df,
            "test_scaled":  test_scaled_df
        },
        "trainer": {
            "eval": trainer_eval_df
        }
    }

    # 12. Generate report
    report_gen = ReportGenerator(
        config=base_config_not_reportable,
        plot_data_dict=plot_data_dict,
        output_dir=os.path.join(full_output_path, "reports")
    )
    plot_paths = report_gen.generate_report()

    # 13. Combine metadata
    #combined_metadata = {}
    #combined_metadata.update(getattr(data_loader, "meta_data", {}))
    #combined_metadata.update(getattr(splitter, "meta_data", {}))
    #combined_metadata.update(getattr(preprocessor, "meta_data", {}))
    #combined_metadata.update(getattr(trainer, "meta_data", {}))
    #combined_metadata["plot_paths"] = plot_paths
    #metadata = metadata | plot_paths
    print(metadata)

    # 14. Record to DB
    db_api.record_model_run(
        config_dict=metadata,
        training_results=training_results,
        model_path=trainer.meta_data.get("final_model_path", ""),
        plot_paths=plot_paths
    )

    logger.info("Done.")

if __name__ == "__main__":
    main()
