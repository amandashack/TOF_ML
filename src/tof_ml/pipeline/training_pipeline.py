# src/pipelines/training_pipeline.py
import os
import importlib
import datetime
from src.tof_ml.database.api import DBApi
from src.tof_ml.logging.logging_utils import setup_logger
from src.tof_ml.training.trainer import Trainer
from src.tof_ml.data.data_filtering import filter_data
from src.tof_ml.utils.config_utils import load_config
from src.tof_ml.data.data_splitting import BaseDataSplitter, UniformSplitter, SubsetSplitter
from src.tof_ml.data.preprocessor import DataPreprocessor

SPLITTER_MAPPING = {
    "UniformSplitter": UniformSplitter,
    "SubsetSplitter": SubsetSplitter,
    # You can add more splitters here
}


def main():
    # 1. Load base config
    base_config = load_config("config/base_config.yaml")

    # 2. loader_config
    loader_config = load_config("config/loader_config.yaml")

    # 3. Get loader info
    loader_key = base_config["data"]["loader_config_key"]
    loader_info = loader_config[loader_key]

    # 4. Dynamically import loader
    loader_class_path = loader_info["loader_class"]
    module_name, class_name = loader_class_path.rsplit('.', 1)
    loader_module = importlib.import_module(module_name)
    LoaderClass = getattr(loader_module, class_name)

    # 5. Loader params
    loader_params = base_config["data"].get("parameters", {})
    data_loader = LoaderClass(config=base_config)

    # 6. Load data
    df = data_loader.load_data()

    # 7. Filter
    mid1 = loader_params.get("mid1", None)
    mid2 = loader_params.get("mid2", None)
    retardation_range = loader_params.get("retardation_range", None)
    number_of_samples = loader_params.get("number_of_samples", None)

    data_filtered = filter_data(
        df,
        retardation_range=None if retardation_range == "none" else retardation_range,
        mid1=None if mid1 == "none" else mid1,
        mid2=None if mid2 == "none" else mid2,
        number_of_samples=number_of_samples if number_of_samples else None,
        random_state=42
    )

    # 8. Split the data into train/val/test using your SPLITTER_MAPPING
    #    The base_config "data_splitting" block chooses which splitter and how.
    splitting_config = base_config.get("data_splitting", {})
    splitter_type = splitting_config.get("type", "UniformSplitter")
    splitter_class = SPLITTER_MAPPING[splitter_type]

    # 9. The splitter needs to know which columns are features vs. target,
    #    plus any subset columns. So let's set that in the config:
    #    We'll combine splitting_config + features_config
    #    so that the splitter can figure out the correct columns by index.
    features_config = base_config["features"]
    combined_split_config = {
        **splitting_config,
        "features": features_config
    }

    # 10. Instantiate and run the splitter
    splitter = splitter_class(config=combined_split_config, local_col_mapping=data_loader.column_mapping)
    X_train, y_train, X_val, y_val, X_test, y_test = splitter.split(data_filtered)

    # 11. Preprocess: (fit only on train, then transform val/test)
    # e.g. your base_config might have a "preprocessing" block, or you can unify with "scaler" + "features"
    # For example:
    preproc_config = {
        "apply_log": True,  # or True if you want to log-transform
        "scaler_type": base_config.get("scaler", {}).get("type", "None"),
        "generate_interactions": features_config.get("generate_interactions", False)
    }

    preprocessor = DataPreprocessor(preproc_config)
    X_train_proc, y_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_val_proc, y_val_proc = preprocessor.transform(X_val, y_val)
    X_test_proc, y_test_proc = preprocessor.transform(X_test, y_test)

    # 12. Setup logger, DB
    logger = setup_logger()
    logger.info("Setting up the database connection.")
    db_api = DBApi(config_path="config/database_config.yaml")

    # 13. Create output subdirectory
    base_output_dir = base_config.get("model_output_dir", "./artifacts")
    timestamp_str = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    unique_subdir = f"run_{timestamp_str}"
    full_output_path = os.path.join(base_output_dir, unique_subdir)

    # 14. Extract model config & training config
    model_config = base_config["model"]
    training_config = base_config["training"]

    # 15. Instantiate Trainer with final, processed arrays
    trainer = Trainer(
        model_config=model_config,
        training_config=training_config,
        db_api=db_api,
        X_train=X_train_proc,
        y_train=y_train_proc,
        X_val=X_val_proc,
        y_val=y_val_proc,
        X_test=X_test_proc,
        y_test=y_test_proc,
        output_path=full_output_path
    )

    # 16. Run training
    trainer.run_training()

    # 17. Evaluate
    trainer.evaluate_model()

    # 18. Record results to DB
    trainer.record_to_database(base_config)

    logger.info(f"Pipeline complete! Model artifacts stored in: {full_output_path}")


if __name__ == "__main__":
    main()
