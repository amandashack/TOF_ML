# src/pipelines/training_pipeline.py
import os
import importlib
from src.database.api import DBApi
from src.logging.logging_utils import setup_logger
from src.training.trainer import Trainer
from src.data.data_filtering import filter_data
from src.utils.config_utils import load_config
import datetime


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
    data_directory = base_config["data"]
    data_loader = LoaderClass(config=data_directory)

    # 6. Load data
    df = data_loader.load_data()
    print(df, df.shape)
    #df = df[df[:, 6] > 406]
    print(df, df.shape)

    # 7. Filter
    mid1 = loader_params.get("mid1", None)
    mid2 = loader_params.get("mid2", None)
    retardation_range = loader_params.get("retardation_range", None)
    number_of_samples = loader_params.get("number_of_samples", None)

    df_filtered = filter_data(
        df,
        retardation_range=None if retardation_range == "none" else retardation_range,
        mid1=None if mid1 == "none" else mid1,
        mid2=None if mid2 == "none" else mid2,
        number_of_samples=number_of_samples if number_of_samples else None,
        random_state=42
    )

    # 8. Extract model & training config
    model_config = base_config["model"]
    training_config = base_config["training"]
    features_config = base_config["features"]
    training_config["features"] = features_config

    # 9. Setup logger, DB
    logger = setup_logger("trainer")
    logger.info("Setting up the database connection.")
    db_api = DBApi(config_path="config/database_config.yaml")

    # 10. Generate a unique subdirectory under base_config["model_output_dir"]
    base_output_dir = base_config.get("model_output_dir", "./artifacts")
    # E.g. "2023_10_09_153012" or a random UUID
    timestamp_str = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    unique_subdir = f"run_{timestamp_str}"
    full_output_path = os.path.join(base_output_dir, unique_subdir)

    # 11. Instantiate Trainer
    logger.info(f"Creating trainer with output path: {full_output_path}")
    trainer = Trainer(
        model_config=model_config,
        training_config=training_config,
        logger=logger,
        db_api=db_api,
        df=df_filtered,
        output_path=full_output_path
    )

    # 12. Prepare data (splits train/val/test, scales, adds interactions)
    trainer.prepare_data()

    # 13. Run training (report best epoch val MSE)
    trainer.run_training()

    # 14. Evaluate on test
    trainer.evaluate_model()

    # 15. Record to DB
    trainer.record_to_database(base_config)

    logger.info(f"Pipeline complete! Model artifacts stored in: {full_output_path}")


if __name__ == "__main__":
    main()
