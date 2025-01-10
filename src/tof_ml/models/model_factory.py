from .tof_to_energy_model import MLPKerasRegressor
# from .rf_regressor import RFRegressorWrapper  # etc.


class ModelFactory:
    @staticmethod
    def create_model(model_config: dict):
        model_type = model_config.get("type", "MLPRegressor")
        params = model_config.get("params", {})

        if model_type == "MLPRegressor":
            # Return your new Keras model
            return MLPKerasRegressor(**params)
        # elif model_type == "RandomForestRegressor":
        #     return RFRegressorWrapper(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
