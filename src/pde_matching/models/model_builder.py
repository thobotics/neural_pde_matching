from omegaconf import DictConfig, OmegaConf


class ModelBuilder:
    models = {}

    @classmethod
    def register(cls, model_name):
        def decorator(model_cls):
            cls.models[model_name] = model_cls
            return model_cls

        return decorator

    def __init__(self, config: DictConfig):
        self.config = config

    def build(self, example_data=None):
        model_type = self.config.name
        model_params = OmegaConf.to_container(self.config, resolve=True)
        del model_params["name"]
        if example_data is not None:
            model_params["input_dim_node"] = example_data.get_input_dim()
            model_params["input_dim_edge"] = example_data.get_edge_dim()
            model_params["output_dim"] = example_data.get_output_dim()
            model_params["output_is_a_state_pair"] = (
                example_data.is_output_a_state_pair()
            )
        model_cls = self.models.get(model_type)
        if model_cls:
            return model_cls(**model_params)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
