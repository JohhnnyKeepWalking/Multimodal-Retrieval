import yaml
from configs.config_schema import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    LoggingConfig,
)


def load_config(config_path: str) -> Config:
    """
    Load YAML config file into structured Config object.
    """

    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    config = Config(
        data=DataConfig(**raw_cfg["data"]),
        model=ModelConfig(**raw_cfg["model"]),
        training=TrainingConfig(**raw_cfg["training"]),
        logging=LoggingConfig(**raw_cfg["logging"]),
    )

    return config
