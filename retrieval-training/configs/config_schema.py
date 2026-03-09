from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    train_path: str
    val_path: str
    invalid_tcin: List[int]


@dataclass
class ModelConfig:
    model_class: str
    clip_model_name: str
    pretrained: str
    normalize: bool
    in_batch_negative_sampling_strategy: str
    checkpoint_path: str
    share_text_encoder: bool
    freeze_query_text: bool
    freeze_item_text: bool
    freeze_item_image: bool
    unlock_layers: int


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    ddp_world_size: int
    ddp_rank: int
    lr: float
    weight_decay: float
    num_warmup_steps: int
    save_dir: str
    save_model_name: str


@dataclass
class LoggingConfig:
    log_dir: str
    log_name: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
