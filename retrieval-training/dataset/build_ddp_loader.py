import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.dataset import ImageDataset
from dataset.collator import Collator


def build_loader_ddp(
    df: pd.DataFrame,
    batch_size: int = 128,
    num_workers: int = 16,
    image_size: int = 224,
    ddp_world_size: int = 1,
    ddp_rank: int = 0,
    clip_model_name: str = "ViT-B-32",
    train: bool = True,
):
    dataset = ImageDataset(df)
    collate = Collator(image_size=image_size, clip_model_name=clip_model_name)

    sampler = DistributedSampler(
        dataset,
        num_replicas=ddp_world_size,
        rank=ddp_rank,
        shuffle=train,
        drop_last=train,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    return loader, sampler
