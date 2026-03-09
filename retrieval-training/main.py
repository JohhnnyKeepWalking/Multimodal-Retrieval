import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup

from model import clip_tt_moe_fusion, clip_tt_mlp_fusion, clip_tt_attention_fusion, clip_tt_moe_bilinear_fusion, clip_tt_moe_fusion_query_projection
model_registry = {
    "clip_tt_moe_fusion": clip_tt_moe_fusion.CLIPTwoTower,
    "clip_tt_mlp_fusion": clip_tt_mlp_fusion.CLIPTwoTower,
    "clip_tt_attention_fusion": clip_tt_attention_fusion.CLIPTwoTower,
    "clip_tt_moe_bilinear_fusion": clip_tt_moe_bilinear_fusion.CLIPTwoTower,
    "clip_tt_moe_fusion_query_projection": clip_tt_moe_fusion_query_projection.CLIPTwoTower,
}

from trainer.trainer import Trainer
from dataset.build_ddp_loader import build_loader_ddp
from loss.multi_objective_hinge_loss import MultiTaskHingeLoss

from utils.load_config import load_config
from utils.model_util import print_trainable_parameters
from utils.setup_logger import setup_logger, log_config



if __name__ == "__main__":
    config = load_config("/home/jovyan/clip/clip_tt/configs/moe_fusion_with_projection_freeze_all.yaml")
    logger = setup_logger(config.logging.log_dir, config.logging.log_name)
    log_config(logger, config)

    train_df = pd.read_parquet(config.data.train_path)
    train_df = train_df[~train_df["tcin"].isin(config.data.invalid_tcin)]
    val_df = pd.read_parquet(config.data.val_path)

    epochs = config.training.epochs
    train_loader, train_sampler = build_loader_ddp(df=train_df, batch_size=config.training.batch_size, ddp_world_size=config.training.ddp_world_size, ddp_rank=config.training.ddp_rank)
    val_loader, val_sampler = build_loader_ddp(df=val_df, batch_size=config.training.batch_size, ddp_world_size=config.training.ddp_world_size, ddp_rank=config.training.ddp_rank, train=False)

    model_cls = model_registry[config.model.model_class]
    model = model_cls(config.model.clip_model_name, pretrained=config.model.pretrained, checkpoint_path=config.model.checkpoint_path, share_text_encoder=config.model.share_text_encoder,
        freeze_query_text=config.model.freeze_query_text, freeze_item_text=config.model.freeze_item_text, freeze_item_image=config.model.freeze_item_image, unlock_layers=config.model.unlock_layers)
    optimizer = AdamW(model.parameters(), lr=float(config.training.lr), weight_decay=float(config.training.weight_decay))
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=config.training.num_warmup_steps,
                                                num_training_steps=len(train_loader) * epochs)
    loss_fn = MultiTaskHingeLoss()

    print_trainable_parameters(model, logger)

    trainer = Trainer(model, optimizer, loss_fn, scheduler=scheduler, save_dir=config.training.save_dir, model_name=config.training.save_model_name, logger=logger)
    trainer.fit(train_loader, epochs, val_loader=val_loader)