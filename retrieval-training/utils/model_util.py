import torch
import torch.nn as nn


def print_trainable_parameters(model: nn.Module, logger):
    total, trainable = 0, 0
    trainable_names = []
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            trainable_names.append(f"  [trainable] {name} — {p.numel():,}")
    
    for line in trainable_names:
        logger.info(line)
    logger.info(f"\nTrainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
