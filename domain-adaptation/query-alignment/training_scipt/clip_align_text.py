import os
import torch
import open_clip
import pandas as pd
import torch.nn as nn
from tqdm.auto import tqdm
from PIL import Image
from accelerate import Accelerator
import torchvision.transforms as T
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "query": row["query"],
            "title": row["title"],
            "desirability_label": row["desirability_label"],
            "relevance_label": row["relevance_label"],
        }


class Collator:
    def __init__(self, image_size: int = 224, clip_model_name: str = "ViT-B-32"):
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries = [b["query"] for b in batch]
        titles   = [b["title"] for b in batch]

        query_ids = self.tokenizer(queries)
        title_ids = self.tokenizer(titles)

        desirability = torch.tensor([b["desirability_label"] for b in batch])
        relevance = torch.tensor([b["relevance_label"] for b in batch])

        return {
            "query_ids": query_ids,
            "title_ids": title_ids,
            "desirability_label": desirability,
            "relevance_label": relevance,
        }


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

class CLIPTwoTower(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        normalize: bool = True,
        in_batch_negative_sampling_strategy: str = "weighted_sampling",
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )

        if checkpoint_path is not None:
            self.model = self.load_from_checkpoint(checkpoint_path)
        for p in self.model.parameters():
            p.requires_grad = True
        # for p in self.model.transformer.resblocks[-2:].parameters():
        #     p.requires_grad = True
        # for p in self.model.visual.transformer.resblocks[-2:].parameters():
        #     p.requires_grad = True

        self.normalize = normalize
        self.in_batch_negative_sampling_strategy = in_batch_negative_sampling_strategy

    def load_from_checkpoint(self, checkpoint_path: str):
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu")

            # unwrap checkpoint if needed
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt

            # strip DDP "module." prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                new_state_dict[k] = v

            missing, unexpected = self.model.load_state_dict(
                new_state_dict, strict=False
            )

            if missing:
                print(f"[CLIP] Missing keys: {missing}")
            if unexpected:
                print(f"[CLIP] Unexpected keys: {unexpected}")
            print(f"[CLIP] Loaded weights from {checkpoint_path}")

        return self.model
    
    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_text(text_ids)
        if self.normalize:
            feats = F.normalize(feats, dim=-1)
        return feats
    
    def in_batch_negative_sampling(self, query_emb, item_emb):
        similarity_matrix = torch.matmul(query_emb, item_emb.T)
        n = similarity_matrix.shape[0]
        positive_cosine = similarity_matrix.diag()
        device = similarity_matrix.device

        if self.in_batch_negative_sampling_strategy == "random":
            mask = ~torch.eye(n, dtype=bool, device=similarity_matrix.device)
            indices = torch.nonzero(mask, as_tuple=False)
            sampled_indices = indices[torch.randperm(indices.size(0))[:n]]
            negative_cosine = similarity_matrix[sampled_indices[:, 0], sampled_indices[:, 1]]

        elif self.in_batch_negative_sampling_strategy == "weighted_sampling":
            full_mask = torch.eye(n, dtype=torch.bool, device=device)
            probabilities = torch.relu(similarity_matrix.masked_fill(full_mask, 0))
            try:
                probabilities = probabilities / probabilities.sum()
                sampled_indices = torch.multinomial(probabilities.flatten(), n, replacement=False)
                row_indices = sampled_indices // n
                col_indices = sampled_indices % n
                negative_cosine = similarity_matrix[row_indices, col_indices]
            except:
                probabilities = probabilities + 1e-4
                probabilities = probabilities / probabilities.sum()
                sampled_indices = torch.multinomial(probabilities.flatten(), n, replacement=False)
                row_indices = sampled_indices // n
                col_indices = sampled_indices % n
                negative_cosine = similarity_matrix[row_indices, col_indices]
        combined_cosine = torch.cat([positive_cosine, negative_cosine])
        return combined_cosine

    def forward(self, batch: dict) -> dict:
        query_emb = self.encode_text(batch["query_ids"])
        title_emb = self.encode_text(batch["title_ids"])
        q_t_align = self.in_batch_negative_sampling(query_emb, title_emb)
        return {
            "q_t_align": q_t_align
        }


class MultiTaskHingeLoss(nn.Module):
    def __init__(self, epsilon_high=0.8, epsilon_low=0.65, epsilon_neg=0.4, m=2, desirability_weight=0.7, relevance_weight=0.3):
        super(MultiTaskHingeLoss, self).__init__()
        self.epsilon_high = epsilon_high
        self.epsilon_low = epsilon_low
        self.epsilon_neg = epsilon_neg
        self.m = m
        self.desirability_weight = desirability_weight
        self.relevance_weight = relevance_weight

    def compute_loss(self, y_pred, y_true):
        label_high = (y_true == 2).float()
        label_low = (y_true == 1).float()
        label_neg = (y_true == 0).float()

        loss_high = torch.pow(torch.clamp(self.epsilon_high - y_pred, min=0), self.m)
        loss_low = torch.pow(torch.clamp(y_pred - self.epsilon_low, min=0), self.m)
        loss_neg = torch.pow(torch.clamp(y_pred - self.epsilon_neg, min=0), self.m)

        loss = label_high * loss_high + label_low * loss_low + label_neg * loss_neg
        return loss.mean()

    def forward(self, y_pred, desirability_label, relevance_label):
        padded_desirability_label = torch.cat([desirability_label, 
                                               torch.zeros(y_pred.size(0) - desirability_label.size(0), 
                                                           device=y_pred.device)])
        padded_relevance_label = torch.cat([relevance_label, 
                                            torch.zeros(y_pred.size(0) - relevance_label.size(0), 
                                                        device=y_pred.device)])

        desirability_loss = self.compute_loss(y_pred, padded_desirability_label)
        relevance_loss = self.compute_loss(y_pred, padded_relevance_label)

        total_loss = self.desirability_weight * desirability_loss + self.relevance_weight * relevance_loss
        return total_loss

class Trainer:
    def __init__(self, model, optimizer, loss_fn, scheduler):
        self.accelerator = Accelerator()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.save_dir = "/home/jovyan/image_tt/checkpoints/clip_align_text/"

    def fit(self, train_loader, epochs: int, val_loader: Optional[DataLoader] = None):
        self.model, self.optimizer, self.scheduler, train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, train_loader
        )
        if val_loader is not None:
            val_loader = self.accelerator.prepare(val_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["val_loss"]
            
            if self.accelerator.is_main_process:
                print(f"Initial evaluation: "
                        f"val_loss={val_loss:.4f}", flush=True)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            steps = 0
            last_100_losses = []  # Track last 100 losses

            if self.accelerator.is_main_process:
                pbar = tqdm(
                    total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{epochs}",
                    dynamic_ncols=True,
                    ncols=0,
                    position=0,
                    leave=True,
                )

            for batch in train_loader:
                results = self.model(batch)
                loss = self.loss_fn(results["q_t_align"], batch["desirability_label"], batch["relevance_label"])

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                loss = loss.detach()
                running_loss += loss
                steps += 1
                
                # Track last 100 losses
                last_100_losses.append(loss.item())
                if len(last_100_losses) > 100:
                    last_100_losses.pop(0)

                if self.accelerator.is_main_process:
                    lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
                    pbar.update(1)
                    
                    if steps % 100 == 0:
                        avg_100_loss = sum(last_100_losses) / len(last_100_losses)
                        print(f"\n[Epoch {epoch+1}, Step {steps}] avg_loss_last_100_steps={avg_100_loss:.4f}", flush=True)

            avg_loss = self.accelerator.reduce(running_loss / steps, reduction="mean")
            print(f"[Epoch {epoch}] avg_train_loss={avg_loss.item():.4f}", flush=True)
            if self.accelerator.is_main_process:
                pbar.close()
                unwrapped = self.accelerator.unwrap_model(self.model)
                save_path = os.path.join(self.save_dir, f"clip_emb_fusion_{epoch+1}.pt")
                torch.save(unwrapped.state_dict(), save_path)
                self.accelerator.print(f"✅ Saved model checkpoint to {save_path}", flush=True)
            
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics["val_loss"]
                
                if self.accelerator.is_main_process:
                    print(f"[Epoch {epoch+1}] train_loss={avg_loss.item():.4f} | "
                          f"val_loss={val_loss:.4f}", flush=True)
                    
    
    def evaluate(self, eval_loader):
        self.model.eval()
        running_loss = 0.0
        steps = 0
        
        if self.accelerator.is_main_process:
            pbar = tqdm(
                total=len(eval_loader),
                desc="Evaluating",
                dynamic_ncols=True,
                ncols=0,
                position=0,
                leave=True,
            )
        
        with torch.no_grad():
            for batch in eval_loader:
                results = self.model(batch)
                
                loss = self.loss_fn(results["q_t_align"], batch["desirability_label"], batch["relevance_label"])
                
                running_loss += loss.detach()
                steps += 1
                
                if self.accelerator.is_main_process:
                    pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
                    pbar.update(1)
        
        if self.accelerator.is_main_process:
            pbar.close()
        
        avg_loss = self.accelerator.reduce(running_loss / steps, reduction="mean")
        
        metrics = {
            "val_loss": avg_loss.item()
        }
        
        return metrics


if __name__ == "__main__":
    train_df = pd.read_parquet("/home/jovyan/image_tt/data/image_train.parquet")
    train_df = train_df[~train_df["tcin"].isin([88992305, 90471880])]
    val_df = pd.read_parquet("/home/jovyan/image_tt/data/image_val.parquet")

    epochs = 1
    train_loader, train_sampler = build_loader_ddp(df=train_df, batch_size=512, ddp_world_size=1, ddp_rank=0, train=True)
    val_loader, val_sampler = build_loader_ddp(df=val_df, batch_size=512, ddp_world_size=1, ddp_rank=0, train=False)

    model = CLIPTwoTower("ViT-B-32", pretrained="openai", checkpoint_path="/home/jovyan/two-tower-retrieval-datavol-1/checkpoints/fastclip_train_from_pretrained/finetune_both_tower/epoch_5.pt")
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=200,
                                                num_training_steps=len(train_loader) * epochs)
    loss_fn = MultiTaskHingeLoss()

    trainer = Trainer(model, optimizer, loss_fn, scheduler=scheduler)
    trainer.fit(train_loader, epochs, val_loader=val_loader)