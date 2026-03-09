import torch
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple



class CLIPTwoTower(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        normalize: bool = True,
        in_batch_negative_sampling_strategy: str = "weighted_sampling",
        checkpoint_path: Optional[str] = None,
        share_text_encoder: bool = True,
        freeze_query_text: bool = True,
        freeze_item_text: bool = True,
        freeze_item_image: bool = True,
        unlock_layers: Optional[int] = None,
    ):
        super().__init__()
        self.normalize = normalize
        self.share_text_encoder = share_text_encoder
        self.in_batch_negative_sampling_strategy = in_batch_negative_sampling_strategy

        self.model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )

        if not share_text_encoder:
            self.query_text_model, _, _ = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=pretrained
            )

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self._apply_freezing(freeze_query_text, freeze_item_text, freeze_item_image, unlock_layers)

        self.alpha_network = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def _load_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Strip "module." and "model." prefixes
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("model."):
                k = k[len("model."):]
            clean_state_dict[k] = v

        for m, name in (
            [(self.model, "item model")]
            + ([(self.query_text_model, "query text model")] if not self.share_text_encoder else [])
        ):
            missing, unexpected = m.load_state_dict(clean_state_dict, strict=False)
            if missing:
                print(f"[CLIP] {name} missing keys: {missing}")
            if unexpected:
                print(f"[CLIP] {name} unexpected keys: {unexpected}")
        print(f"[CLIP] Loaded weights from {checkpoint_path}")

    def _apply_freezing(
        self,
        freeze_query_text: bool,
        freeze_item_text: bool,
        freeze_item_image: bool,
        unlock_layers: Optional[int] = None,
    ):
        # Freeze item text encoder
        if freeze_item_text:
            for p in self.model.transformer.parameters():
                p.requires_grad = False
            for attr in ("token_embedding", "ln_final", "text_projection", "positional_embedding", "logit_scale"):
                obj = getattr(self.model, attr, None)
                if obj is not None:
                    if isinstance(obj, nn.Parameter):
                        obj.requires_grad = False
                    else:
                        for p in obj.parameters():
                            p.requires_grad = False
            if unlock_layers is not None and unlock_layers > 0:
                for p in self.model.transformer.resblocks[-unlock_layers:].parameters():
                    p.requires_grad = True

        # Freeze item image encoder
        if freeze_item_image:
            for p in self.model.visual.parameters():
                p.requires_grad = False
            if unlock_layers is not None and unlock_layers > 0:
                for p in self.model.visual.transformer.resblocks[-unlock_layers:].parameters():
                    p.requires_grad = True
                for attr in ("ln_final", "text_projection", "logit_scale"):
                    obj = getattr(self.model, attr)
                    if isinstance(obj, nn.Parameter):
                        obj.requires_grad = True
                    else:
                        for p in obj.parameters():
                            p.requires_grad = True

        # Freeze query text encoder (only relevant when not sharing)
        if not self.share_text_encoder and freeze_query_text:
            for p in self.query_text_model.transformer.parameters():
                p.requires_grad = False
            for attr in ("token_embedding", "ln_final", "text_projection", "positional_embedding", "logit_scale"):
                obj = getattr(self.query_text_model, attr, None)
                if obj is not None:
                    if isinstance(obj, nn.Parameter):
                        obj.requires_grad = False
                    else:
                        for p in obj.parameters():
                            p.requires_grad = False
            if unlock_layers is not None and unlock_layers > 0:
                for p in self.query_text_model.transformer.resblocks[-unlock_layers:].parameters():
                    p.requires_grad = True
                for attr in ("ln_final", "text_projection", "logit_scale"):
                    obj = getattr(self.query_text_model, attr)
                    if isinstance(obj, nn.Parameter):
                        obj.requires_grad = True
                    else:
                        for p in obj.parameters():
                            p.requires_grad = True

    def encode_text(self, text_ids: torch.Tensor, use_query_encoder: bool = False) -> torch.Tensor:
        m = self.query_text_model if (use_query_encoder and not self.share_text_encoder) else self.model
        feats = m.encode_text(text_ids)
        return F.normalize(feats, dim=-1) if self.normalize else feats

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images)
        return F.normalize(feats, dim=-1) if self.normalize else feats

    def encode_item(self, text_ids: torch.Tensor, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        text_feats = self.encode_text(text_ids, use_query_encoder=False)
        image_feats = self.encode_image(images)
        alpha = self.alpha_network(torch.cat([text_feats, image_feats], dim=-1))
        weighted_feats = alpha * text_feats + (1 - alpha) * image_feats
        return F.normalize(weighted_feats, dim=-1) if self.normalize else weighted_feats, alpha

    def in_batch_negative_sampling(self, query_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        similarity_matrix = torch.matmul(query_emb, item_emb.T)
        n = similarity_matrix.shape[0]
        positive_cosine = similarity_matrix.diag()
        device = similarity_matrix.device

        if self.in_batch_negative_sampling_strategy == "random":
            mask = ~torch.eye(n, dtype=torch.bool, device=device)
            indices = torch.nonzero(mask, as_tuple=False)
            sampled_indices = indices[torch.randperm(indices.size(0))[:n]]
            negative_cosine = similarity_matrix[sampled_indices[:, 0], sampled_indices[:, 1]]

        elif self.in_batch_negative_sampling_strategy == "weighted_sampling":
            mask = torch.eye(n, dtype=torch.bool, device=device)
            probabilities = torch.relu(similarity_matrix.masked_fill(mask, 0))
            probabilities = probabilities / probabilities.sum()
            sampled_indices = torch.multinomial(probabilities.flatten(), n, replacement=False)
            negative_cosine = similarity_matrix[sampled_indices // n, sampled_indices % n]

        return torch.cat([positive_cosine, negative_cosine])

    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        query_emb = self.encode_text(batch["query_ids"], use_query_encoder=True)
        item_emb, alpha = self.encode_item(batch["title_ids"], batch["images"])
        return self.in_batch_negative_sampling(query_emb, item_emb), alpha

    def evaluate(self, batch: dict) -> torch.Tensor:
        query_emb = self.encode_text(batch["query_ids"], use_query_encoder=True)
        item_emb, alpha = self.encode_item(batch["title_ids"], batch["images"])
        similarity = F.cosine_similarity(query_emb, item_emb, dim=-1)
        return similarity