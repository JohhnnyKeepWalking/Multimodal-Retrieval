import torch
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossAttentionFusion(nn.Module):
    def __init__(self, d=512, heads=8, ff_hidden=None, depth=2, attn_dropout=0.03, drop=0.03):
        super().__init__()
        self.depth = depth
        self.d = d
        self.layers = nn.ModuleList()
        ff_hidden = ff_hidden or (d*4)
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(embed_dim=d, num_heads=heads, dropout=attn_dropout, batch_first=True),
                "ln1": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, ff_hidden), nn.GELU(), nn.Linear(ff_hidden, d)),
                "ln2": nn.LayerNorm(d),
                "drop": nn.Dropout(drop)
            }))
        self.out_proj = nn.Linear(d, d)

    def forward(self, txt_vec, img_vec):
        # img_vec, txt_vec: (B, d)
        I = img_vec.unsqueeze(1)   # (B,1,d)
        T = txt_vec.unsqueeze(1)   # (B,1,d)
        # Option A: make a joint sequence [T, I]
        seq = torch.cat([T, I], dim=1)  # (B,2,d)
        x = seq
        for layer in self.layers:
            attn_out, _ = layer["attn"](x, x, x, need_weights=False)  # self-attn on joint seq
            x = layer["ln1"](x + layer["drop"](attn_out))
            ff = layer["ff"](x)
            x = layer["ln2"](x + layer["drop"](ff))
        # pool: take mean or first token
        pooled = x[:,0,:]  # x.mean(dim=1) or x[:,0,:]
        out = self.out_proj(pooled)
        return out
    

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
        unlock_layers: int = 2,
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

        self.attention_fusion = CrossAttentionFusion(d=512, heads=4, depth=2, attn_dropout=0.03, drop=0.03)

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
        unlock_layers: int,
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
            if unlock_layers > 0:
                for p in self.model.transformer.resblocks[-unlock_layers:].parameters():
                    p.requires_grad = True
                for attr in ("ln_final", "text_projection", "logit_scale"):
                    obj = getattr(self.model, attr)
                    if isinstance(obj, nn.Parameter):
                        obj.requires_grad = True
                    else:
                        for p in obj.parameters():
                            p.requires_grad = True

        # Freeze item image encoder
        if freeze_item_image:
            for p in self.model.visual.parameters():
                p.requires_grad = False
            if unlock_layers > 0:
                for p in self.model.visual.transformer.resblocks[-unlock_layers:].parameters():
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
            if unlock_layers > 0:
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

    def encode_item(self, text_ids: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        text_feats = self.encode_text(text_ids, use_query_encoder=False)
        image_feats = self.encode_image(images)
        item_emb = self.attention_fusion(text_feats, image_feats)
        
        if self.normalize:
            item_emb = F.normalize(item_emb, dim=-1)
        return item_emb

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
        item_emb = self.encode_item(batch["title_ids"], batch["images"])
        dummy_alpha = torch.full((2,), 0.5, device=query_emb.device)
        return self.in_batch_negative_sampling(query_emb, item_emb), dummy_alpha
    
    def evaluate(self, batch: dict) -> torch.Tensor: 
        query_emb = self.encode_text(batch["query_ids"], use_query_encoder=True)
        item_emb = self.encode_item(batch["title_ids"], batch["images"])
        similarity = F.cosine_similarity(query_emb, item_emb, dim=-1)
        return similarity