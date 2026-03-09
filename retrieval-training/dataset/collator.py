import torch
import open_clip
from PIL import Image
import torchvision.transforms as T
from typing import Dict, Any, List


class Collator:
    def __init__(self, image_size: int = 224, clip_model_name: str = "ViT-B-32"):
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        self.tf = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),  # CLIP defaults
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries = [b["query"] for b in batch]
        titles   = [b["title"] for b in batch]
        paths   = [b["image"] for b in batch]

        imgs = []
        for p in paths:
            with Image.open(p) as im:
                imgs.append(self.tf(im.convert("RGB")))
        images = torch.stack(imgs, dim=0)

        query_ids = self.tokenizer(queries)
        title_ids = self.tokenizer(titles)

        desirability = torch.tensor([b["desirability_label"] for b in batch])
        relevance = torch.tensor([b["relevance_label"] for b in batch])

        return {
            "query_ids": query_ids,
            "title_ids": title_ids,
            "images": images,
            "desirability_label": desirability,
            "relevance_label": relevance,
        }
