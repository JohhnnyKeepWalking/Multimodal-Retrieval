import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, Any


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
            "image": row["local_path"],
            "desirability_label": row["desirability_label"],
            "relevance_label": row["relevance_label"],
        }
