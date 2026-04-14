from enum import StrEnum

import numpy as np
import pandas as pd
import torch
from scipy import load_npz
from torch.utils.data import DataLoader, Dataset


class FeatureType(StrEnum):
    TFIDF = "tf_df"
    W2V = "w2v"
    MEL = "mel"
    MFCC = "mfcc"


class SplitType(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SentimentDataset(Dataset):
    def __init__(
        self,
        feature_type: FeatureType,
        split: SplitType,
        mosei_path: str = "/home/ext-yudin-a@ad.speechpro.com/dml/datasets/CMU-MOSEI/",
    ) -> None:
        match feature_type:
            case FeatureType.TFIDF:
                data = load_npz(f"features/{split}_tfidf.npz")
            case FeatureType.W2V:
                data = np.load(f"features/{split}_w2v.npy")
            case FeatureType.MEL:
                data = np.load(f"features/{split}_mel.npy")
            case FeatureType.MFCC:
                data = np.load(f"features/{split}_mfcc.npy")
        data = torch.from_numpy(data)
        suffix = "original" if split == "test" else "modified"
        split_name = split[0].upper() + split[1:]
        df = pd.read_csv(mosei_path + f"/Data_{split_name}_{suffix}.csv")
        sentiment = df["sentiment"].to_list()
        self.data = list(zip(data, sentiment))
        self.sent_cls = {"negative": 0, "neutral": 1, "positive": 2}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        features, sentiment = self.data[index]
        if sentiment < 0:
            label = self.sent_cls["negative"]
        elif sentiment > 0:
            label = self.sent_cls["positive"]
        else:
            label = self.sent_cls["neutral"]

        return features, label


def get_dataloader(encoding_type: FeatureType, split: SplitType) -> DataLoader:
    ds = SentimentDataset(encoding_type, split)
    return DataLoader(
        ds,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        shuffle=split == "train",
        drop_last=split == "train",
    )
