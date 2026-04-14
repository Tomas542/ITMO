from enum import StrEnum

from torch.utils.data import DataLoader, Dataset


class FeatureType(StrEnum):
    TFIDF = "tf_df"
    SENT = "sent"
    MEL = "mel"
    MFCC = "mfcc"


class SplitType(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SentimentDataset(Dataset):
    def __init__(self, feature_type: FeatureType, split: SplitType) -> None:
        match feature_type:
            case FeatureType.TFIDF:
                pass
            case FeatureType.SENT:
                pass
            case FeatureType.MEL:
                pass
            case FeatureType.MFCC:
                pass
        self.data = ...
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
