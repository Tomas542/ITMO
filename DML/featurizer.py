import gensim
import numpy as np
import pandas as pd
import torchaudio
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor
from torchaudio import transforms as T


def clean_text(text: str) -> str:
    text = text.lower().strip()
    # remove punctuation
    # remove apostroph
    # remove stopwords
    return text


def tokenize(text: str) -> list[str]:
    return text.split()


def get_sentence_mean_vec(w2v, text: list[str]) -> np.ndarray:
    vecs = []
    for word in text:
        try:
            vec = w2v.get_vector(word)
        except Exception:
            vec = [0] * w2v.vector_size
        vecs.append(vec)
    return np.mean(vecs, axis=0)


train_df = pd.read_csv("data/Data_Train_modified.csv")
val_df = pd.read_csv("data/Data_Val_modified.csv")
test_df = pd.read_csv("data/Data_Test_modified.csv")

# for text features
train_text = train_df["text"].apply(clean_text).to_list()
val_text = val_df["text"].apply(clean_text).to_list()
test_text = test_df["text"].apply(clean_text).to_list()

# for audio features
train_wavs = None
val_wavs = None
test_wavs = None

# TF-iDF
tf_idf = TfidfVectorizer(min_df=0.005, max_df=0.995, max_features=450)
train_tf_idf = tf_idf.fit_transform(train_text)
val_tf_idf = tf_idf.transform(val_text)
test_tf_idf = tf_idf.transform(test_text)

save_npz("features/train_tfidf.npz", train_tf_idf)
save_npz("features/val_tfidf.npz", val_tf_idf)
save_npz("features/test_tfidf.npz", test_tf_idf)

train_tokenized = [tokenize(text) for text in train_text]
val_tokenized = [tokenize(text) for text in val_text]
test_tokenized = [tokenize(text) for text in test_text]


# Word2Vec
w2v = gensim.models.Word2Vec(train_tokenized, vector_size=250, min_count=2, window=3)

train_w2v = [get_sentence_mean_vec(w2v, text) for text in train_tokenized]
val_w2v = [get_sentence_mean_vec(w2v, text) for text in train_tokenized]
test_w2v = [get_sentence_mean_vec(w2v, text) for text in train_tokenized]

np.save("features/train_w2v.npy", np.array(train_w2v))
np.save("features/val_w2v.npy", np.array(val_w2v))
np.save("features/test_w2v.npy", np.array(test_w2v))


# LogMel
class LogMelSpectrogram(T.MelSpectrogram):
    def __init__(self, eps=1e-8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = eps

    def forward(self, wav_path: str) -> Tensor:
        waveform, sr = torchaudio.load(wav_path, normalize=True)
        if sr != 16_000:
            waveform = torchaudio.functional.resample(waveform, sr, 16_000)
        log_mel_spec = (super().forward(waveform) + self.eps).log()
        return log_mel_spec.mean(dim=-1)


mel = LogMelSpectrogram(
    sample_rate=16000,
    n_fft=2048,
    hop_length=128,
    n_mels=128,
    f_min=40,
    f_max=8000,
    mel_scale="slaney",
)

train_mel = [mel(wav_path).numpy() for wav_path in train_wavs]
val_mel = [mel(wav_path).numpy() for wav_path in val_wavs]
test_mel = [mel(wav_path).numpy() for wav_path in test_wavs]

np.save("features/train_mel.npy", np.array(train_mel))
np.save("features/val_mel.npy", np.array(val_mel))
np.save("features/test_mel.npy", np.array(test_mel))


# MFCC
class MFCC(T.MFCC):
    def __init__(self, eps=1e-8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = eps

    def forward(self, wav_path: str) -> Tensor:
        waveform, sr = torchaudio.load(wav_path, normalize=True)
        if sr != 16_000:
            waveform = torchaudio.functional.resample(waveform, sr, 16_000)
        mfcc = super().forward(waveform) + self.eps
        return mfcc.mean(dim=-1)


mfcc = MFCC(
    sample_rate=16_000,
    n_mfcc=40,
)

train_mfcc = [mfcc(wav_path).numpy() for wav_path in train_wavs]
val_mfcc = [mfcc(wav_path).numpy() for wav_path in val_wavs]
test_mfcc = [mfcc(wav_path).numpy() for wav_path in test_wavs]

np.save("features/train_mfcc.npy", np.array(train_mfcc))
np.save("features/val_mfcc.npy", np.array(val_mfcc))
np.save("features/test_mfcc.npy", np.array(test_mfcc))
