from typing import *
import torch as T
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights, quantization
from transformers import AutoTokenizer, AutoModelForCausalLM  # language models
from sentence_transformers import SentenceTransformer
from sys import stderr
from scipy.spatial import distance as dist
from einops import rearrange
import Levenshtein


class Featurizer:
    def apply(self, batch: Any) -> np.ndarray:
        """Applies the feature extractor to a batch of inputs"""
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        """Returns the number of features per input"""
        raise NotImplementedError


class TextClassifier(Featurizer):

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def apply(self, batch: List[str]) -> np.ndarray:
        v = self.model.encode(batch)  # [batch, seq, feat]
        assert v.shape[-1] == self.n_features
        return v

    @property
    def n_features(self) -> int:
        return 384


class TextPredictor(Featurizer):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B")
        self.model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B")

    def apply(self, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        raise NotImplementedError


class ResnetFeaturizer(Featurizer):

    def __init__(self, quantize=False, disable_last_layer=False, softmax_outputs=True):
        self.quantize = quantize
        if quantize:
            weights = quantization.ResNet50_QuantizedWeights.DEFAULT
            resnet = quantization.resnet50(weights=weights, quantize=True)
        else:
            weights = ResNet50_Weights.DEFAULT
            resnet = resnet50(weights=weights)

        self.preprocess = weights.transforms()
        self.disable_last_layer = disable_last_layer
        if disable_last_layer:
            self.model = T.nn.Sequential(*list(resnet.children())[:-1])
        else:
            self.model = resnet
            self.categories = weights.meta["categories"]
        self.model.eval()
        self.softmax_outputs = softmax_outputs

    def __repr__(self) -> str:
        return ("<ResnetFeaturizer: "
                f"disable_last_layer={self.disable_last_layer}, softmax_outputs={self.softmax_outputs}>")

    @property
    def n_features(self) -> int:
        return 2048 if self.disable_last_layer else 1000

    def apply(self, batch: List[np.ndarray]) -> np.ndarray:
        batch = np.stack(batch)
        assert isinstance(batch, np.ndarray), f"Expected ndarray, but got {type(batch)}"
        assert len(batch.shape) == 4, f"Expected shape [b, c, w, h] but got {batch.shape}"

        # resnet only plays nice with uint8 matrices
        if batch.dtype != np.uint8:
            print(f"WARNING: casting image of type {batch.dtype} to uint8", file=stderr)
            batch = batch.astype(np.uint8)

        # run resnet
        batch = self.preprocess(T.from_numpy(batch))
        features = self.model(batch).squeeze()  # doesn't depend on whether last layer is removed

        if self.softmax_outputs:
            features = features.softmax(-1)
        return features.detach().numpy()

    def top_k_classes(self, features: np.ndarray, k: int) -> List[str]:
        top_class_ids = [int(x) for x in (-features).argsort()[:k]]
        labels = [f"{self.classify(class_id)}: {features[class_id].item(): .4f}"
                  for class_id in top_class_ids]
        return labels

    def classify(self, class_id: int) -> str:
        if self.disable_last_layer:
            raise AttributeError("ResnetFeaturizer with last layer disabled cannot classify")
        elif not isinstance(class_id, int):
            raise ValueError("class_id must be an integer")
        else:
            return self.categories[class_id]


class RawFeaturizer(Featurizer):
    """
    Treat each input image as a feature vector
    """
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def __repr__(self) -> str:
        return f"<RawFeaturizer: shape=[{self.n_rows}, {self.n_cols}]>"

    @property
    def n_features(self) -> int:
        return self.n_rows * self.n_cols

    def apply(self, batch: np.ndarray) -> np.ndarray:
        assert batch.shape == (self.n_rows, self.n_cols), \
            f"Found image of shape {batch.shape}, but expected [{self.n_rows}, {self.n_cols}]"

        # reshape x to column vector
        vec = batch.reshape(self.n_rows * self.n_cols)

        # map values to [0, 1]
        return vec / 255


class DummyFeaturizer(Featurizer):

    def __init__(self):
        pass

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.array([np.mean(x), np.var(x)])

    @property
    def n_features(self) -> int:
        return 2


if __name__ == "__main__":
    C = TextClassifier()
    corpus = [
        "$100",
        "$10,000",
        "$10b",
        "$10 billion",
        "$10,000,000,000",
        "10 TB",
        "71 MB",
        "100000000000 KB",
        "0.001 MB",
        "johndoe@mail.org",
        "ellisk@cs.cornell.edu",
        "djsl@cs.cornell.edu",
        "djl328@cornell.edu",
        "djl5@williams.edu",
        "ab1@williams.edu",
        "zz11@williams.edu",
        "2023/10/01",
        "2023/03/08",
        "1970/01/01",
    ]
    embeddings = C.apply(corpus)
    with open("../out/embedding_dist.txt", "w") as f_embed, open("../out/leven_dist.txt", "w") as f_leven:
        for a, u in zip(corpus, embeddings):
            embed_sort = sorted([(dist.minkowski(u, v), b) for b, v in zip(corpus, embeddings)])
            leven_sort = sorted([(Levenshtein.distance(a, b), b) for b in corpus])

            f_embed.write(f"{a}:\n")
            for _, b in embed_sort:
                f_embed.write(f"  {b}\n")

            f_leven.write(f"{a}:\n")
            for _, b in leven_sort:
                f_leven.write(f"  {b}\n")
