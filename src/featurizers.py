import torch as T
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from typing import *
from sys import stderr
import util


class Featurizer:
    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        raise NotImplementedError


class ResnetFeaturizer(Featurizer):

    def __init__(self, disable_last_layer=False, softmax_outputs=True):
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

    def apply(self, img: np.ndarray) -> np.ndarray:
        assert len(img.shape) in [2, 3], f"Got imgs with shape {img.shape}"
        assert isinstance(img, np.ndarray), f"Expected ndarray, but got {type(img)}"

        # resnet only plays nice with uint8 matrices
        if img.dtype != np.uint8:
            print(f"WARNING: casting image of type {img.dtype} to uint8", file=stderr)
            img = img.astype(np.uint8)

        # handle alpha channels, grayscale images -> rgb
        if len(img.shape) == 2:  # images with no color channel
            img = util.stack_repeat(img, 3)
        elif img.shape[0] != 3:  # remove alpha channel -> grayscale w/ 3 channels (rgb)
            img = util.stack_repeat(img[0], 3)

        # run resnet
        batch = self.preprocess(T.from_numpy(img)).unsqueeze(0)
        features = self.model(batch).squeeze()
        if self.softmax_outputs:
            features = features.softmax(0)
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

    def apply(self, img: np.ndarray) -> np.ndarray:
        assert img.shape == (self.n_rows, self.n_cols), \
            f"Found image of shape {img.shape}, but expected [{self.n_rows}, {self.n_cols}]"

        # reshape img to column vector
        vec = img.reshape(self.n_rows * self.n_cols)

        # map values to [0, 1]
        return vec / 255


class DummyFeaturizer(Featurizer):

    def __init__(self):
        pass

    def apply(self, img: np.ndarray) -> np.ndarray:
        return np.array([np.mean(img), np.var(img)])

    @property
    def n_features(self) -> int:
        return 2
