import pdb

import torch as T
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights


class Featurizer:
    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        raise NotImplementedError


class ResnetFeaturizer(Featurizer):

    def __init__(self, disable_last_layer=False, softmax=True):
        self.disable_last_layer = disable_last_layer
        weights = ResNet50_Weights.DEFAULT
        self.preprocess = weights.transforms()
        resnet = resnet50(weights=weights)
        if disable_last_layer:
            self.model = T.nn.Sequential(*list(resnet.children())[:-1])
        else:
            self.model = resnet
            self.categories = weights.meta["categories"]
        self.model.eval()
        self.softmax = softmax

    def apply(self, img: np.ndarray) -> np.ndarray:
        tensor = T.from_numpy(np.repeat(img[None, ...], 3, axis=0))  # stack array over RGB channels
        batch = self.preprocess(tensor).unsqueeze(0)
        features = self.model(batch).squeeze()
        if self.softmax:
            features = features.softmax(0)
        return features.detach().numpy()

    @property
    def n_features(self) -> int:
        return 2048 if self.disable_last_layer else 1000

    def classify(self, class_id: int) -> str:
        if self.disable_last_layer:
            raise AttributeError("ResnetFeaturizer with last layer disabled cannot classify")
        elif not isinstance(class_id, int):
            raise ValueError("class_id must be an integer")
        else:
            return self.categories[class_id]


class DummyFeaturizer(Featurizer):

    def __init__(self):
        pass

    def apply(self, img: np.ndarray) -> np.ndarray:
        return np.array([np.mean(img), np.var(img)])

    @property
    def n_features(self) -> int:
        return 2


class SyntacticSemanticFeaturizer(Featurizer):

    def __init__(self):
        raise NotImplementedError

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        raise NotImplementedError
