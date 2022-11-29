import torch as T
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import GaussianBlur
from typing import *


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

    @staticmethod
    def np_to_tensor(array: np.ndarray) -> T.Tensor:
        # stack array over RGB channels
        return T.from_numpy(np.repeat(array[None, ...], 3, axis=0))

    def apply(self, img: np.ndarray) -> np.ndarray:
        tensor = self.np_to_tensor(img)
        features = self.apply_to_tensor(tensor)
        return features.detach().numpy()

    def apply_to_tensor(self, t: T.Tensor) -> T.Tensor:
        batch = self.preprocess(t).unsqueeze(0)
        features = self.model(batch).squeeze()
        if self.softmax:
            features = features.softmax(0)
        return features

    @property
    def n_features(self) -> int:
        return 2048 if self.disable_last_layer else 1000

    def top_k_classes(self, t: T.Tensor, k: int) -> List[str]:
        features = self.apply_to_tensor(t)
        top_class_ids = [int(x) for x in (-features).argsort()[:k]]
        labels = [f"{self.classify(class_id)}: {features[class_id].item(): .1f}"
                  for class_id in top_class_ids]
        return labels

    def classify(self, class_id: int) -> str:
        if self.disable_last_layer:
            raise AttributeError("ResnetFeaturizer with last layer disabled cannot classify")
        elif not isinstance(class_id, int):
            raise ValueError("class_id must be an integer")
        else:
            return self.categories[class_id]


class BlurredResnetFeaturizer(Featurizer):

    def __init__(self, resnet: ResnetFeaturizer, kernel_size: int):
        self.resnet = resnet
        self.gaussian_blur = GaussianBlur(kernel_size)

    def apply(self, img: np.ndarray) -> np.ndarray:
        tensor = T.from_numpy(np.repeat(img[None, ...], 3, axis=0))  # stack array over RGB channels
        blurred = self.gaussian_blur(tensor)
        return self.resnet.apply_to_tensor(blurred).detach().numpy()

    @property
    def n_features(self) -> int:
        return self.resnet.n_features


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
