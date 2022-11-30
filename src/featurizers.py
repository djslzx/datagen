import torch as T
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import GaussianBlur
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

    def __init__(self, disable_last_layer=False, softmax=True):
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
        self.softmax = softmax

    @property
    def n_features(self) -> int:
        return 2048 if self.disable_last_layer else 1000

    def apply(self, img: np.ndarray) -> np.ndarray:
        assert len(img.shape) <= 3
        assert isinstance(img, np.ndarray), f"Expected ndarray, but got {type(img)}"

        # resnet only plays nice with uint8 matrices
        if img.dtype != np.uint8:
            print(f"WARNING: casting image of type {img.dtype} to uint8", file=stderr)
            img = img.astype(np.uint8)

        # handle alpha channels, grayscale images -> rgb
        if len(img.shape) == 2:
            img = util.stack_repeat(img, 3)
        elif img.shape[0] != 3:
            img = util.stack_repeat(img[0], 3)

        # run resnet
        batch = self.preprocess(T.from_numpy(img)).unsqueeze(0)
        features = self.model(batch).squeeze()
        if self.softmax:
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
