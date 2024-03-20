import pdb
import json
from typing import *
import matplotlib.pyplot as plt
import torch as T
import numpy as np
from sklearn import manifold
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoTokenizer, AutoModelForCausalLM, CodeGenModel  # language models
from sentence_transformers import SentenceTransformer
from sys import stderr
from scipy.spatial import distance as dist
from einops import rearrange
import Levenshtein
from skimage import filters

import util

# set default device to cuda if available
if T.cuda.is_available():
    T.set_default_device('cuda')


class Featurizer:
    def apply(self, batch: Any) -> np.ndarray:
        """Applies the feature extractor to a batch of inputs"""
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        """Returns the number of features per input"""
        raise NotImplementedError


class SentenceFeaturizer(Featurizer):

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def apply(self, batch: List[str]) -> np.ndarray:
        v = self.model.encode(batch)  # [batch, seq, feat]
        assert v.shape[-1] == self.n_features
        return v

    @property
    def n_features(self) -> int:
        return 384


class CodeGen(Featurizer):

    def __init__(self, size: str):
        assert size in {"350M", "2B", "6B", "16B"}, f"Invalid size {size}"
        checkpoint = f"Salesforce/codegen-{size}-mono"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = CodeGenModel.from_pretrained(checkpoint)
        self.model.eval()

    def apply(self, batch: List[str]) -> np.ndarray:
        inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        # extract the last layer hidden state for the last token
        return outputs.last_hidden_state[:, -1].detach().numpy()


class StarCoder(Featurizer):

    def __init__(self):
        checkpoint = "bigcode/starcoder"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.model.eval()

    def apply(self, batch: List[str]) -> np.ndarray:
        inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, -1].detach().numpy()


class PolyCoder(Featurizer):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B")
        self.model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B")

    def apply(self, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        raise NotImplementedError


class ResnetFeaturizer(Featurizer):

    def __init__(self, disable_last_layer=True, softmax_outputs=False, center=False, sigma=0.):
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights, antialias=True)

        self.center = center
        self.preprocess = weights.transforms()
        self.disable_last_layer = disable_last_layer
        if disable_last_layer:
            self.model = T.nn.Sequential(*list(resnet.children())[:-1])
        else:
            self.model = resnet
            self.categories = weights.meta["categories"]
        self.model.eval()
        self.softmax_outputs = softmax_outputs
        self.sigma = sigma

        if T.cuda.is_available():
            self.device = T.device("cuda")
            print("CUDA is available. Using GPU.")
        else:
            self.device = T.device("cpu")
            print("CUDA is not available. Using CPU.")


    def __repr__(self) -> str:
        return ("<ResnetFeaturizer: "
                f"disable_last_layer={self.disable_last_layer}, softmax_outputs={self.softmax_outputs}>")

    @property
    def n_features(self) -> int:
        return 2048 if self.disable_last_layer else 1000

    def apply(self, batch: List[np.ndarray]) -> np.ndarray:
        # center
        if self.center:
            batch = [util.center_image(img) for img in batch]

        # gaussian filter
        if self.sigma > 0.:
            batch = [filters.gaussian(img, sigma=self.sigma, channel_axis=-1) for img in batch]
            batch = np.stack(batch) * 255  # compensate for gaussian blur output in [0, 1]
            batch = batch.astype(np.uint8)
        elif isinstance(batch, List):
            batch = np.stack(batch)

        assert len(batch.shape) == 4, f"Expected shape [b h w c] but got {batch.shape}"
        assert batch.shape[-1] in {3, 4}, f"Expected 3 or 4 channels but got {batch.shape[-1]} in shape {batch.shape}"

        # resnet only plays nice with uint8 matrices
        if batch.dtype != np.uint8:
            print(f"WARNING: casting image of type {batch.dtype} to uint8", file=stderr)
            batch = batch.astype(np.uint8)

        # run resnet
        batch = T.from_numpy(rearrange(batch[..., :3], "b h w c -> b c h w"))  # remove alpha channel, reshape
        batch = self.preprocess(batch).to(self.device)
        features = self.model(batch).squeeze()  # doesn't depend on whether last layer is removed

        # softmax
        if self.softmax_outputs:
            features = features.softmax(-1)

        return features.detach().cpu().numpy()

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
        if isinstance(batch, List):
            batch = np.stack(batch)

        assert batch.ndim == 4, f"Expected batch of shape [b h w c] but got {batch.shape}"
        assert batch.shape[1:] == (self.n_rows, self.n_cols, 3), \
            f"Expected batch of shape [b {self.n_rows} {self.n_cols} 3] but got {batch.shape}"
        assert batch.dtype == np.uint8, f"Expected batch of type uint8 but got {batch.dtype}"

        # reshape x to column vector and map to [0, 1]
        return rearrange(batch, "b h w c -> b (h w c)") / 255


class DummyFeaturizer(Featurizer):

    def __init__(self):
        pass

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.array([np.mean(x), np.var(x)])

    @property
    def n_features(self) -> int:
        return 2


def check_embeddings():
    C = SentenceFeaturizer()
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


def _vet_featurizer_on_humaneval(ft: Featurizer, embed_keys=None, label_key=None):
    name = ft.__class__.__name__
    if not embed_keys:
        embed_keys = ["prompt", "canonical_solution"]
    if not label_key:
        label_key = "entry_point"

    # load jsonl file from "../datasets/HumanEval.jsonl"
    inputs = []
    labels = []
    for line in open("../datasets/HumanEval.jsonl", "r").readlines():
        d = json.loads(line)
        inputs.append("\n".join([d[k] for k in embed_keys]))
        labels.append(d[label_key])

    embeddings = ft.apply(inputs)
    print(embeddings.shape)

    mds = manifold.MDS(n_components=2, random_state=0)
    points = mds.fit_transform(embeddings)
    util.plot_labeled_points(
        points[:, 0],
        points[:, 1],
        labels=labels,
        fontsize=5,
        title=f"MDS on {name}, embed w/ {embed_keys}, label={label_key}"
    )
    plt.savefig(f"../out/mds{name}_emb={embed_keys}_lab={label_key}.png")
    plt.show()

    # tsne
    for perplexity in [2, 5, 30, 50, 100]:
        tsne = manifold.TSNE(
            n_components=2,
            n_iter=5000,
            perplexity=perplexity,
            n_iter_without_progress=150,
            n_jobs=2,
        )
        points = tsne.fit_transform(embeddings)
        util.plot_labeled_points(
            points[:, 0],
            points[:, 1],
            labels=labels,
            fontsize=5,
            title=f"t-SNE on {name}, embed w/ {embed_keys}, label={label_key}, perplexity={perplexity}",
        )
        plt.savefig(f"../out/tsne_{name}_emb={embed_keys}_lab={label_key}_perp={perplexity}.png")
        plt.show()


if __name__ == "__main__":
    ft = SentenceFeaturizer()
    _vet_featurizer_on_humaneval(
        ft,
        embed_keys=["prompt", "canonical_solution"],
        label_key="entry_point",
    )
    _vet_featurizer_on_humaneval(
        ft,
        embed_keys=["prompt"],
        label_key="entry_point",
    )
