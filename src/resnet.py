"""
Prototype code for distinguishing between synthetic and 'real' examples.
"""
import torch as T
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pdb
from glob import glob
from pprint import pp
from typing import Dict, List, Set, Tuple


def featurize(img_paths: List[str]) -> Dict[str, T.Tensor]:
    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)
    model = T.nn.Sequential(*list(resnet.children())[:-1])  # disable last layer in resnet
    model.eval()
    preprocess = weights.transforms()
    out = {}
    for img_path in img_paths:
        basename = img_path.split("/")[-1]
        img = read_image(img_path)
        if img.shape[0] == 4:
            img = img[:-1, :, :]  # cut out alpha channel
        batch = preprocess(img).unsqueeze(0)
        prediction = model(batch).squeeze().softmax(0)
        out[basename] = prediction
    return out


def classify(predictions: Dict[str, T.Tensor]) -> Dict[str, Tuple[str, float]]:
    weights = ResNet50_Weights.DEFAULT
    out = {}
    for name, prediction in predictions.items():
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]
        out[name] = (category_name, score)
    return out


def name(filename: str) -> str:
    basename = filename.split("/")[-1]
    num_str = basename
    for x in ['render', 'system-', 'ref-']:
        num_str = num_str.replace(x, '')
    num = int(num_str.split("-")[0])
    return str(num)


def plot_pca(n_points: int, globs: List[str], markers: List[str], legend: List[str]):
    pts = []
    for i, glob_str in enumerate(globs):
        paths = sorted(glob(glob_str))[:n_points]
        predictions = featurize(paths)
        points = [x.detach().numpy() for x in predictions.values()]
        labels = [name(path) for path in paths]

        pca = PCA(n_components=2)
        pcomps = pca.fit_transform(points)
        xs, ys = [x for x, y in pcomps], [y for x, y in pcomps]
        pts.append(plt.scatter(x=xs, y=ys, marker=markers[i]))

        for label, x, y in zip(labels, xs, ys):
            plt.annotate(text=label, xy=(x, y))

    plt.legend(pts, legend)


if __name__ == "__main__":
    n_points = 100
    glob_strs = [
        "../out/io-samples/ref/png/*.png",
        "../out/io-samples/png/system*.png",
        "../out/codex-samples/text/renders/png/*.png",
        "../out/codex-samples/code/renders/png/*.png"
    ]
    markers = ['o', 'x', '^', 'v']
    legend = ['Reference', 'Synthetic IO', 'text-davinci-002', 'code-davinci-002']
    plot_pca(n_points, glob_strs, markers, legend)
    plt.show()
