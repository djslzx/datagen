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


def process_images(img_paths: List[str]) -> Dict[str, T.Tensor]:
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    out = {}
    for img_path in img_paths:
        basename = img_path.split("/")[-1]
        img = read_image(img_path)
        if img.shape[0] == 4:
            img = img[:-1, :, :]  # cut out alpha channel
        batch = preprocess(img).unsqueeze(0)
        prediction = model(batch).squeeze(0).softmax(0)
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


def classes(classifications: Dict[str, Tuple[str, float]]) -> Set[str]:
    return {cname for cname, score in classifications.values()}


def name(filename: str) -> str:
    basename = filename.split("/")[-1]
    num_str = basename
    for x in ['render', 'system-', 'ref-']:
        num_str = num_str.replace(x, '')
    num = int(num_str.split("-")[0])
    return str(num)


if __name__ == "__main__":
    n_points = 100
    pts = []
    markers = ['o', 'x', '^', 'v']
    for i, s in enumerate([
        "../out/io-samples/ref/png/*.png",
        "../out/io-samples/png/system*.png",
        "../out/codex-samples/text/renders/png/*.png",
        "../out/codex-samples/code/renders/png/*.png"
    ]):
        paths = sorted(glob(s))[:n_points]
        predictions = process_images(paths)
        points = [x.detach().numpy() for x in predictions.values()]
        labels = [name(path) for path in paths]

        pca = PCA(n_components=2)
        pcomps = pca.fit_transform(points)
        xs, ys = [x for x, y in pcomps], [y for x, y in pcomps]
        pts.append(plt.scatter(x=xs, y=ys, marker=markers[i]))

        for label, x, y in zip(labels, xs, ys):
            plt.annotate(text=label, xy=(x, y))

    plt.legend(pts, ('Reference', 'Synthetic IO', 'text-davinci-002', 'code-davinci-002'))
    plt.show()
