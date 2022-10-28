"""
Prototype code for distinguishing between synthetic and 'real' examples.
"""
import torch as T
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
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


if __name__ == "__main__":
    for s in ["../out/io-samples/pngs/ref*.png",
              "../out/io-samples/pngs/system*.png"]:
        paths = sorted(glob(s))
        preds = process_images(paths)
        # pp({k: v.shape for k, v in preds.items()})
        classified = classify(preds)
        pp(classified)
        print('classes:', classes(classified))
