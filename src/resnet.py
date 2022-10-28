"""
Prototype code for distinguishing between synthetic and 'real' examples.
"""

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
import matplotlib.pyplot as plt
import pdb
import glob
from pprint import pp
from typing import Dict, List, Set, Tuple


def classify(file_glob: str, verbose=False) -> Dict:
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    class_map = {}
    for img_path in sorted(glob.glob(file_glob)):
        name = img_path.split("/")[-1]
        img = read_image(img_path)
        if img.shape[0] == 4:
            img = img[:-1, :, :]  # cut out alpha channel

        batch = preprocess(img).unsqueeze(0)
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]

        if verbose: print(f"{name}: {category_name} @ {100 * score:.1f}%")
        class_map[name] = (category_name, score)
    return class_map


def extract_classes(d: Dict[str, Tuple[str, float]]) -> Set[str]:
    return {cname for cname, score in d.values()}


if __name__ == "__main__":
    reference_glob = "../out/io-samples/pngs/ref*.png"
    synthetic_glob = "../out/io-samples/pngs/system*.png"

    ref_dict = classify(reference_glob, verbose=True)
    synth_dict = classify(synthetic_glob, verbose=True)

    print('ref dict:')
    pp(ref_dict)
    print('synth dict:')
    pp(synth_dict)

    print('reference classes:', extract_classes(ref_dict))
    print('synthetic classes:', extract_classes(synth_dict))

