import pdb

from featurizers import *
from book_zoo import zoo
from torchvision.io import read_image, ImageReadMode
from sklearn.neighbors import NearestNeighbors
from typing import *
from lindenmayer import S0LSystem
from os import listdir
import util


def check_resnet_classifier(popn: List[Tuple[S0LSystem, float]], n_samples: int):  # pragma: no cover
    """
    Make sure that resnet classes for each L-system render look reasonable
    """
    classifier = ResnetFeaturizer(disable_last_layer=False, softmax=True)
    n = len(popn)
    images = np.empty((n, n_samples, 128, 128))
    labels = np.empty((n, n_samples), dtype=object)
    for i, (sys, angle) in enumerate(popn):
        for j in range(n_samples):
            img = sys.draw(sys.nth_expansion(3), d=3, theta=angle, n_rows=128, n_cols=128)
            images[i, j] = img
            features = classifier.apply(img)
            class_labels = classifier.top_k_classes(features, k=3)
            labels[i, j] = "\n".join(class_labels)

    # check resnet classes
    for i in range(n):
        util.plot(imgs=images[i], shape=(1, n_samples), labels=labels[i])


def check_resnet_with_images(dirpath: str):  # pragma: no cover
    """
    Make sure that resnet gives reasonable answers to stock images
    """
    classifier = ResnetFeaturizer()
    for img_file in listdir(dirpath):
        print(f"Testing image {img_file}")
        img = read_image(f"{dirpath}/{img_file}", mode=ImageReadMode.GRAY).numpy()
        print(f"  shape: {img.shape}")
        features = classifier.apply(img)
        labels = classifier.top_k_classes(features, 3)
        util.plot(imgs=[img[0]], shape=(1, 1), labels=["\n".join(labels)])


def check_resnet_featurizer(popn: List[Tuple[S0LSystem, float]], n_samples: int):  # pragma: no cover
    """
    Make sure that the k nearest neighbors of each L-system look reasonable
    """
    featurizer = ResnetFeaturizer(disable_last_layer=True, softmax=False)  # softmax or not?
    n = len(popn)
    images = np.empty((n, n_samples, 128, 128))
    features = np.empty((n, n_samples, featurizer.n_features))
    for i, (sys, angle) in enumerate(popn):
        for j in range(n_samples):
            img = sys.draw(sys.nth_expansion(3), d=3, theta=angle, n_rows=128, n_cols=128)
            images[i, j] = img
            features[i, j] = featurizer.apply(img)

    # check feature neighbors
    n_neighbors = 5
    features = features.reshape((n, n_samples * featurizer.n_features))
    features_knn = NearestNeighbors(n_neighbors=n_neighbors).fit(features)
    for i in range(n):
        distances, indices = features_knn.kneighbors(features[i].reshape(1, -1))
        distances, indices = distances[0], indices[0]
        neighbor_indices = indices[distances.argsort()]
        all_imgs = np.concatenate((images[i].reshape(1, n_samples, 128, 128),
                                   images[neighbor_indices]), axis=0).reshape((-1, 128, 128))
        util.plot(imgs=all_imgs,
                  shape=(n_neighbors + 1, n_samples),
                  labels=[f"{x:.4e}" for x in [0.0] * n_samples + np.repeat(distances, n_samples).tolist()])


def check_resnet_preprocess(popn: List[Tuple[S0LSystem, float]], n_samples: int):  # pragma: no cover
    classifier = ResnetFeaturizer()
    blur_classifier = BlurredResnetFeaturizer(classifier, 5)
    for i, (sys, angle) in enumerate(popn):
        for j in range(n_samples):
            img = sys.draw(sys.nth_expansion(3), d=4, theta=angle, n_rows=128, n_cols=128)
            tensor_img = T.from_numpy(np.repeat(img[None, ...], 3, axis=0))  # stack array over RGB channels
            processed_img = classifier.preprocess(tensor_img)[0]
            blurred_img = blur_classifier.gaussian_blur(tensor_img)[0]
            util.plot(imgs=[img, processed_img, blurred_img],
                      shape=(1, 3),
                      labels=["Raw", "Processed", "Blurred"])


if __name__ == '__main__':  # pragma: no cover
    popn = [
        (S0LSystem.from_sentence(list(string)), 45)
        for string in [
            "F;F~[[+[-+[FF]+F[F]][F-][[F-]F]F]-F][F-]FF",
            "F;F~F+[F]F[FF][F][-[F]]",
            "FF;F~[[F]],F~-F[FF[F[[[[FF-]F][+[[F-]]]]]]]",
            "+F;F~F-F[+++F+[-F]F[F-+[---+FF-]F]F++-][[F]],F~F",
            "-F;F~F",
        ]
    ]

    # check_resnet_preprocess(popn=popn, n_samples=1)
    # check_resnet_with_images("../resnet-test/screenshots")
    check_resnet_classifier(popn=popn, n_samples=1)
    # check_resnet_featurizer(popn=zoo, n_samples=1)
