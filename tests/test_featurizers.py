from torchvision.io import read_image, ImageReadMode
from sklearn.neighbors import NearestNeighbors
from typing import *
from os import listdir
from featurizers import *
from lang.lindenmayer import S0LSystem
from util import plot_image_grid

N_ROWS = 128
N_COLS = 128


def get_images(popn: List[Tuple[S0LSystem, float]], n_samples: int) -> np.ndarray:
    images = np.empty((len(popn), n_samples, N_ROWS, N_COLS), dtype=np.uint8)
    for i, (sys, angle) in enumerate(popn):
        for j in range(n_samples):
            images[i, j] = sys.draw(sys.nth_expansion(3), d=3, theta=angle, n_rows=N_ROWS, n_cols=N_COLS)
    return images


def get_features(featurizer: Featurizer, images: np.ndarray) -> np.ndarray:
    features = np.empty((images.shape[0], images.shape[1], featurizer.n_features))
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            features[i, j] = featurizer.apply(images[i, j])
    return features


def check_resnet_classifier(popn: List[Tuple[S0LSystem, float]], n_samples: int):  # pragma: no cover
    """
    Make sure that resnet classes for each L-system render look reasonable
    """
    classifier = ResnetFeaturizer(disable_last_layer=False, softmax_outputs=True)
    images = get_images(popn, n_samples)
    features = get_features(classifier, images)
    n = images.shape[0]
    labels = np.empty((n, n_samples), dtype=object)
    for i in range(n):
        for j in range(n_samples):
            class_labels = classifier.top_k_classes(features[i, j], k=3)
            labels[i, j] = "\n".join(class_labels)

    # check resnet classes
    for i in range(n):
        plot_image_grid(imgs=images[i], shape=(1, n_samples), labels=labels[i])


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
        plot_image_grid(imgs=[img[0]], shape=(1, 1), labels=["\n".join(labels)])


def check_featurizer(featurizer: Featurizer, popn: List[Tuple[S0LSystem, float]], n_samples: int,
                     n_neighbors: int):  # pragma: no cover
    """
    Make sure that the k nearest neighbors of each L-system look reasonable
    """
    n = len(popn)
    images = get_images(popn, n_samples)
    features = get_features(featurizer, images).reshape((n, n_samples * featurizer.n_features))
    features_knn = NearestNeighbors(n_neighbors=n_neighbors).fit(features)
    for i in range(n):
        distances, indices = features_knn.kneighbors(features[i].reshape(1, -1))  # add empty dim
        distances, indices = distances[0], indices[0]  # knn takes in groups of points instead of single points
        neighbor_indices = indices[distances.argsort()]
        all_imgs = np.concatenate((images[i].reshape(1, n_samples, N_ROWS, N_COLS),
                                   images[neighbor_indices]), axis=0).reshape((-1, N_ROWS, N_COLS))
        plot_image_grid(title=str(featurizer),
                        imgs=all_imgs,
                        shape=(n_samples, n_neighbors + 1),
                        labels=[""] + [f"{x:.4e}" for x in np.repeat(distances, n_samples).tolist()])


def test_raw_featurizer():
    images = np.random.randint(0, 255, size=(10, 128, 128, 3), dtype=np.uint8)
    featurizer = RawFeaturizer(128, 128)
    features = featurizer.apply(images)
    assert features.shape == (10, 128 * 128 * 3), f"Expected shape (10, 128 * 128 * 3), but got {features.shape}"

    rearranged = rearrange(images, "b h w c -> b (h w c)", b=10, h=128, w=128, c=3)
    assert np.all(features == rearranged / 255), "Expected features to be equal to images"


if __name__ == '__main__':  # pragma: no cover
    seed = [
        (S0LSystem.from_sentence(list(string)), 45)
        for string in [
            "F;F~[[+[-+[FF]+F[F]][F-][[F-]F]F]-F][F-]FF",
            "F;F~F+[F]F[FF][F][-[F]]",
            "FF;F~[[F]],F~-F[FF[F[[[[FF-]F][+[[F-]]]]]]]",
            "+F;F~F-F[+++F+[-F]F[F-+[---+FF-]F]F++-][[F]],F~F",
            "-F;F~F",
            "F;F~F",
            "F;F~F-",
            "F+F;F~F",
            "F+F;F~FF",
        ]
    ]
    # check_resnet_with_images("../resnet-tests/screenshots")
    check_resnet_classifier(popn=seed, n_samples=1)
    # check_featurizer(featurizer=ResnetFeaturizer(disable_last_layer=False, softmax_outputs=True),
    #                  popn=seed, n_samples=1, n_neighbors=len(seed))
    # check_featurizer(featurizer=ResnetFeaturizer(disable_last_layer=True, softmax_outputs=False),
    #                  popn=seed, n_samples=1, n_neighbors=len(seed))
    # check_featurizer(featurizer=RawFeaturizer(N_ROWS, N_COLS),
    #                  popn=seed, n_samples=1, n_neighbors=len(seed))
