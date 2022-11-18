from featurizers import *
from book_zoo import zoo
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from typing import *


def check_resnet_classifier():  # pragma: no cover
    """
    Make sure that resnet classes for each L-system render look reasonable
    """
    classifier = ResnetFeaturizer(disable_last_layer=False, softmax=True)
    n = len(zoo)
    n_samples = 3
    images = np.empty((n, n_samples, 128, 128))
    labels = np.empty((n, n_samples), dtype=object)
    for i, (sys, angle) in enumerate(zoo):
        for j in range(n_samples):
            img = sys.draw(sys.nth_expansion(3), d=3, theta=angle, n_rows=128, n_cols=128)
            images[i, j] = img
            v = classifier.apply(img)
            class_ids = [int(x) for x in (-v).argsort()[:3]]
            labels[i, j] = ""
            for class_id in class_ids:
                label = classifier.classify(class_id)
                score = v[class_id].item()
                labels[i, j] += f"{label}: {score:.1f}\n"

    # check resnet classes
    for i in range(n):
        plot(images[i], shape=(1, n_samples), labels=labels[i])


def check_resnet_featurizer():  # pragma: no cover
    """
    Make sure that the k nearest neighbors of each L-system look reasonable
    """
    featurizer = ResnetFeaturizer(disable_last_layer=True, softmax=False)
    n = len(zoo)
    n_samples = 3
    images = np.empty((n, n_samples, 128, 128))
    features = np.empty((n, n_samples, featurizer.n_features))
    for i, (sys, angle) in enumerate(zoo):
        for j in range(n_samples):
            img = sys.draw(sys.nth_expansion(3), d=3, theta=angle, n_rows=128, n_cols=128)
            images[i, j] = img
            features[i, j] = featurizer.apply(img)

    # check feature neighbors
    n_neighbors = 5
    features = features.reshape((n, n_samples * featurizer.n_features))
    features_knn = NearestNeighbors(n_neighbors=n_neighbors).fit(features)
    for i in range(n):
        # TODO: assert that distance from a point to itself is 0
        # FIXME: getting nonzero distance from self to self
        distances, indices = features_knn.kneighbors(features[i].reshape(1, -1))
        distances, indices = distances[0], indices[0]
        neighbor_indices = indices[distances.argsort()]
        all_imgs = np.concatenate((images[i].reshape(1, n_samples, 128, 128),
                                   images[neighbor_indices]), axis=0).reshape((-1, 128, 128))
        plot(all_imgs, shape=(n_neighbors + 1, n_samples),
             labels=[f"{x:.4e}" for x in [0.0] * n_samples + np.repeat(distances, n_samples).tolist()])


def plot(imgs: List[np.array], shape: Tuple[int, int], labels: Optional[List[str]] = None):  # pragma: no cover
    assert len(imgs) == shape[0] * shape[1], f"Received {len(imgs)} with shape {shape}"
    assert labels is None or len(imgs) == len(labels), f"Received {len(imgs)} images and {len(labels)} labels"

    fig, ax = plt.subplots(*shape)

    # clear axis ticks
    for axis in ax.flat:
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    # plot bitmaps
    axes: List[plt.Axes] = ax.flat
    for i, img in enumerate(imgs):
        axis = axes[i]
        axis.imshow(img)
        if labels is not None:
            axis.set_title(labels[i], pad=3, fontsize=6)

    plt.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.1)
    plt.show()
    plt.close()


if __name__ == '__main__':  # pragma: no cover
    check_resnet_classifier()
    check_resnet_featurizer()
