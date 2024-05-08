from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# data = np.array([-7, -5, 1, 4, 5], dtype=np.float64)
# kde1 = stats.gaussian_kde(data)
# kde2 = stats.gaussian_kde(data, bw_method='silverman')

# fig, ax = plt.subplots()

# ax.plot(data, np.zeros(data.shape), 'b+', ms=20)

# x_eval = np.linspace(-10, 10, num=200)

# ax.plot(x_eval, kde1(x_eval), 'k-', label="Scott")
# ax.plot(x_eval, kde2(x_eval), 'r-', label="Silverman")

# kde3 = stats.gaussian_kde(data, bw_method=.5)
# ax.plot(x_eval, kde3(x_eval), 'g-', label="10")

# def bandwidth(obj):
#     n = obj.n
#     d = obj.d
#     return np.power(n, -1./(d + 4))

# kde4 = stats.gaussian_kde(data, bw_method=bandwidth)
# ax.plot(x_eval, kde4(x_eval), 'y.', label="mine")

# kde5 = stats.gaussian_kde(data, bw_method=0.1)
# ax.plot(x_eval, kde5(x_eval), 'g*', label="narrow")

# plt.legend()
# plt.show()


def measure(k, n):
    x, y = [], []
    for i in range(k):
        x_mean = np.random.uniform(100)
        x.extend(np.random.normal(x_mean, size=n))

        y_mean = np.random.uniform(100)
        y.extend(np.random.normal(y_mean, size=n))

    return np.array(x), np.array(y)

# m1, m2 = measure(3, 100)
m1 = np.array([3, 7])
m2 = np.array([2, 3])

values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values, bw_method=1)
samples = kernel.resample(100)

margin=10
xmin = min(m1.min(), samples[0, :].min()) - margin
xmax = max(m1.max(), samples[0, :].max()) + margin
ymin = min(m2.min(), samples[1, :].min()) - margin
ymax = max(m2.max(), samples[1, :].max()) + margin

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kernel.evaluate(positions).T, X.shape)
fig, ax = plt.subplots()


ax.imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax])

ax.plot(m1, m2, 'k.', markersize=2, label="data")
ax.plot(*samples, 'rx', markersize=5, label="samples")
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])

plt.legend()
plt.show()
