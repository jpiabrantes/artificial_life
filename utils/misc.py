import time

import ray
import numpy as np
import scipy.signal


def get_explained_variance(y_true, y_pred):
    return 1 - np.cov(np.array(y_true) - np.array(y_pred)) / np.cov(y_true)


class Enum(tuple):
    __getattr__ = tuple.index


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


class MeanTracker:
    def __init__(self):
        self._counter = 0
        self._total = 0

    @property
    def mean(self):
        if self._counter:
            return self._total / self._counter
        else:
            return 0

    def add_value(self, value):
        if not np.isnan(value):
            self._total += value
            self._counter += 1


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class SpeciesSamplerManager(object):

    @staticmethod
    def synchronize(local_species_sampler, workers):

        remote_ss = ray.get([r.get_species_sampler.remote() for r in workers])
        for rss in remote_ss:
            local_species_sampler.update(rss)

        remote_copy = ray.put(local_species_sampler)
        [r.sync_species_sampler.remote(remote_copy) for r in workers]


class SpeciesSampler:
    def __init__(self, population_size):
        self.population_size = population_size
        self._m = np.zeros(population_size)
        self._s = np.zeros(population_size)
        self._n = np.zeros(population_size)
        self._last_sample = None

    def show_results(self, species_indices, values):
        unique_values_1, counts_1 = np.unique(species_indices, return_counts=True)
        unique_values_2, counts_2 = np.unique(self._last_sample, return_counts=True)
        assert np.all(unique_values_1 == unique_values_2) and np.all(counts_1 == counts_2), "Showing results for " \
                                                                                            "policies that weren't " \
                                                                                            "sampled"
        for i, value in zip(species_indices, values):
            last_n = self._n[i]
            self._n[i] += 1
            if self._n[i] == 1:
                self._m[i] = value
            else:
                delta = value - self._m[i]
                self._m[i] += delta / self._n[i]
                self._s[i] += delta ** 2 * last_n / self._n[i]
        self._last_sample = None

    def sample(self, size):
        assert self._last_sample is None, 'Show results before sampling again'
        if np.any(self._n == 0):  # if there is a species that hasn't been sampled yet, assume uniform distribution.
            prob = np.ones(self.population_size)/self.population_size
        else:
            population = np.maximum([np.random.normal(mu, std) for mu, std in zip(self.mean, self.std)],
                                    np.zeros(self.population_size))
            prob = population / np.sum(population)
        samples = np.random.choice(range(self.population_size), p=prob, replace=True, size=size)
        self._last_sample = samples
        return samples

    @property
    def mean(self):
        return self._m

    @property
    def var(self):
        return [s / (n - 1) if n > 1 else m ** 2 for s, n, m in zip(self._s, self._n, self._m)]

    @property
    def std(self):
        return np.sqrt(self.var)

    def update(self, other):
        n1 = self._n
        n2 = other._n
        n = n1 + n2
        # Avoid divide by zero, which creates nans
        mask = n != 0
        n, n1, n2 = n[mask], n1[mask], n2[mask]
        delta = self._m[mask] - other._m[mask]
        delta2 = delta * delta
        m = (n1 * self._m[mask] + n2 * other._m[mask]) / n
        s = self._s[mask] + other._s[mask] + delta2 * n1 * n2 / n
        self._n[mask] = n
        self._m[mask] = m
        self._s[mask] = s

    def sync(self, other):
        self._n, self._m, self._s = other._n, other._m, other._s