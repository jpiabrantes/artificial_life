import os
import pickle
import matplotlib.pylab as plt
import numpy as np
import scipy.stats
import scipy.signal
plt.style.use('bmh')
ep_len = 750


def mean_confidence_interval(arr, axis=0, confidence=0.95):
    n = arr.shape[axis]
    m, se = np.mean(arr, axis=axis), scipy.stats.sem(arr, axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h


with open('VDN.pkl', 'rb') as f:
    alleles_counts = pickle.load(f).astype(np.int)
allele_counts = alleles_counts[:, 1, :]

with open('VDN_nc.pkl', 'rb') as f:
    alleles_counts = pickle.load(f).astype(np.int)
allele_counts_nc = alleles_counts[:, 1, :]

plt.figure()
x = np.arange(ep_len)
mu = allele_counts.mean(axis=0)
l_yerr, u_yerr = mean_confidence_interval(allele_counts)
# l_yerr = np.quantile(allele_counts[0, :, :], 0.05, axis=0)
# u_yerr = np.quantile(allele_counts[0, :, :], 0.95, axis=0)
plt.plot(x, mu, label='No cannibalism')
plt.fill_between(x, l_yerr, u_yerr, alpha=0.5)

mu = allele_counts_nc.mean(axis=0)
l_yerr, u_yerr = mean_confidence_interval(allele_counts_nc)
# l_yerr = np.quantile(allele_counts[1, :, :], 0.05, axis=0)
# u_yerr = np.quantile(allele_counts[1, :, :], 0.95, axis=0)
plt.plot(x, mu, label='Cannibalism')
plt.fill_between(x, l_yerr, u_yerr, alpha=0.5)
plt.legend(loc='best')
