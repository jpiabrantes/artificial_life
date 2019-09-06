import os
import pickle
import matplotlib.pylab as plt
import numpy as np
import scipy.stats
import scipy.signal
import seaborn as sns

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
ep_len = 500


def mean_confidence_interval(arr, axis=0, confidence=0.95):
    n = arr.shape[axis]
    m, se = np.mean(arr, axis=axis), scipy.stats.sem(arr, axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h


with open('greedy/VDN.pkl', 'rb') as f:
    alleles_counts = pickle.load(f).astype(np.int)
allele_counts = alleles_counts[:, 0, :ep_len] #1

with open('greedy/nc_VDN.pkl', 'rb') as f:
    alleles_counts_nc = pickle.load(f).astype(np.int)
allele_counts_nc = alleles_counts_nc[:, 0, :ep_len]

fig, axs = plt.subplots(1, 2)
ax = axs[1]
x = np.arange(ep_len)
mu = allele_counts.mean(axis=0)
l_yerr, u_yerr = mean_confidence_interval(allele_counts)
# l_yerr = np.quantile(allele_counts[0, :, :], 0.05, axis=0)
# u_yerr = np.quantile(allele_counts[0, :, :], 0.95, axis=0)
ax.plot(x, mu, label='Cannibalism')
ax.fill_between(x, l_yerr, u_yerr, alpha=0.5)

mu = allele_counts_nc.mean(axis=0)
l_yerr, u_yerr = mean_confidence_interval(allele_counts_nc)
# l_yerr = np.quantile(allele_counts[1, :, :], 0.05, axis=0)
# u_yerr = np.quantile(allele_counts[1, :, :], 0.95, axis=0)
ax.plot(x, mu, label='No cannibalism')
ax.fill_between(x, l_yerr, u_yerr, alpha=0.5)
ax.legend(loc='upper right')
# plt.xlim([0, 499])
# plt.ylim([0, 18])
ax.set_xlabel('Time steps')
ax.set_ylabel('Family size')
ax.set_title('b)')


DNA_COLORS = {1: (255, 255, 0), 2: (0, 255, 255), 3: (255, 0, 255), 4: (0, 0, 0), 5: (255, 255, 255)}
with open('fight.pkl', 'rb') as f:
    alleles_counts = pickle.load(f).astype(np.int)

fig, axs = plt.subplots(1, 4)
ax = axs[0]
x = np.arange(ep_len)
labels = ['ES 1', 'ES 2 ', 'E-VDN 1', 'E-VDN 2']
for i in range(4):
    mu = alleles_counts.mean(axis=0)[i, :]
    l_yerr, u_yerr = mean_confidence_interval(alleles_counts[:, i, :])
    ax.plot(x, mu, label=labels[i], color=np.array(DNA_COLORS[i+1])/255)
    ax.fill_between(x, l_yerr, u_yerr, alpha=0.5, color=np.array(DNA_COLORS[i+1])/255)

ax.legend(loc='best')
ax.set_xlabel('Steps')
ax.set_ylabel('Family size')
