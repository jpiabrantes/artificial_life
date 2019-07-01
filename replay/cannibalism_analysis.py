import os
import pickle
import matplotlib.pylab as plt
import numpy as np
import scipy.stats
import scipy.signal


def mean_confidence_interval(arr, axis=0, confidence=0.95):
    n = arr.shape[axis]
    m, se = np.mean(arr, axis=axis), scipy.stats.sem(arr, axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h


plt.style.use('bmh')

ep_len = 750
n_dicts = 33
path = '/home/joao/github/artificial_life/replay/dicts/'
allele_counts = np.empty((2, n_dicts, ep_len))

for j in range(2):
    for dict_idx in range(n_dicts):
        print(dict_idx)
        string = '%d_VDN.pkl' if j else '%d_VDN_no.pkl'
        with open(os.path.join(path, string % dict_idx), 'rb') as f:
            dicts_ = pickle.load(f)

        for i, dict_ in enumerate(dicts_):
            allele_counts[j, dict_idx, i] = sum([agent_dict['dna'] == 1 for agent_dict in dict_['agents'].values()])

plt.figure()
x = np.arange(ep_len)
mu = allele_counts[0, :, :].mean(axis=0)
l_yerr, u_yerr = mean_confidence_interval(allele_counts[0, :, :])
# l_yerr = np.quantile(allele_counts[0, :, :], 0.05, axis=0)
# u_yerr = np.quantile(allele_counts[0, :, :], 0.95, axis=0)
plt.plot(x, mu, label='No cannibalism')
plt.fill_between(x, l_yerr, u_yerr, alpha=0.5)

mu = allele_counts[1, :, :].mean(axis=0)
l_yerr, u_yerr = mean_confidence_interval(allele_counts[1, :, :])
# l_yerr = np.quantile(allele_counts[1, :, :], 0.05, axis=0)
# u_yerr = np.quantile(allele_counts[1, :, :], 0.95, axis=0)
plt.plot(x, mu, label='Cannibalism')
plt.fill_between(x, l_yerr, u_yerr, alpha=0.5)
plt.legend(loc='best')
