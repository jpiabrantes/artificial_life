import os
import pickle
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

from envs.deadly_colony.deadly_colony import DeadlyColony
from envs.deadly_colony.env_config import env_default_config

env = DeadlyColony(env_default_config)

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
ep_len = 100
n_dicts = 300
path = '/home/joao/github/artificial_life/replay/dicts/'
entropy = np.empty((2, n_dicts, ep_len))
births = np.zeros((2, n_dicts, ep_len))
deaths = np.zeros((2, n_dicts, ep_len))
population = np.zeros((2, n_dicts, ep_len))

for j in range(2):
    for dict_idx in range(n_dicts):
        print(dict_idx)
        string = '%d_VDN.pkl' if j else '%d_VDN_gd.pkl'
        with open(os.path.join(path, string % dict_idx), 'rb') as f:
            dicts_ = pickle.load(f)

        assert len(dicts_) == ep_len
        last_agent_set = None
        for i, dict_ in enumerate(dicts_):
            agent_set = set(dict_['agents'].keys())
            state = dict_['state']
            dna = state[:, :, env.State.DNA]
            dna = dna[dna != 0]
            unique_alleles, counts = np.unique(dna, return_counts=True, axis=0)
            probs = counts / np.sum(counts)
            entropy[j, dict_idx, i] = -np.sum(probs * np.log(np.where(probs == 0, 1, probs)))
            if i:
                # who is here and wasnt
                births[j, dict_idx, i] = births[j, dict_idx, i-1] + len(agent_set.difference(last_agent_set))

                # who is not here and was
                deaths[j, dict_idx, i] = deaths[j, dict_idx, i-1] + len(last_agent_set.difference(agent_set))
            population[j, dict_idx, i] = len(agent_set)
            last_agent_set = agent_set


import scipy.stats
import scipy.signal


def mean_confidence_interval(arr, axis=0, confidence=0.95):
    n = arr.shape[axis]
    m, se = np.mean(arr, axis=axis), scipy.stats.sem(arr, axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h


window_size = 100
death_rate = np.zeros_like(deaths)
birth_rate = np.zeros_like(births)
death_rate[:, :, 1:] = deaths[:, :, 1:] - deaths[:, :, :-1]
birth_rate[:, :, 1:] = births[:, :, 1:] - births[:, :, :-1]
n_death_rate = np.zeros((2, n_dicts, ep_len-window_size+1))
n_birth_rate = np.zeros((2, n_dicts, ep_len-window_size+1))
for i in range(n_dicts):
    for j in range(2):
        n_death_rate[j, i, :] = scipy.signal.convolve(death_rate[j, i, :], np.ones((window_size,)) * 1 / window_size, 'valid')
        n_birth_rate[j, i, :] = scipy.signal.convolve(birth_rate[j, i, :], np.ones((window_size,)) * 1 / window_size, 'valid')


with open('greedy/VDN.pkl', 'rb') as f:
    alleles_counts = pickle.load(f)
probs = alleles_counts/np.sum(alleles_counts, axis=1)[:, None, :]
entropy = probs*np.log(probs)
entropy[np.isnan(entropy)] = 0
entropy_ks = -entropy.sum(axis=1)

with open('greedy/nk_VDN.pkl', 'rb') as f:
    alleles_counts = pickle.load(f)
probs = alleles_counts/np.sum(alleles_counts, axis=1)[:, None, :]
entropy = probs*np.log(probs)
entropy[np.isnan(entropy)] = 0
entropy_gd = -entropy.sum(axis=1)

plt.figure()
x = np.arange(500)
y1, y2 = mean_confidence_interval(entropy_ks)
y2 = np.where(y1 == y2, y2+1e-3, y2)
plt.fill_between(x, y1=y1, y2=y2, label='Kinship detection', alpha=0.5)
plt.plot(x, np.mean(entropy_ks, axis=0))

y1, y2 = mean_confidence_interval(entropy_gd)
y2 = np.where(y1 == y2, y2+1e-3, y2)
plt.plot(x, np.mean(entropy_gd, axis=0))
plt.fill_between(x, y1=y1, y2=y2, label='No kinship detection', alpha=0.5)
plt.xlabel('Time steps')
plt.ylabel('Entropy')
plt.legend(loc='best')
plt.show()


fig, axs = plt.subplots(1, 3, sharex=False, sharey=False)
ax = axs[0]
x = np.arange(ep_len)
y1, y2 = mean_confidence_interval(deaths[1, :, :])
y2 = np.where(y1 == y2, y2+1e-3, y2)
ax.plot(x, deaths[1, :, :].mean(axis=0))
ax.fill_between(x, y1=y1, y2=y2, label='Kinship detection', alpha=0.5)
y1, y2 = mean_confidence_interval(deaths[0, :, :])
y2 = np.where(y1 == y2, y2+1e-3, y2)
ax.plot(x, deaths[0, :, :].mean(axis=0))
ax.fill_between(x, y1=y1, y2=y2, label='No kinship detection', alpha=0.5)
# ax.legend(loc='lower left')
ax.set_xlabel('iteration')
ax.set_ylabel('Cumulative deaths')
ax.set_xlim([0, 100])
# ax.set_ylim([0.8, 1.06])

ax = axs[1]
y1, y2 = mean_confidence_interval(births[1, :, :])
y2 = np.where(y1 == y2, y2+1e-3, y2)
ax.plot(x, births[1, :, :].mean(axis=0))
ax.fill_between(x, y1=y1, y2=y2, label='Kinship detection', alpha=0.5)
y1, y2 = mean_confidence_interval(births[0, :, :])
y2 = np.where(y1 == y2, y2+1e-3, y2)
ax.plot(x, births[0, :, :].mean(axis=0))
ax.fill_between(x, y1=y1, y2=y2, label='No kinship detection', alpha=0.5)
# ax.legend(loc='lower left')
ax.set_xlabel('iteration')
ax.set_ylabel('Cumulative births')
ax.set_xlim([0, 100])
# ax.set_ylim([0.8, 1.06])

ax = axs[2]
x = np.arange(ep_len)
y1, y2 = mean_confidence_interval(population[1, :, :])
y2 = np.where(y1 == y2, y2+1e-3, y2)
ax.plot(x, population[1, :, :].mean(axis=0))
ax.fill_between(x, y1=y1, y2=y2, label='Kinship detection', alpha=0.5)
y1, y2 = mean_confidence_interval(population[0, :, :])
y2 = np.where(y1 == y2, y2+1e-3, y2)
ax.plot(x, population[0, :, :].mean(axis=0))
ax.fill_between(x, y1=y1, y2=y2, label='No kinship detection', alpha=0.5)
ax.legend(loc='upper left')
ax.set_xlabel('iteration')
ax.set_ylabel('Total population')
ax.set_xlim([0, 100])
# ax = axs[1]
# y1, y2 = mean_confidence_interval(births[1, :, :])
# y2 = np.where(y1 == y2, y2+1e-3, y2)
# ax.fill_between(x, y1=y1, y2=y2, label='Kinship detection', alpha=0.5)
# y1, y2 = mean_confidence_interval(births[0, :, :])
# y2 = np.where(y1 == y2, y2+1e-3, y2)
# ax.fill_between(x, y1=y1, y2=y2, label='No kinship detection', alpha=0.5)
# ax.legend(loc='upper left')

plt.show()
