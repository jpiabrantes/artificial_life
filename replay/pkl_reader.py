import os
import pickle
import numpy as np

from envs.deadly_colony.deadly_colony import DeadlyColony
from envs.deadly_colony.env_config import env_default_config


env = DeadlyColony(env_default_config)

n_dicts = 90
ep_len = 500
path = '/home/joao/github/artificial_life/replay/dicts/'
allele_counts = np.zeros((n_dicts, 5, ep_len))
for dict_idx in range(n_dicts):
    print(dict_idx)
    string = 'fight_%d.pkl'
    with open(os.path.join(path, string % dict_idx), 'rb') as f:
        dicts_ = pickle.load(f)

    for i, dict_ in enumerate(dicts_):
        state = dict_['state']
        dna = state[:, :, env.State.DNA]
        dna = dna[dna != 0]
        unique_alleles, counts = np.unique(dna, return_counts=True, axis=0)
        for gene, count in zip(unique_alleles, counts):
            allele_counts[dict_idx, int(gene)-1, i] = count

filepath = 'fight.pkl'
if os.path.isfile(filepath):
    with open(filepath, 'rb') as f:
        allele_counts = np.concatenate((pickle.load(f), allele_counts), axis=0)

with open(filepath, 'wb') as f:
    pickle.dump(allele_counts, f)
