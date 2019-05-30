import matplotlib.pylab as plt
import numpy as np

n_species = 5
n_iters = 500
family_sizes = np.ones((n_iters, n_species))*np.nan

family_sizes[0, :] = 1
for i in range(1, n_iters-20):
    family_sizes[i, :] = np.maximum(family_sizes[i-1, :] + (np.random.randint(0, 2, size=5)*2-1), 1)

fig, ax = plt.subplots(1, 1)
line = ax.stackplot(np.arange(n_iters), family_sizes.T)

ax.set_xlim([0, 500])


#line = plt.fill_between(np.arange(5), np.ones(5), np.ones(5)*2)

