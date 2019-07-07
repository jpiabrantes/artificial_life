import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns

plt.style.use('bmh')

columns = ['species_index', 'dna', 'health', 'sugar', 'age', 'family_size']


def load_and_pre_process(path):
    df = pd.read_csv(path)
    mask = df.age_2 > df.age_1
    for c in columns:
        feature_1, feature_2 = '%s_1' % c, '%s_2' % c
        temp = df.loc[mask, feature_1].copy()
        df.loc[mask, feature_1] = df.loc[mask, feature_2].copy()
        df.loc[mask, feature_2] = temp
    df['DD'] = np.logical_and(df.compete_1 == True, df.compete_2 == True).astype(np.int)
    df['CC'] = np.logical_and(df.compete_1 == False, df.compete_2 == False).astype(np.int)
    df['CD'] = np.logical_and(df.compete_1 == False, df.compete_2 == True).astype(np.int)
    df['DC'] = np.logical_and(df.compete_1 == True, df.compete_2 == False).astype(np.int)
    df['same_family'] = (df.dna_1 == df.dna_2)
    return df

df_names = ['vdncompetitive']
paths = ['replay/data/%s.csv' % exp_name for exp_name in df_names]
dfs = [load_and_pre_process(path) for path in paths]

# ********************************************************************************** #
# ALLELE FRENQUENCY VS FAMILY SIZE
# ********************************************************************************** #

from scipy.special import btdtri
def beta_mean(a, b):
    return a/(a+b)

df = dfs[0]
df['allele_freq_1'] = df['family_size_1']/df['total_population']
df['allele_freq_2'] = df['family_size_2']/df['total_population']
bins = np.linspace(0, 1.001, 10)
n_bins = len(bins)-1
df['bin_allele_freq_1'] = np.digitize(df['allele_freq_1'], bins)
df['bin_allele_freq_2'] = np.digitize(df['allele_freq_2'], bins)
tdf = df.loc[df.same_family == False]
# tdf = df.loc[df.same_family == True]


cond_prob = np.empty((n_bins, ))
a = np.ones((n_bins, ))
b = np.ones((n_bins, ))
mu = np.zeros((n_bins, ))
yerr = np.zeros((2, n_bins))
for i, bin in enumerate(range(1, n_bins+1)):
    prior_1 = tdf['bin_allele_freq_1'] == bin
    prior_2 = tdf['bin_allele_freq_2'] == bin
    prior = np.logical_or(prior_1, prior_2)
    joint_1 = np.logical_and(prior_1, tdf.compete_1 == True)
    joint_2 = np.logical_and(prior_2 == bin, tdf.compete_2 == True)
    joint = np.logical_or(joint_1, joint_2)
    cond_prob[i] = np.sum(joint)/np.sum(prior)
    a[i] = 1 + np.sum(joint)
    b[i] = 1 + np.sum(prior)-np.sum(joint)
    yerr[0, i] = btdtri(a[i], b[i], 0.05)
    yerr[1, i] = btdtri(a[i], b[i], 0.95)
    mu[i] = beta_mean(a[i], b[i])


width = bins[1]-bins[0]
bin_centers = (bins[:-1]+bins[1:])/2

fig, ax = plt.subplots()
ax.bar(bin_centers*100, mu*100, width=width*100*0.95, yerr=yerr)
# ax.bar(bin_centers*100, cond_prob*100, width=width*100*0.95)
ax.set_xticks(bin_centers*100)
ax.set_xticklabels('%.1f' % v for v in (bin_centers*100))
ax.set_xlim(0, 100)
ax.set_ylabel('Probability of defecting (%)')
ax.set_xlabel('Allele Relative Frequency (%)')


# ********************************************************************************** #
# DEFECTING VS AGE
# ********************************************************************************** #
def bootstrap_resample(n_samples, df):
    results = np.zeros((4, n_samples))
    mu = np.zeros((4, ))
    yerr = np.zeros((2, 4))
    for i in tqdm(range(n_samples)):
        tdf = df.loc[df.index[np.random.randint(len(df), size=len(df))]]
        for i_f, f in enumerate(('CD', 'CC', 'DC', 'DD')):
            results[i_f, i] = tdf[f].sum()/len(tdf)*100
    for i_f in range(4):
        mu[i_f] = results[i_f, :].mean()
        sem = np.std(results[i_f, :])/np.sqrt(results.shape[1])
        yerr[0, i_f] = 1.96*sem  # mu[i_f]-np.quantile(results[i_f, :], 0.05)
        yerr[1, i_f] = 1.96*sem
    return mu, yerr

x = np.ones((4, ))*1/4

df = dfs[0]
tdf = df.copy()

# tdf = tdf.loc[np.logical_and(df.age_1 > 45, df.age_2 < 5)]
start = tdf.copy()


fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
x = np.arange(4)

ax = axs[0]
tdf = start.loc[start.same_family == True]
mu, yerr = bootstrap_resample(2, tdf)
ax.set_title('Intra-family interaction')
ax.bar(x, mu, yerr=yerr)
ax.set_ylabel('Probability (%)')
ax.set_xticks(np.arange(4))
ax.set_xticklabels(('CD', 'CC', 'DC', 'DD'))

ax = axs[1]
tdf = start.loc[start.same_family == False]
mu, yerr = bootstrap_resample(2, tdf)
plt.bar(x, mu, yerr=yerr)
ax.set_title('Inter-family interaction')
ax.set_xticks(np.arange(4))
ax.set_xticklabels(('CD', 'CC', 'DC', 'DD'))

# ********************************************************************************** #
# DEFECTING VS AGE_DIFFERENCE
# ********************************************************************************** #
df = dfs[0]
df['age_delta'] = -df.age_2 + df.age_1
age_dif = sorted(df.age_delta.unique())


results = np.zeros((2, 4, len(age_dif)))

for j in range(2):
    tdf = df.copy()
    tdf = tdf.loc[tdf.same_family == bool(j)]
    for i in age_dif:
        mu, yerr = bootstrap_resample(2, tdf.loc[tdf.age_delta == i])
        results[j, :, i] = mu

plt.figure()
plt.stackplot(age_dif, results[0, :, :])
plt.figure()
plt.stackplot(age_dif, results[1, :, :])


