import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns

plt.style.use('bmh')


def load_and_pre_process(path):
    df = pd.read_csv(path)
    df['DD'] = np.logical_and(df.compete_1 == True, df.compete_2 == True).astype(np.int)
    df['CC'] = np.logical_and(df.compete_1 == False, df.compete_2 == False).astype(np.int)
    df['CD'] = np.logical_and(df.compete_1 == False, df.compete_2 == True).astype(np.int)
    df['DC'] = np.logical_and(df.compete_1 == True, df.compete_2 == False).astype(np.int)
    df['same_family'] = (df.dna_1 == df.dna_2)
    return df


df_names = ['competitive']
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
        yerr[0, i_f] = mu[i_f]-np.quantile(results[i_f, :], 0.05)
        yerr[1, i_f] = np.quantile(results[i_f, :], 0.95)-mu[i_f]
    return mu, yerr

x = np.ones((4, ))*1/4

df = dfs[0]
mask = df.age_2 > df.age_1
tdf = df.copy()
tdf.loc[np.logical_and(mask, df.CD), 'CD'] = False
tdf.loc[np.logical_and(mask, df.CD), 'DC'] = True
tdf.loc[np.logical_and(mask, df.DC), 'DC'] = False
tdf.loc[np.logical_and(mask, df.DC), 'CD'] = True

tdf = tdf.loc[np.logical_or(np.logical_and(np.logical_and(mask, df.age_2 > 39), df.age_1<6),
                            np.logical_and(np.logical_and(np.logical_not(mask), df.age_1 > 39), df.age_2<6))]
# tdf = tdf.loc[np.logical_or(np.logical_and(mask, df.age_2 > 39),
#                             np.logical_and(np.logical_not(mask), df.age_1 > 39))]
start = tdf.copy()


fig, ax = plt.subplots()
width = 0.3
x = np.arange(4)-width*0.5
tdf = start.loc[start.same_family == True]
mu, yerr = bootstrap_resample(10000, tdf)
plt.bar(x, mu, yerr=yerr, width=width, label='intra-family')
tdf = start.loc[start.same_family == False]
mu, yerr = bootstrap_resample(10000, tdf)
plt.bar(x + width, mu, yerr=yerr, width=width, label='inter-family')
ax.set_xticks(np.arange(4))
ax.set_xticklabels(('CD', 'CC', 'DC', 'DD'))
plt.ylabel('Probability (%)')
plt.legend(loc='best')


# bar plot

masks = [tdf.age_1 > tdf.age_2, tdf.age_1 < tdf.age_2, tdf.age_1 >= 40, tdf.age_2 >= 40]
data = []  # masks x features
for mask in masks:
    data.append([np.sum(tdf.loc[mask, f] == 1) for f in ('CD', 'CC', 'DC', 'DD')])

fig, ax = plt.subplots()
plots = []
ind = np.arange(4)
width = 0.1
data = np.array(data)

for i, feature in enumerate(data.T):
    plots.append(ax.bar(ind + width * i, feature, width)[0])
ax.set_xticks(ind + width*3/2)
ax.set_xticklabels(('Age 1 > Age 2', 'Age 1 < Age 2', 'Age 1 > 40', 'Age 2 > 40'))

ax.legend(plots, ('CD', 'CC', 'DC', 'DD'))


fig, axs = plt.subplots(1, 1, sharey=True, sharex=True)
for ax, df, df_name in zip([axs], dfs, df_names):
    tdf = df.loc[df['CD'] == 1]
    ax.scatter(tdf.sugar_1, tdf.sugar_2, color='green', alpha=0.5, label='CD')
    tdf = df.loc[df['DC'] == 1]
    ax.scatter(tdf.sugar_1, tdf.sugar_2, color='orange', alpha=0.5, label='DC')
    tdf = df.loc[df['DD'] == 1]
    ax.scatter(tdf.sugar_1, tdf.sugar_2, color='red', alpha=0.5, label='DD')
    tdf = df.loc[df['CC'] == 1]
    ax.scatter(tdf.sugar_1, tdf.sugar_2, color='blue', alpha=0.5, label='CC')
    y = np.linspace(0, max(df.sugar_1.max(), df.sugar_2.max()), 100)
    ax.plot(y, y, color='k')
    ax.set_xlabel('Sugar of the first agent')
    ax.legend(loc='best')


fig, axs = plt.subplots(1, 1, sharey=True, sharex=True)
for ax, df, df_name in zip([axs], dfs, df_names):
    df.health_1 += (np.random.rand(len(df))-0.5)*0.5
    df.health_2 += (np.random.rand(len(df))-0.5)*0.5
    tdf = df.loc[df['CD'] == 1]
    ax.scatter(tdf.health_1, tdf.health_2, color='green', alpha=0.5, label='CD')
    tdf = df.loc[df['DC'] == 1]
    ax.scatter(tdf.health_1, tdf.health_2, color='orange', alpha=0.5, label='DC')
    tdf = df.loc[df['DD'] == 1]
    ax.scatter(tdf.health_1, tdf.health_2, color='red', alpha=0.5, label='DD')
    tdf = df.loc[df['CC'] == 1]
    ax.scatter(tdf.health_1, tdf.health_2, color='blue', alpha=0.5, label='CC')
    y = np.linspace(0, max(df.health_2.max(), df.health_1.max()), 100)
    ax.plot(y, y, color='k')
    ax.set_xlabel('Health of the first agent')
    ax.legend(loc='best')




# df['same_family'] = (df.dna_1 == df.dna_2)
# same_df = df.loc[df.same_family == True].copy()
# same_df['delta age'] = same_df.age_1 - same_df.age_2
# same_df['delta sugar'] = same_df.sugar_1 - same_df.sugar_2
# same_df['delta health'] = same_df.health_1 - same_df.health_2
# same_df['old_1'] = (same_df.age_1 >= 45).astype(np.int)
# same_df['old_2'] = (same_df.age_2 >= 45).astype(np.int)
#
# features = ['delta age', 'delta sugar', 'delta health', 'old_1', 'old_2']
#
# survive_mask = same_df.species_index_1 < 3
# legacy_mask = same_df.species_index_1 > 5
# family_mask = np.logical_not(np.logical_or(survive_mask, legacy_mask))
#
# label = ['CD']
# tdf = same_df.loc[survive_mask, features+label]
# # sns.pairplot(tdf)
# # corr = tdf.corr()
# # sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))
#
#
# def plot_conditional(dfs, df_names, label, value, feature, range):
#     for df, df_name in zip(dfs, df_names):
#         results = np.zeros(len(range))
#         for i, x in enumerate(range):
#             results[i] = ((df[label].values == value) & (df[feature].values == x)).sum()/(df[feature].values == x).sum()
#         plt.plot(range, results, label=df_name)
#     plt.ylabel('P(%s==%d, %s=x)' % (label, value, feature))
#     plt.xlabel(feature)
#     plt.legend(loc='best')
#     plt.show()
#
#
# df_names = ['VDN']
# dfs = [same_df]
# plot_conditional(dfs, df_names, 'compete_1', 1, 'delta age', np.arange(-50, 50, 1))