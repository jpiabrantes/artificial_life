import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import seaborn as sns


path = 'replay/data/competitive.csv'
df = pd.read_csv(path)

df['DD'] = np.logical_and(df.compete_1 == True, df.compete_2 == True).astype(np.int)
df['CC'] = np.logical_and(df.compete_1 == False, df.compete_2 == False).astype(np.int)
df['CD'] = np.logical_and(df.compete_1 == False, df.compete_2 == True).astype(np.int)#np.logical_not(np.logical_or(df['CC'], df['DD'])).astype(np.int)
df['DC'] = np.logical_and(df.compete_1 == True, df.compete_2 == False).astype(np.int)

df['same_family'] = (df.dna_1 == df.dna_2)
same_df = df.loc[df.same_family == True].copy()
same_df['delta age'] = same_df.age_1 - same_df.age_2
same_df['delta sugar'] = same_df.sugar_1 - same_df.sugar_2
same_df['delta health'] = same_df.health_1 - same_df.health_2
same_df['old_1'] = (same_df.age_1 >= 45).astype(np.int)
same_df['old_2'] = (same_df.age_2 >= 45).astype(np.int)

features = ['delta age', 'delta sugar', 'delta health', 'old_1', 'old_2']

survive_mask = same_df.species_index_1 < 3
legacy_mask = same_df.species_index_1 > 5
family_mask = np.logical_not(np.logical_or(survive_mask, legacy_mask))

label = ['CD']
tdf = same_df.loc[survive_mask, features+label]
# sns.pairplot(tdf)
# corr = tdf.corr()
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))


def plot_conditional(dfs, df_names, label, value, feature, range):
    for df, df_name in zip(dfs, df_names):
        results = np.zeros(len(range))
        for i, x in enumerate(range):
            results[i] = ((df[label].values == value) & (df[feature].values == x)).sum()/(df[feature].values == x).sum()
        plt.plot(range, results, '.', label=df_name)
    plt.ylabel('P(%s==%d, %s=x)' % (label, value, feature))
    plt.xlabel(feature)
    plt.legend(loc='best')
    plt.show()


df_names = ['Survive', 'Family', 'Legacy']
dfs = [same_df.loc[mask, :] for mask in (survive_mask, family_mask, legacy_mask)]
plot_conditional(dfs, df_names, 'compete_1', 1, 'delta age', np.arange(-50, 50, 1))

fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
for ax, df, df_name in zip(axs, dfs, df_names):
    tdf = df.loc[df['CD'] == 1]
    ax.scatter(tdf.age_1, tdf.age_2, color='green', alpha=0.5, label='CD')
    tdf = df.loc[df['DC'] == 1]
    ax.scatter(tdf.age_1, tdf.age_2, color='yellow', alpha=0.5, label='DC')
    tdf = df.loc[df['DD'] == 1]
    ax.scatter(tdf.age_1, tdf.age_2, color='red', alpha=0.5, label='DD')
    tdf = df.loc[df['CC'] == 1]
    ax.scatter(tdf.age_1, tdf.age_2, color='blue', alpha=0.5, label='CC')
    ax.plot(range(51), range(51), color='k')
    ax.axvline(x=45, color='black')
    ax.axhline(y=45, color='black')
    ax.set_xlabel('Age of the first agent')
    ax.legend(loc='best')


fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
for ax, df, df_name in zip(axs, dfs, df_names):
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


fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
for ax, df, df_name in zip(axs, dfs, df_names):
    df.health_1 += np.random.rand(len(df))
    df.health_2 += np.random.rand(len(df))
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




