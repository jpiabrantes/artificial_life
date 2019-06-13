import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


features = ['age', 'health', 'sugar', 'family_size']
# 'attacked', 'kill', 'victim', 'cannibal_attack', 'cannibal_kill', 'cannibal_victim', 'species_index']


def load_df(path):
    df = pd.read_csv(path)
    df[features] = df[features].shift(-1)
    df.dropna(inplace=True)
    return df


def plot_conditional(dfs, df_names, label, value, feature, range):
    for df, df_name in zip(dfs, df_names):
        results = np.zeros(len(range))
        for i, x in enumerate(range):
            results[i] = ((df[label].values == value) & (df[feature].values == x)).sum()/(df[feature].values == x).sum()
        plt.plot(range, results, '.-', label=df_name)
    plt.ylabel('P(%s==%d, %s=x)' % (label, value, feature))
    plt.xlabel(feature)
    plt.legend(loc='best')
    plt.show()


paths = ['replay/data/%s' % name for name in ('CENTRAL_PPO_SURVIVE.csv', 'CENTRAL_PPO_FAMILY.csv',
                                              'CENTRAL_PPO_LEGACY.csv', 'EvolutionStrategies.csv')]
df_names = [r'$\mathrm{PPO}^\mathrm{S}$', r'$\mathrm{PPO}^\mathrm{S+F}$', r'$\mathrm{PPO}^\mathrm{S+F+L}$', 'ES']

dfs = [load_df(path) for path in paths]



plot_conditional(dfs, df_names, 'cannibal_attack', 1, 'age', list(range(1, 51)))
plot_conditional(dfs, df_names, 'cannibal_kill', 1, 'age', list(range(1, 51)))
plot_conditional(dfs, df_names, 'cannibal_victim', 1, 'age', list(range(1, 51)))

plot_conditional(dfs, df_names, 'cannibal_attack', 1, 'sugar', list(range(30)))
plot_conditional(dfs, df_names, 'cannibal_kill', 1, 'sugar', list(range(30)))
plot_conditional(dfs, df_names, 'cannibal_victim', 1, 'sugar', list(range(30)))



plot_conditional(dfs, df_names, 'attacked', 1, 'age', list(range(1, 51)))
plot_conditional(dfs, df_names, 'kill', 1, 'age', list(range(1, 51)))
plot_conditional(dfs, df_names, 'victim', 1, 'age', list(range(1, 51)))

plot_conditional(dfs, df_names, 'attacked', 1, 'sugar', list(range(30)))
plot_conditional(dfs, df_names, 'kill', 1, 'sugar', list(range(30)))
plot_conditional(dfs, df_names, 'victim', 1, 'sugar', list(range(30)))



# label = ['attack']
# df = df.loc[df.species_index == 0, :]
# df = df[features+label]
# df[label[0]] = df[label[0]].values.astype(np.int)
# sns.pairplot(df)
# corr = df.corr()
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))