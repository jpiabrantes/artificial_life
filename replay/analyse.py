import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


features = ['age', 'health', 'sugar', 'family_size']
# 'attack', 'kill', 'victim', 'cannibal_attack', 'cannibal_kill', 'cannibal_victim', 'species_index']
path = 'replay/data/EvolutionStrategies.csv'
df = pd.read_csv(path)
df.dropna(inplace=True)
df.drop(df.loc[df.age == 0].index, inplace=True)
# label = ['attack']
# df = df.loc[df.species_index == 0, :]
# df = df[features+label]
# df[label[0]] = df[label[0]].values.astype(np.int)
# sns.pairplot(df)
# corr = df.corr()
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))


def plot_conditional(label, value, feature, range):
    results = np.zeros(len(range))
    for i, x in enumerate(range):
        results[i] = ((df[label].values == value) & (df[feature].values == x)).sum()/(df[feature].values == x).sum()
    plt.plot(range, results, '.-')
    plt.ylabel('P(%s==%d, %s=x)' % (label, value, feature))
    plt.xlabel(feature)
    plt.show()

plot_conditional('cannibal_kill', 1, 'age', list(range(1, 51)))

plot_conditional('cannibal_victim', 1, 'sugar', list(range(0, 20)))
