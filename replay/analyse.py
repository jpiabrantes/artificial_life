import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from collections import defaultdict
from tqdm import tqdm
plt.style.use('bmh')

features = ['age', 'health', 'sugar', 'allele_freq']
e_features = ['e_%s' % f for f in features]

paths = ['replay/data/%s' % name for name in ('0_VDN.csv',)]
df_names = ['VDN']
dfs = [pd.read_csv(path) for path in paths]
df = dfs[0]


# ******************************************
# P(AGE|ATTACK)
# *****************************************
from scipy.special import btdtri
def beta_mean(a, b):
    return a/(a+b)


df['cannibal_attack'] = np.logical_and(df.e_age, df.e_dna == df.dna)
attack_df = df.loc[df.cannibal_attack]
ages = range(1, 49)
a = [0] * len(ages)
b = [0]*len(ages)
mu = np.zeros((len(ages), ))
yerr = np.zeros((2, len(ages)))
for i, age in enumerate(ages):
    joint = (attack_df.age == age).sum()
    a = 1 + joint
    b = 1 + len(attack_df) - joint
    mu[i] = beta_mean(a, b)
    yerr[0, i] = btdtri(a, b, 0.05)
    yerr[1, i] = btdtri(a, b, 0.95)

plt.figure()
plt.errorbar(ages, mu, yerr=yerr)


# *****************************
# P(ATTACK | AGE)
# *****************************
df['cannibal_attack'] = np.logical_and(df.e_age, df.e_dna == df.dna)


def bootstrap_resample(n_samples, ages, df):
    results = np.zeros((len(ages), n_samples))
    mu = np.zeros((len(ages), ))
    yerr = np.zeros((2, len(ages)))
    for i in tqdm(range(n_samples)):
        tdf = df.loc[df.index[np.random.randint(len(df), size=len(df))]]
        for j, age in enumerate(ages):
            prior = tdf.age == age
            joint = np.logical_and(tdf.cannibal_attack, prior)
            results[j, i] = np.sum(joint)/np.sum(prior)
    for j in range(len(ages)):
        mu[j] = results[j, :].mean()
        yerr[0, j] = mu[j]-np.quantile(results[j, :], 0.05)
        yerr[1, j] = np.quantile(results[j, :], 0.95)-mu[j]
    return mu, yerr


ages = range(49)
mu, yerr = bootstrap_resample(10, ages, df[:2000000])

plt.figure()
plt.plot(ages, mu*100, 'o-')
plt.fill_between(ages, (mu+yerr[1, :])*100, (mu-yerr[0, :])*100, alpha=0.5)
plt.ylabel('Probability of attacking (%)')
plt.xlabel("Age")

# *****************************
# P(VICTIM | AGE)
# *****************************
df['cannibal_attack'] = np.logical_and(df.e_age, df.e_dna == df.dna)


def bootstrap_resample(n_samples, ages, df):
    results = np.zeros((len(ages), n_samples))
    mu = np.zeros((len(ages), ))
    yerr = np.zeros((2, len(ages)))
    for i in tqdm(range(n_samples)):
        tdf = df.loc[df.index[np.random.randint(len(df), size=len(df))]]
        for j, age in enumerate(ages):
            prior = tdf.age == age
            joint = np.logical_and(tdf.cannibal_attack, tdf.e_age == age)
            results[j, i] = np.sum(joint)/np.sum(prior)
    for j in range(len(ages)):
        mu[j] = results[j, :].mean()
        yerr[0, j] = mu[j]-np.quantile(results[j, :], 0.05)
        yerr[1, j] = np.quantile(results[j, :], 0.95)-mu[j]
    return mu, yerr


ages = range(49)
mu, yerr = bootstrap_resample(10, ages, df[:2000000])

plt.figure()
plt.plot(ages[1:], mu[1:]*100, 'o-')
plt.fill_between(ages[1:], (mu+yerr[1, 1:])*100, (mu-yerr[0, 1:])*100, alpha=0.5)
plt.ylabel('Probability of attacking (%)')
plt.xlabel("Age")
# plt.xlim([0, 49])










#df = df.loc[df.dna==1]
total = 0
# age_dif = defaultdict(int)
# for iter in tqdm(df.iteration.unique()):
#     rows = df.loc[df.iteration == iter]
#     for i in range(len(rows)):
#         row = rows.iloc[i]
#         for j in range(i, len(rows)):
#             age_dif[abs(row.age-rows.iloc[j].age)] += 1
#             total += 1



df['allele_freq'] = df['family_size']/df['population']
df['e_allele_freq'] = df['e_family_size']/df['population']

attack_df = df.loc[df.cannibal_attack]
attack_df['age_gap'] = attack_df.e_age - attack_df.age
ages = np.unique(attack_df['age_gap'])
result = [0]*len(ages)
for i, age in enumerate(ages):
    result[i] = (attack_df.age_gap == age).mean()#/age_dif[abs(age)]*total

plt.figure()
plt.plot(ages, result)
plt.ylabel("Probability of an attack (%)")
plt.xlabel("Victim's age minus Attacker's age")




df['health_gap'] = df.e_health - df.health
df['sugar_gap'] = df.e_sugar - df.sugar



def plot_conditional(dfs, df_names, label, value, feature, range_, digitize=False):
    if digitize:
        n, bins = np.histogram(dfs[0][feature], bins=10)
        digi_f = 'digi_' + feature
        range_ = range(len(bins) - 1)

    for df, df_name in zip(dfs, df_names):
        if digitize:
            df[digi_f] = np.digitize(df[feature], bins)
        results = np.zeros(len(range_))
        for i, x in enumerate(range_):
            if digitize:
                prior = df[digi_f].values == x
            else:
                prior = df[feature].values == x
            joint = np.logical_and(df[label] == value, df[feature].values == x)
            results[i] = np.sum(joint)/np.sum(prior)
        if digitize:
            fig, ax = plt.subplots()
            bin_centers = (bins[1:]+bins[:-1])/2
            width = (bins[1]-bins[0])*0.95
            ax.bar(bin_centers, results, width=width)
            ax.set_xticks(bin_centers)
            ax.set_xticklabels('%.1f' % v for v in (bin_centers))
            ax.set_ylabel('Probability of defecting (%)')
            ax.set_xlabel('Allele Relative Frequency (%)')
        else:
            plt.plot(range_, results, '.-', label=df_name)
    plt.ylabel('P(%s==%d, %s=x)' % (label, value, feature))
    plt.xlabel(feature)
    plt.legend(loc='best')
    plt.show()


# from sklearn import tree
# import graphviz
# from sklearn.model_selection import cross_val_score
#
# y = df['cannibal_attack'].values.astype(np.int)
# X = df[features+e_features]
# clf = tree.DecisionTreeClassifier()
# clf.fit(X, y)
# scores = cross_val_score(clf, X, y, cv=5)
# dot_data = tree.export_graphviz(clf, out_file='tree1.dot', feature_names=features, class_names='cannibalism victim',
#                                 rounded=True, proportion=False, precision=2, filled=True, max_depth=3)
#
# plt.figure()
# plt.scatter(df.loc[df.cannibal_attack, 'age'], df.loc[df.cannibal_attack, 'e_age'], color='red', alpha=0.1)
#
# plt.figure()
# plt.scatter(df.loc[df.cannibal_attack, 'health'], df.loc[df.cannibal_attack, 'e_health'], color='red', alpha=0.1)
#
# mask = np.logical_not(df.cannibal_attack)
# plt.scatter(df.loc[mask, 'health'], df.loc[mask, 'e_health'], color='green', alpha=0.1)

mask = np.logical_and(df.cannibal_attack, df.age == 1)

values = range(1, 51)
results = [0]*len(values)
for i, v in enumerate(values):
    joint = np.logical_and(np.logical_and(df.cannibal_attack, df.age==1), df.e_age==v)
    prior = np.logical_and(df.cannibal_attack, df.e_age==1)
    results[i] = np.sum(joint)/np.sum(prior)

plt.figure()
plt.plot(results, '.-')

plot_conditional([df.loc[mask]], df_names, 'cannibal_attack', 1, 'age', list(range(1, 51)))
plot_conditional([df], df_names, 'cannibal_attack', 1, 'age', list(range(1, 51)))
plot_conditional(dfs, df_names, 'cannibal_attack', 1, 'age_gap', list(range(-50, 51)))
plot_conditional(dfs, df_names, 'cannibal_attack', 1, 'health_gap', list(range(-2, 3)))
plot_conditional(dfs, df_names, 'cannibal_attack', 1, 'sugar_gap', None, digitize=True)



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