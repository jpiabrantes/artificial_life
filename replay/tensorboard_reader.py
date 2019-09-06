from collections import defaultdict

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

sns.set_style("darkgrid", {"axes.facecolor": ".9"})


# read ES
path = '~/github/artificial_life/algorithms/evolution/checkpoints/DeadlyColony-v0/ep_stats.csv'
es_df = pd.read_csv(path)
es_df.rename(columns={'Avg_babies_born': 'Babies born', 'Avg_average_population': 'Average population size',
                      'Avg_life_expectancy': 'Life expectancy', 'Avg_n_cannibalism_acts': 'Intra-family attacks'},
             inplace=True)
es_df.set_index('episodes', inplace=True)
# es_df = es_df[es_df.index <= 200000]
# last_time = None
# time = np.zeros(len(es_df))

# for i, t in enumerate(es_df.time):
#     if last_time is None or last_time < t:
#         time[i] = t
#         last_time = t
#     else:
#         time[i] = last_time
# es_df.time = time

# es_df.set_index('time', inplace=True)

# True Conv Original
path = '/home/joao/github/artificial_life/algorithms/dqn/checkpoints_true/checkpoints_conv/DeadlyColony-v0/basic/tensorboard/vdn_1566989301/events.out.tfevents.1566989302.ip-172-31-17-169.2528.1251.v2'
# True FC original
path = '/home/joao/github/artificial_life/algorithms/dqn/checkpoints_true/checkpoints/DeadlyColony-v0/basic/tensorboard/vdn_1562663861/events.out.tfevents.1562663862.ip-172-31-30-186.30414.816.v2'
path = '/home/joao/github/artificial_life/algorithms/dqn/checkpoints_true/checkpoints_o_super_long/DeadlyColony-v0/basic/tensorboard/vdn_1562663861/events.out.tfevents.1562663862.ip-172-31-30-186.30414.816.v2'

# # Greedy FC original
path = '/home/joao/github/artificial_life/algorithms/dqn/checkpoints_greedy/fc_original/DeadlyColony-v0/basic/tensorboard/vdn_1567068307/events.out.tfevents.1567068308.ip-172-31-30-186.2523.816.v2'


# features = ('Avg_average_population', 'Avg_babies_born', 'Avg_life_expectancy', 'Avg_n_attacks',
#             'Avg_n_cannibalism_acts')
features = ('test_Avg_average_population', 'test_Avg_babies_born', 'test_Avg_life_expectancy', 'test_Avg_n_attacks',
            'test_Avg_n_cannibalism_acts', 'test_Avg_attacker_age', 'test_Avg_victim_age', 'test_Avg_cannibal_age',
            'test_Avg_cannibalism_victim_age', 'Episodes')

wall_times = defaultdict(list)
data = defaultdict(dict)
for e in tf.compat.v1.train.summary_iterator(path):
    for v in e.summary.value:
        if v.tag in features:
            wall_times[e.step].append(e.wall_time)
            data[e.step][v.tag] = float(np.frombuffer(v.tensor.tensor_content, dtype=np.float32))

list_of_rows = []
for k in sorted(data.keys()):
    dict_ = data[k]
    dict_['wall_time'] = np.mean(wall_times[k])
    list_of_rows.append(dict_)

df = pd.DataFrame(list_of_rows)
df.drop(df.index[df.test_Avg_average_population.isnull()], inplace=True)
df.wall_time -= df.wall_time.iloc[0]
df.wall_time /= 3600
# df.set_index('wall_time', inplace=True)
df.set_index('Episodes', inplace=True)
df.rename(columns={'test_Avg_babies_born': 'Babies born', 'test_Avg_average_population': 'Average population size',
                   'test_Avg_life_expectancy': 'Life expectancy',
                   'test_Avg_n_cannibalism_acts': 'Intra-family attacks',
                   'test_Avg_n_attacks': 'Inter-families attacks',
                   'test_Avg_attacker_age': 'Attacker age',
                   'test_Avg_victim_age': 'Victim age',
                   'test_Avg_cannibal_age': 'Cannibal age',
                   'test_Avg_cannibalism_victim_age': "Cannibalism's victim age"}, inplace=True)

# df['Attack age delta'] = df["Victim age"]-df['Attacker age']
# df['Cannibalism age delta'] = df["Cannibalism's victim age"]-df['Cannibal age']
features = ['Average population size', 'Babies born', 'Life expectancy', 'Intra-family attacks',
            'Inter-families attacks', "Cannibalism's victim age", 'Cannibal age']

fig, axs = plt.subplots(1, 4, sharex=True)
axs = axs.ravel()
for i, (ax, f, title) in enumerate(zip(axs, features[:-1], ('a)', 'b)', 'c)', 'd)'))):
    # ax.plot(df.index, df[f].values, label='RL')
    ax.plot(es_df.index, es_df[f].values, label='ES')
    # ax.set_title(title)
    ax.set_ylabel(f)
    ax.set_xlabel('episodes')
df = es_df

fig, axs = plt.subplots(1, 4, sharex=True)
for ax, f, title in zip(axs, features[:-1], ('a)', 'b)', 'c)', 'd)')):
    ax.plot(df.index, df[f].values)
    ax.set_title(title)
    ax.set_ylabel(f)
    ax.set_xlabel('episodes')
    for x in [500, 5184, 10157]:
        ax.axvline(x=x, color='k', ls='--', lw=1, alpha=0.5)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
x_pos = [0, 630, 8263, 24496, 62971]  # greedy
x_pos = [0, 45071, 124426, 177721, 350000]
text_pos = [[(100, 7.13), (4452.42, 22.8647), (6996.21, 42.4397), (14203.4, 40.96)],
            [(229, 63.01), (2147.98, 474.511), (7370.75, 743.9), (13288.2, 530.4)],
            [(-150, 37.2), (1827.85, 26.8), (7593.56, 32.124), (14117.9, 46.0315)],
            [(229.909, 90.87), (1823.07, 849.241), (7588.78, 853.818), (13885.5, 235.912)]]
era = ['I', 'II', 'III', 'IV']
fig, axs = plt.subplots(2, 2, sharex='row')
axs = axs.ravel()
for i, (ax, f, title) in enumerate(zip(axs, features[:-3], ('a)', 'b)', 'c)', 'd)'))):
    for j in range(4):
        tdf = df.loc[np.logical_and(x_pos[j] <= df.index, df.index <= x_pos[j + 1])]
        ax.plot(tdf.index/1000, tdf[f].values)
        # x, y = text_pos[i][j]
        # ax.text(x, y, era[j], fontsize=12, color=colors[j], horizontalalignment='left', verticalalignment='top',
        #          fontname='Roman Font 7')
    # ax.set_title(title)
    ax.set_ylabel(f)
    ax.set_xlabel('Training episodes (thousands)')
for ax in axs:
    y_min, y_max = ax.get_ylim()
    ax.fill_between([135.624, 155.661], y_min, y_max, alpha=0.2, color='red')
    ax.fill_between([179.516, 213.530], y_min, y_max, alpha=0.2, color='red')
    ax.set_ylim([y_min, y_max])

fig, ax = plt.subplots()
ax = axs[0]
ax.plot(df.index/1000, df["Cannibalism's victim age"], label="Cannibal's victim")
ax.plot(df.index/1000, df['Cannibal age'], label='Cannibal')
ax.set_ylabel('Age')
ax.set_xlabel('Training episodes (thousands)')
ax.axvline(x=24.496, color=colors[3])
ax.legend(loc='best')

fig, axs = plt.subplots(1, 1, sharex=True, sharey=False)
axs = axs.ravel()
for ax, f, title in zip(axs, features[-4:], ['a)']*4):
    ax.plot(df.index, df[f].values)
    # ax.set_title(title)
    ax.set_ylabel(f)
    ax.set_xlabel('episodes')
    ax.axvline(x=139111)


'''
(df["Cannibalism's victim age"]-df['Cannibal age']).plot()
Phases.

Survival.
Reproduction - internal war
First World War
Heritage
'''

