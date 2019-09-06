from collections import defaultdict

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
# plt.style.use('seaborn')


# read ES
path = '~/github/artificial_life/algorithms/evolution/checkpoints/DeadlyColony-v0/ep_stats.csv'
es_df = pd.read_csv(path)
es_df.rename(columns={'Avg_babies_born': 'Babies born', 'Avg_average_population': 'Average population size',
                      'Avg_life_expectancy': 'Life expectancy', 'Avg_n_cannibalism_acts': 'Intra-family attacks'},
             inplace=True)
es_df.time /= 3600
es_df.set_index('time', inplace=True)

path = '/home/joao/github/artificial_life/algorithms/dqn/checkpoints_true/checkpoints_conv/DeadlyColony-v0/basic/tensorboard/vdn_1566989301/events.out.tfevents.1566989302.ip-172-31-17-169.2528.1251.v2'

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
df.set_index('wall_time', inplace=True)
# df.set_index('Episodes', inplace=True)
df.rename(columns={'test_Avg_babies_born': 'Babies born', 'test_Avg_average_population': 'Average population size',
                   'test_Avg_life_expectancy': 'Life expectancy',
                   'test_Avg_n_cannibalism_acts': 'Intra-family attacks',
                   'test_Avg_n_attacks': 'Inter-families attacks',
                   'test_Avg_attacker_age': 'Attacker age',
                   'test_Avg_victim_age': 'Victim age',
                   'test_Avg_cannibal_age': 'Cannibal age',
                   'test_Avg_cannibalism_victim_age': "Cannibalism's victim age"}, inplace=True)

features = ['Average population size', 'Babies born', 'Life expectancy', 'Intra-family attacks',
            'Inter-families attacks', "Cannibalism's victim age", 'Cannibal age']

fig, axs = plt.subplots(1, 4)
axs = axs.ravel()
es_df = es_df[es_df.index <= df.index[-1]]
for i, (ax, f, title) in enumerate(zip(axs[4:], features[:-1], ('a)', 'b)', 'c)', 'd)'))):
    ax.plot(es_df.index, es_df[f].values, label='CMA-ES')
    ax.plot(df.index, df[f].values, label='E-VDN')
    ax.set_ylabel(f)
    ax.set_xlabel('Wall time (hours)')
    if not i:
        ax.legend(loc='best')

fig, ax = plt.subplots()
ax = axs[2]
ax.plot(df.index, df["Cannibalism's victim age"], label="Cannibal's victim")
ax.plot(df.index, df['Cannibal age'], label='Cannibal')
ax.set_ylabel('Age')
ax.set_xlabel('Training episodes (thousands)')
ax.legend(loc='best')


fig, ax = plt.subplots()
ax = axs[1]
ax.plot(es_df.index, es_df["Avg_cannibalism_victim_age"], label="Cannibal's victim")
ax.plot(es_df.index, es_df['Avg_cannibal_age'], label='Cannibal')
ax.set_ylabel('Age')
ax.set_xlabel('Training episodes (thousands)')
ax.legend(loc='best')
