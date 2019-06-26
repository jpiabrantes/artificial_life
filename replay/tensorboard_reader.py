from collections import defaultdict

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

plt.style.use('bmh')


# read ES
path = '~/github/artificial_life/algorithms/evolution/checkpoints/DeadlyColony-v0/ep_stats.csv'
es_df = pd.read_csv(path)
es_df.rename(columns={'Avg_babies_born': 'Babies Born', 'Avg_average_population': 'Average Population Size',
                      'Avg_life_expectancy': 'Life Expectancy'}, inplace=True)
es_df.set_index('episodes', inplace=True)





path = '/home/joao/github/artificial_life/algorithms/dqn/checkpoints/DeadlyColony-v0/basic/tensorboard/vdn_1561399561/events.out.tfevents.1561399561.ip-172-31-30-186.41430.786.v2'


wall_times = defaultdict(list)
data = defaultdict(dict)
for e in tf.compat.v1.train.summary_iterator(path):
    for v in e.summary.value:
        if v.tag in ('test_Avg_babies_born', 'test_Avg_average_population', 'test_Avg_life_expectancy', 'Episodes'):
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
df.set_index('Episodes', inplace=True)
df.rename(columns={'test_Avg_babies_born': 'Babies Born', 'test_Avg_average_population': 'Average Population Size',
                   'test_Avg_life_expectancy': 'Life Expectancy'}, inplace=True)


df = es_df
features = ['Average Population Size', 'Babies Born', 'Life Expectancy']

fig, axs = plt.subplots(1, 3)
for ax, f in zip(axs, features):
    ax.plot(df.index, df[f].values)
    ax.set_ylabel(f)
    ax.set_xlabel('episodes')
