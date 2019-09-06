import os
from collections import defaultdict

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

sns.set_style("darkgrid", {"axes.facecolor": ".9"})


def load_df(path):
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
    # df.wall_time -= df.wall_time.iloc[0]
    # df.wall_time /= 3600
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
    return df

path = '/home/joao/github/artificial_life/algorithms/maeq/checkpoints/SexualColony-v0/basic/tensorboard'
df = None
for folder in sorted(os.listdir(path)):
    new_path = os.path.join(path, folder)
    new_path = os.path.join(new_path, os.listdir(new_path)[0])
    tdf = load_df(new_path)
    if df is None:
        df = tdf
    else:
        df.append(tdf)

df.sort_index(inplace=True)
axs[-1].plot(df.index/1000, df['Average population size'])
axs[-1].set_ylabel('Average population size')
axs[-1].set_xlabel('Training episodes (thousands)')

axs[-2].plot(df.index/1000, df['Life expectancy'])
axs[-2].set_ylabel('Life expectancy')
axs[-2].set_xlabel('Training episodes (thousands)')

axs[-3].plot(df.index/1000, df['Babies born'])
axs[-3].set_ylabel('Babies born')
axs[-3].set_xlabel('Training episodes (thousands)')
