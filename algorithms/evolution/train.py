import os
import pickle
from time import time

import pandas as pd
import tensorflow as tf
import numpy as np
import ray
from tqdm import tqdm

from algorithms.evolution.worker import Worker
from algorithms.evolution.es import CMAES
from models.base import DiscreteActor
from algorithms.evolution.helpers import load_variables
from utils.filters import FilterManager, MeanStdFilter
from utils.buffers import concatenate_ep_stats

# from envs.bacteria_colony.bacteria_colony import BacteriaColony
# from envs.bacteria_colony.env_config import env_default_config
from envs.deadly_colony.deadly_colony import DeadlyColony
from envs.deadly_colony.env_config import env_default_config

eager = False
if not eager:
    tf.compat.v1.disable_eager_execution()

env_creator = lambda: DeadlyColony(env_default_config)
env = env_creator()
hidden_sizes = [512, 256, 128]
num_outputs = env.action_space.n
input_shape = env.observation_space.shape


def policy_creator():
    input_layer = tf.keras.layers.Input(shape=input_shape)
    out = input_layer
    for h in hidden_sizes:
        out = tf.keras.layers.Dense(h, activation='relu')(out)
    out = tf.keras.layers.Dense(num_outputs, activation='linear')(out)
    return DiscreteActor(inputs=input_layer, outputs=[out])


def save_ep_stats(ep_stats, path):
    df = pd.DataFrame(ep_stats)
    if os.path.isfile(path):
        df.to_csv(path, header=False, mode='a')
    else:
        df.to_csv(path)


po = policy_creator()
rollouts_per_group = 1
popsize = 65
n_groups_per_sample = 10
n_agents = 5
std0_list = [1.0]*n_agents
seed = 0
generations = 10000
save_freq = 10
n_workers = 40
load = False
assert popsize % n_agents == 0, 'population size must be a multiple of n_agents'

ray.init(local_mode=(n_workers == 1))
start_time = time()
if __name__ == '__main__':
    episodes = 0
    policy = policy_creator()
    checkpoint_path = './checkpoints/{}'.format(env.name)
    os.makedirs(checkpoint_path, exist_ok=True)
    opts_list = [{'popsize': popsize, 'seed': seed} for _ in range(n_agents)]
    if load:
        last_generation, mu0_list, stds_list, filters = load_variables(env)
        for opts, stds in zip(opts_list, stds_list):
            opts['CMA_stds'] = stds
    else:
        n_params = policy.count_params()
        mu0_list = [[0]*n_params]*n_agents
        last_generation = 0
        filters = {'MeanStdFilter': MeanStdFilter(shape=env.observation_space.shape)}
    es_list = [CMAES(mu0_list[i], sigma0=std0_list[i], opts=opts_list[i]) for i in range(n_agents)]
    workers = [Worker.remote(i, env_creator, [policy_creator] * n_agents, rollouts_per_group,
                             normalise_observation=True) for i in range(n_workers)]

    filter_manager = FilterManager()

    for generation in tqdm(range(last_generation, last_generation+generations)):
        if not generation:
            episodes_per_gen = popsize * rollouts_per_group * n_groups_per_sample
            print('Sampling %d episodes per generation!' % episodes_per_gen)
        filter_manager.synchronize(filters, workers)
        solutions_list = [es.ask() for es in es_list]
        solution_ids = [ray.put(solution) for solutions in solutions_list for solution in solutions]

        # make groups
        indices = np.arange(popsize*n_agents).reshape(n_agents, popsize)  # n_agents x popsize

        groups = []
        for sample_i in range(n_groups_per_sample):
            for r in range(indices.shape[0]):
                np.random.shuffle(indices[r, :])
            groups.extend(indices.T.tolist())

        # gather fitness
        result_ids = []
        for i, group in enumerate(groups):
            i = i % n_workers
            worker = workers[i]
            result_ids.append(worker.rollout.remote([solution_ids[j] for j in group], group, generation))
        results = ray.get(result_ids)
        returns = zip(results)
        negative_fitness_list = [np.zeros((popsize,), np.float32) for _ in range(n_agents)]
        for return_dict in returns:
            for k, v in return_dict.items():
                index = int(k)
                es_index = index//popsize
                pop_index = index % popsize
                negative_fitness_list[es_index][pop_index] -= v * 1 / n_groups_per_sample

        for es, solutions, negative_fitness in zip(es_list, solutions_list, negative_fitness_list):
            es.tell(solutions, negative_fitness)

        ep_stats = [s.remote.get_stats() for s in workers]
        ep_stats = concatenate_ep_stats(ep_stats)
        ep_stats['time'] = time() - start_time
        ep_stats['episodes'] = episodes_per_gen*generation

        print('\nGeneration {}'.format(generation))

        for i, negative_fitness in enumerate(negative_fitness_list):
            print('ES: %d' % i)
            print('Fitness mean: {}, min: {}, max: {}, std: {}'.format(-np.mean(negative_fitness),
                                                                       -np.max(negative_fitness),
                                                                       -np.min(negative_fitness),
                                                                       np.std(negative_fitness)))
            for k, v in ep_stats.items():
                print(k, v)

        save_ep_stats(ep_stats, os.path.join(checkpoint_path, 'ep_stats.csv'))
        if (generation % save_freq) == save_freq-1:
            with open(os.path.join(checkpoint_path, 'last_checkpoint.txt'), 'w') as f:
                f.write('%d\n' % generation)
            filters_to_save = {k: v.as_serializable() for k, v in filters.items()}
            means, stds = [], []
            for es in es_list:
                means.append(es.result.xbest)
                stds.append(es.result.stds)
            with open(os.path.join(checkpoint_path, str(generation) + '_variables.pkl'), 'wb') as f:
                pickle.dump((means, stds, filters_to_save), f)
