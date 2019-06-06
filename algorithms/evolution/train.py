import os
import pickle

import tensorflow as tf
import numpy as np
import ray
from tqdm import tqdm

from algorithms.evolution.worker import Worker
from algorithms.evolution.es import CMAES
from models.base import create_vision_and_fc_network
from algorithms.evolution.helpers import load_variables
from utils.filters import FilterManager, MeanStdFilter

# from envs.bacteria_colony.bacteria_colony import BacteriaColony
# from envs.bacteria_colony.env_config import env_default_config
from envs.deadly_colony.deadly_colony import DeadlyColony
from envs.deadly_colony.env_config import env_default_config

eager = False
if not eager:
    tf.compat.v1.disable_eager_execution()

env_creator = lambda: DeadlyColony(env_default_config)
env = env_creator()
policy_args = {'conv_sizes': [(32, (3, 3), 1), (32, (3, 3), 1)],
               'fc_sizes': [16],
               'last_fc_sizes': [32],
               'conv_input_shape': env.actor_terrain_obs_shape,
               'fc_input_length': np.prod(env.observation_space.shape) - np.prod(env.actor_terrain_obs_shape),
               'num_outputs': env.action_space.n,
               'obs_input_shape': env.observation_space.shape}

policy_creator = lambda: create_vision_and_fc_network(**policy_args)
po = policy_creator()
rollouts_per_group = 1
popsize = 65
n_groups_per_sample = 10
n_agents = 5
std0_list = [1.0]*n_agents
seed = 0
generations = 1000
save_freq = 10
n_workers = 40 #cpu_count()
load = True
assert popsize % n_agents == 0, 'population size must be a multiple of n_agents'

ray.init(local_mode=(n_workers == 1))

if __name__ == '__main__':
    policy = policy_creator()
    checkpoint_path = './checkpoints/{}'.format(env.name)
    os.makedirs(checkpoint_path, exist_ok=True)
    opts_list = [{'popsize': popsize, 'seed': seed} for _ in range(n_agents)]
    if load:
        last_generation, mu0_list, stds_list, horizons_list, returns_list, filters = load_variables(env)
        for opts, stds in zip(opts_list, stds_list):
            opts['CMA_stds'] = stds
    else:
        n_params = policy.count_params()
        mu0_list = [[0]*n_params]*n_agents
        horizons_list, returns_list = [], []
        last_generation = 0
        filters = {'MeanStdFilter': MeanStdFilter(shape=env.observation_space.shape)}
    es_list = [CMAES(mu0_list[i], sigma0=std0_list[i], opts=opts_list[i]) for i in range(n_agents)]
    workers = [Worker.remote(i, env_creator, [policy_creator] * n_agents, rollouts_per_group,
                             normalise_observation=True) for i in range(n_workers)]

    filter_manager = FilterManager()

    for generation in tqdm(range(last_generation, last_generation+generations)):
        if not generation:
            print('Sampling %d episodes per generation!' % (popsize * rollouts_per_group * n_groups_per_sample))
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
            result_ids.append(worker.rollout.remote([solution_ids[j] for j in group], group))
        results = ray.get(result_ids)
        returns, horizons = zip(*results)
        negative_fitness_list = [np.zeros((popsize,), np.float32) for _ in range(n_agents)]
        for return_dict in returns:
            for k, v in return_dict.items():
                index = int(k)
                es_index = index//popsize
                pop_index = index % popsize
                negative_fitness_list[es_index][pop_index] -= v * 1 / n_groups_per_sample

        for es, solutions, negative_fitness in zip(es_list, solutions_list, negative_fitness_list):
            es.tell(solutions, negative_fitness)

        horizons_list.append((np.mean(horizons), np.min(horizons), np.max(horizons)))
        returns_list.append([(-np.mean(negative_fitness), -np.max(negative_fitness), -np.min(negative_fitness)) for
                            negative_fitness in negative_fitness_list])
        print('\nGeneration {}'.format(generation))
        print('Horizon mean: {}, min: {}, max: {}, std: {}'.format(np.mean(horizons), np.min(horizons), np.max(horizons)
                                                                   , np.std(horizons)))
        for i, negative_fitness in enumerate(negative_fitness_list):
            print('ES: %d' % i)
            print('Fitness mean: {}, min: {}, max: {}, std: {}'.format(-np.mean(negative_fitness),
                                                                       -np.max(negative_fitness),
                                                                       -np.min(negative_fitness),
                                                                       np.std(negative_fitness)))
        if (generation % save_freq) == save_freq-1:
            with open(os.path.join(checkpoint_path, 'last_checkpoint.txt'), 'w') as f:
                f.write('%d\n' % generation)
            filters_to_save = {k: v.as_serializable() for k, v in filters.items()}
            means, stds = [], []
            for es in es_list:
                means.append(es.result.xbest)
                stds.append(es.result.stds)
            with open(os.path.join(checkpoint_path, str(generation) + '_variables.pkl'), 'wb') as f:
                pickle.dump((means, stds, horizons_list, returns_list, filters_to_save), f)
