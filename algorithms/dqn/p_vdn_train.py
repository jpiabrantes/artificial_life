import os
import pickle
from time import time
from collections import namedtuple

import ray
import numpy as np
import tensorflow as tf

from utils.misc import Timer
from utils.filters import MeanStdFilter, FilterManager
from algorithms.dqn.sampler import Sampler
from algorithms.dqn.trainer import Trainer


Networks = namedtuple('Networks', ('main', 'target'))
Weights = namedtuple('Weights', ('main', 'target'))


class VDNTrainer:
    def __init__(self, env_creator,  brain_creator, population_size, gamma=0.99,
                 start_eps=1, end_eps=0.1, annealing_steps=50000, tau=0.001, n_trainers=5,
                 n_samplers=40, num_envs_per_sampler=7, num_of_steps_per_sample=1, learning_rate=0.0005):
        env = env_creator()
        self.env = env

        exp_name = 'basic'
        algorithm_folder = os.path.dirname(os.path.abspath(__file__))
        exp_folder = os.path.join(algorithm_folder, 'checkpoints', env.name, exp_name)
        tensorboard_folder = os.path.join(exp_folder, 'tensorboard', 'dqn_%d' % int(time()))

        self.weights = {}
        for species_index in range(population_size):
            brain = brain_creator()
            main_weights = brain.get_weights()
            self.weights[species_index] = Weights(main_weights, main_weights)

        self.filters = {'ActorObsFilter': MeanStdFilter(shape=self.env.observation_space.shape)}
        filter_manager = FilterManager()

        species_dict = {species_index: {'steps': 0, 'eps': start_eps, 'optimiser_weights': None}
                        for species_index in range(population_size)}
        samplers = [Sampler.remote(env_creator, num_envs_per_sampler, num_of_steps_per_sample,
                                   brain_creator) for _ in range(n_samplers)]
        trainers = [Trainer.remote(brain_creator, gamma, learning_rate) for _ in range(n_trainers)]

        # Set the rate of random action decrease.

        train_summary_writer = tf.summary.create_file_writer(tensorboard_folder)
        episodes = 0
        total_steps = 0
        while True:
            total_steps += 1
            with Timer() as sampling_time:
                filter_manager.synchronize(self.filters, samplers)
                weights_id = ray.put({species_index: w.main for species_index, w in self.weights.items()})
                eps_id = ray.put({species_index: dict_['eps'] for species_index, dict_ in species_dict.items()})
                species_buffers_list = ray.get([sampler.rollout.remote(weights_id, eps_id) for sampler in samplers])
                species_buffers = self._concatenate_sampler_results(species_buffers_list)

            training_species_idx = []
            training_results = []
            i = 0
            with Timer() as training_time:
                for species_index, buffer in species_buffers.items():
                    dict_ = species_dict[species_index]
                    dict_['steps'] += len(buffer.buffer)
                    if dict_['steps'] > annealing_steps:
                        coeff = (dict_['steps']-annealing_steps)/annealing_steps
                        dict_['eps'] = max(coeff * 0.01 + (1 - coeff) * end_eps, 0.01)
                    else:
                        coeff = dict_['steps']/annealing_steps
                        dict_['eps'] = max(coeff*end_eps+(1-coeff)*start_eps, end_eps)
                    training_species_idx.append(species_index)
                    training_results.append(trainers[i].train.remote(self.weights[species_index], buffer.buffer,
                                                                     species_dict[species_index]['optimiser_weights']))
                    i = (i + 1) % len(trainers)
                training_results = ray.get(training_results)

            training_losses = []
            with Timer() as saving_time:
                for species_index, (main_weights, optimiser_weights, loss) in zip(training_species_idx, training_results):
                    species_dict[species_index]['optimiser_weights'] = optimiser_weights
                    training_losses.append(loss)

                    target_w = []
                    for mw, tw in zip(main_weights, self.weights[species_index].target):
                        target_w.append(tau * mw + (1-tau)*tw)
                    self.weights[species_index] = Weights(main_weights, target_w)
                    species_folder = os.path.join(exp_folder, str(species_index))
                    with open(os.path.join(species_folder, 'weights.pkl', 'wb')) as f:
                        pickle.dump(self.weights[species_index], f)

            # get ep_stats from samplers
            metrics = {'Episodes': episodes, 'Sampling time': sampling_time.interval,
                       'Training time': training_time.interval, 'Saving time': saving_time.interval,
                       'Training loss': np.mean(training_losses)}

            ep_metrics = self._concatenate_ep_stats(ray.get([s.get_ep_stats.remote() for s in samplers]))
            if ep_metrics['EpisodesThisIter']:
                metrics.update(ep_metrics)
                episodes += ep_metrics['EpisodesThisIter']

            mean_eps, mean_steps = [], []
            for species_index, dict_ in species_dict.items():
                mean_eps.append(dict_['eps'])
                mean_steps.append(dict_['steps'])
            metrics.update({'Eps': np.mean(mean_eps), 'Steps': np.mean(mean_steps)})

            print('\n\nEpisodes: ', episodes)
            with train_summary_writer.as_default():
                for k, v in metrics.items():
                    print(k, v)
                    tf.summary.scalar(k, v, step=total_steps)

    @staticmethod
    def _concatenate_sampler_results(species_buffers_list):
        result = {}
        for species_buffers in species_buffers_list:
            for species_index, buffer in species_buffers.items():
                if species_index in result:
                    result[species_index].add_buffer(buffer.buffer)
                else:
                    result[species_index] = buffer
        return result

    @staticmethod
    def _concatenate_ep_stats(stats_list, min_and_max=False, include_std=False):
        total_stats = None
        for stats in stats_list:
            if total_stats is None:
                total_stats = stats
            else:
                for k, v in stats.items():
                    total_stats[k].extend(v)

        metrics = {'EpisodesThisIter': len(total_stats['ep_len'])}
        if metrics['EpisodesThisIter']:
            for k, v in total_stats.items():
                metrics['Avg_' + k] = np.mean(v)
                if min_and_max:
                    metrics['Min_' + k] = np.min(v)
                    metrics['Max_' + k] = np.max(v)
                if include_std:
                    metrics['Std_' + k] = np.std(v)
        return metrics


if __name__ == '__main__':
    from models.base import VDNMixer
    from envs.deadly_colony.deadly_colony import DeadlyColony
    from envs.deadly_colony.env_config import env_default_config

    config = env_default_config.copy()
    config['greedy_reward'] = True
    env_creator = lambda: DeadlyColony(config)
    env = env_creator()
    q_kwargs = {'hidden_units': [512, 256, 128],
                'observation_space': env.observation_space,
                'action_space': env.action_space}
    brain_creator = lambda: VDNMixer(**q_kwargs)

    ray.init(local_mode=False)
    trainer = VDNTrainer(env_creator, brain_creator, population_size=5)
