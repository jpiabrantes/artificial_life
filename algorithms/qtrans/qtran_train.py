import os
import pickle
from time import time
from collections import namedtuple

import ray
import numpy as np
import tensorflow as tf

from utils.misc import Timer
from utils.filters import MeanStdFilter, FilterManager
from algorithms.qtrans.sampler import Sampler
from algorithms.qtrans.trainer import Trainer
from models.base import Qtran


Networks = namedtuple('Networks', ('main', 'target'))
Weights = namedtuple('Weights', ('main', 'target'))


class QtranTrainer:
    def __init__(self, env_creator,  brain_kwargs, population_size, gamma=0.99, start_eps=1, end_eps=0.1,
                 annealing_steps=50000, tau=0.001, n_trainers=5, n_samplers=20, num_envs_per_sampler=20,
                 num_of_steps_per_sample=1, learning_rate=0.0005, opt_coeff=1, nopt_coeff=1):
        env = env_creator()
        self.env = env

        exp_name = 'basic'
        algorithm_folder = os.path.dirname(os.path.abspath(__file__))
        exp_folder = os.path.join(algorithm_folder, 'checkpoints', env.name, exp_name)
        tensorboard_folder = os.path.join(exp_folder, 'tensorboard', 'qtran_%d' % int(time()))
        brain = None
        self.weights = {}
        with tf.device('/cpu:0'):
            for species_index in range(population_size):
                brain = Qtran(**brain_kwargs)
                self.weights[species_index] = Weights(brain.get_weights(), brain.Q.get_weights())
                os.makedirs(os.path.join(exp_folder, str(species_index)), exist_ok=True)

        self.actor_filters = {'ActorObsFilter': MeanStdFilter(shape=self.env.observation_space.shape)}
        self.critic_filters = {'CriticObsFilter': MeanStdFilter(shape=self.env.critic_observation_shape)}
        filter_manager = FilterManager()

        species_dict = {species_index: {'steps': 0, 'eps': start_eps, 'optimiser_weights': None}
                        for species_index in range(population_size)}
        samplers = [Sampler.remote(env_creator, num_envs_per_sampler, num_of_steps_per_sample, brain_kwargs)
                    for _ in range(n_samplers)]
        trainers = [Trainer.remote(brain_kwargs, gamma, learning_rate, opt_coeff, nopt_coeff,
                                   env.critic_observation_shape) for _ in range(n_trainers)]

        # Set the rate of random action decrease.

        train_summary_writer = tf.summary.create_file_writer(tensorboard_folder)
        episodes = 0
        total_steps = 0
        while True:
            total_steps += 1
            with Timer() as sampling_time:
                filter_manager.synchronize(self.actor_filters, samplers)
                weights_id = ray.put({species_index: w.main for species_index, w in self.weights.items()})
                eps_id = ray.put({species_index: dict_['eps'] for species_index, dict_ in species_dict.items()})
                species_buffers_list = ray.get([sampler.rollout.remote(weights_id, eps_id) for sampler in samplers])
                species_buffers = self._concatenate_sampler_results(species_buffers_list)

            training_species_idx = []
            training_results = []
            i = 0
            with Timer() as training_time:
                filter_manager.synchronize(self.critic_filters, trainers)
                for species_index, buffer in species_buffers.items():
                    dict_ = species_dict[species_index]
                    dict_['steps'] += len(buffer.step_buf)
                    if dict_['steps'] > annealing_steps:
                        coeff = (dict_['steps']-annealing_steps)/annealing_steps
                        dict_['eps'] = max(coeff * 0.01 + (1 - coeff) * end_eps, 0.01)
                    else:
                        coeff = dict_['steps']/annealing_steps
                        dict_['eps'] = max(coeff*end_eps+(1-coeff)*start_eps, end_eps)
                    training_species_idx.append(species_index)
                    training_results.append(trainers[i].train.remote(self.weights[species_index], buffer.step_buf,
                                                                     buffer.sta_act_spe_buf, buffer.loc_buf,
                                                                     buffer.n_sta_act_spe_buf,
                                                                     species_dict[species_index]['optimiser_weights'],
                                                                     species_index))
                    i = (i + 1) % len(trainers)
                training_results = ray.get(training_results)

            training_losses = []
            with Timer() as saving_time:
                for species_index, (main_weights, optimiser_weights, loss) in zip(training_species_idx, training_results):
                    species_dict[species_index]['optimiser_weights'] = optimiser_weights
                    training_losses.append(loss)
                    brain.set_weights(main_weights)

                    target_w = []
                    for mw, tw in zip(brain.Q.get_weights(), self.weights[species_index].target):
                        target_w.append(tau * mw + (1-tau)*tw)
                    self.weights[species_index] = Weights(main_weights, target_w)
                    species_folder = os.path.join(exp_folder, str(species_index))
                    with open(os.path.join(species_folder, 'weights.pkl'), 'wb') as f:
                        pickle.dump(self.weights[species_index], f)

            if not (total_steps % 10):
                with open(os.path.join(exp_folder, 'variables.pkl'), 'wb') as f:
                    pickle.dump((self.actor_filters, self.critic_filters, episodes, total_steps), f)

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
                    result[species_index].add_buffer(buffer)
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
    from envs.deadly_colony.deadly_colony import DeadlyColony
    from envs.deadly_colony.env_config import env_default_config

    config = env_default_config.copy()
    config['greedy_reward'] = True
    env_creator = lambda: DeadlyColony(config)
    env = env_creator()

    q_kwargs = {'hidden_units': [512, 256, 128], 'observation_space': env.observation_space}
    rows, cols, depth = env.critic_observation_shape
    depth += 1  # will give state-actions
    Q_kwargs = {'conv_sizes': [(32, (6, 6), (3, 3)), (64, (4, 4), (2, 2)), (64, (3, 3), (1, 1))],
                'fc_sizes': [512],
                'input_shape': (rows, cols, depth)}
    V_kwargs = Q_kwargs.copy()
    brain_kwargs = {'q_kwargs': q_kwargs, 'Q_kwargs': Q_kwargs, 'V_kwargs': V_kwargs, 'action_space': env.action_space}

    ray.init(local_mode=False)
    trainer = QtranTrainer(env_creator, brain_kwargs, population_size=5)
