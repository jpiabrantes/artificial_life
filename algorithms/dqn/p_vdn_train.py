import os
from time import time
from collections import namedtuple

import ray
import numpy as np
import tensorflow as tf

from utils.misc import Timer
from utils.filters import MeanStdFilter, FilterManager
from algorithms.dqn.sampler import Sampler


Networks = namedtuple('Networks', ('main', 'target'))
Weights = namedtuple('Weights', ('main', 'target'))


class DQN_trainer:
    def __init__(self, env_creator,  brain_creator, population_size, batch_size=32, update_freq=4, gamma=0.99,
                 start_eps=1, end_eps=0.1, annealing_steps=50000, num_epochs=5000, pre_train_steps=250, tau=0.001,
                 n_samplers=7, num_envs_per_sampler=40, num_of_steps_per_sample=1):
        env = env_creator()
        self.env = env

        exp_name = 'basic'
        algorithm_folder = os.path.dirname(os.path.abspath(__file__))
        exp_folder = os.path.join(algorithm_folder, 'checkpoints', env.name, exp_name)
        tensorboard_folder = os.path.join(exp_folder, 'tensorboard', 'dqn_%d' % int(time()))

        self.weights = {}
        for species_index in range(population_size):
            brain = brain_creator()
            weights = brain.get_weights()
            self.weights[species_index] = Weights(weights, weights)

        main_qn = brain_creator()
        target_qn = brain_creator()

        self.filters = {'ActorObsFilter': MeanStdFilter(shape=self.env.observation_space.shape)}
        filter_manager = FilterManager()

        species_dict = {species_index: {'steps': 0, 'eps': start_eps,
                                        'optimiser': tf.keras.optimizers.Adam(learning_rate=0.001)}
                        for species_index in range(population_size)}
        samplers = [Sampler.remote(env_creator, num_envs_per_sampler, num_of_steps_per_sample,
                                   brain_creator) for _ in range(n_samplers)]

        # Set the rate of random action decrease.

        train_summary_writer = tf.summary.create_file_writer(tensorboard_folder)
        episodes = 0
        training_losses = []
        total_steps = 0
        while True:
            with Timer() as sampling_time:
                filter_manager.synchronize(self.filters, samplers)
                weights_id = ray.put({species_index: w.main for species_index, w in self.weights.items()})
                eps_id = ray.put({species_index: dict_['eps'] for species_index, dict_ in species_dict.items()})
                species_buffers_list = ray.get([sampler.rollout.remote(weights_id, eps_id) for sampler in samplers])
                species_buffers = self._concatenate_sampler_results(species_buffers_list)

            with Timer() as training_time:
                for species_index, buffer in species_buffers.items():
                    dict_ = species_dict[species_index]
                    dict_['steps'] += len(buffer.buffer)
                    if dict_['steps'] > pre_train_steps:
                        if dict_['steps'] > annealing_steps:
                            coeff = (dict_['steps']-annealing_steps)/annealing_steps
                            dict_['eps'] = max(coeff * 0.01 + (1 - coeff) * end_eps, 0.01)
                        else:
                            coeff = dict_['steps']/annealing_steps
                            dict_['eps'] = max(coeff*end_eps+(1-coeff)*start_eps, end_eps)

                        '''
                        steps: list of size batch_size with a list of size n_agents with experiences
                        '''
                        steps = buffer.sample_steps(batch_size)  # Get a random batch of experiences.

                        main_qn.set_weights(self.weights[species_index].main)
                        target_qn.set_weights(self.weights[species_index].target)

                        # Below we perform the Double-DQN update to the target Q-values
                        list_of_obs = [None]*len(steps)
                        list_of_act = [None]*len(steps)
                        target_q = np.zeros((len(steps),), np.float32)
                        for i, step in enumerate(steps):
                            # step: list of experiences of size #n_agents
                            step = np.stack(step)  # n_agents, 5
                            obs, act, rew, n_obs, done = [np.array(x.tolist(), t)
                                                          for x, t in zip(step.T,
                                                                          (np.float32, np.int32, np.float32,
                                                                           np.float32, np.bool))]
                            # obs: n_agents, obs_dim
                            list_of_obs[i] = obs
                            list_of_act[i] = act
                            assert len(np.unique(rew)) == 1

                            target_q[i] = rew[0]
                            alive_mask = np.logical_not(done)
                            if np.any(alive_mask):
                                # compute Qtot for alive agents
                                n_actions = main_qn.get_actions(n_obs[alive_mask], 0)
                                q_out = target_qn.q.predict(n_obs[alive_mask])
                                double_q = q_out[range(len(n_actions)), n_actions]
                                target_q[i] += gamma * np.sum(double_q)

                        # Update the network with our target values.
                        with tf.GradientTape() as t:
                            loss = self.td_loss(main_qn(list_of_obs, list_of_act), target_q)
                        grads = t.gradient(loss, main_qn.variables)
                        # TODO: give a training loss per species
                        training_losses.append(loss)
                        dict_['optimiser'].apply_gradients(zip(grads, main_qn.variables))
                        #result = main_qn.train_on_batch(x=list_of_obs, y=target_q)

                        target_w = []
                        for mw, tw in zip(main_qn.get_weights(), target_qn.get_weights()):
                            target_w.append(tau * mw + (1-tau)*tw)
                        self.weights[species_index] = Weights(main_qn.get_weights(), target_w)

            # get ep_stats from samplers
            metrics = {}

            ep_metrics = self._concatenate_ep_stats(ray.get([s.get_ep_stats.remote() for s in samplers]))
            if ep_metrics['EpisodesThisIter']:
                metrics.update(ep_metrics)
                episodes += ep_metrics['EpisodesThisIter']

            training_loss = 0 if not len(training_losses) else np.mean(training_losses)
            training_losses = []
            training_metrics = {'QLoss': training_loss, 'Episodes': episodes, 'Sampling time': sampling_time.interval,
                                'Training time': training_time.interval}
            metrics.update(training_metrics)

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
    def td_loss(targets, qtot):
        return tf.keras.losses.mean_squared_error(targets, qtot)

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

    ray.init()
    trainer = DQN_trainer(env_creator, brain_creator, population_size=5)
