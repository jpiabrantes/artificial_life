'''
#TODO: implement distribution strategy
#TODO: document
#TODO: rename agents to entities
'''
from collections import defaultdict
import os
import pickle
from datetime import datetime

import scipy
import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
import ray

from utils.misc import Timer, SpeciesSampler, SpeciesSamplerManager, get_explained_variance
from utils.filters import FilterManager, MeanStdFilter
from utils.coma_helper import get_states_actions_for_locs
from algorithms.coma.sampler import Sampler


class MultiAgentCOMATrainer:
    def __init__(self, env_creator, ac_creator, population_size, seed=0, n_workers=1, sample_batch_size=500,
                 batch_size=250, gamma=1., lamb=0.95, clip_ratio=0.2, pi_lr=3e-4, value_lr=1e-3, train_pi_iters=80,
                 train_v_iters=80, target_kl=0.01, save_freq=10, normalise_advantages=False,
                 normalise_observation=False, entropy_coeff=0.01):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.entropy_coeff = entropy_coeff
        self.clip_ratio = clip_ratio
        self.normalize_advantages = normalise_advantages
        self.n_workers = n_workers
        self.sample_batch_size = sample_batch_size
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.train_v_iters = train_v_iters
        self.train_pi_iters = train_pi_iters
        self.target_kl = target_kl
        self.population_size = population_size

        self.filter_manager = FilterManager()
        self.species_sampler_manager = SpeciesSamplerManager()

        env = env_creator()
        self.env = env
        if normalise_observation:
            self.filters = {'ActorObsFilter': MeanStdFilter(shape=self.env.observation_space.shape),
                            'CriticObsFilter': MeanStdFilter(shape=self.env.critic_observation_shape)}
        else:
            self.filters = {}

        self.n_acs = env.n_agents
        self.ac_creator = ac_creator
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            self.ac = ac_creator()
            self.ac.critic.compile(optimizer=kr.optimizers.Adam(learning_rate=value_lr), loss=self._value_loss)
            self.ac.actor.compile(optimizer=kr.optimizers.Adam(learning_rate=pi_lr), loss=self._surrogate_loss)

        self.steps_per_worker = sample_batch_size // n_workers
        if batch_size % n_workers:
            self.batch_size = n_workers * self.steps_per_worker
            print('WARNING: the batch_size was changed to: %d' % self.batch_size)
        self.samplers = [Sampler.remote(self.steps_per_worker, gamma, lamb, env_creator, ac_creator, i,
                                        population_size, normalise_observation) for i in range(n_workers)]

    def train(self, epochs, generation, save_freq=10):
        generation_folder = './checkpoints/{}/{}'.format(self.env.name, generation)
        tensorboard_folder = os.path.join(generation_folder, 'tensorboard')
        if os.path.isdir(generation_folder):
            weights, self.filters, species_sampler, episodes, training_samples = self._load_generation(generation)
        else:
            weights, species_sampler = self._create_generation(generation)
            episodes, training_samples = 0, 0
        weights_id_list = [ray.put(w) for w in weights]

        species_buffers = {}
        train_summary_writer = tf.summary.create_file_writer(tensorboard_folder)
        for epoch in range(epochs):
            samples_this_iter = 0
            with Timer() as sampling_time:
                results_list = ray.get([worker.rollout.remote(weights_id_list) for worker in self.samplers])
            self._concatenate_samplers_results(results_list, species_buffers)
            self.filter_manager.synchronize(self.filters, self.samplers)
            self.species_sampler_manager.synchronize(species_sampler, self.samplers)
            pi_optimisation_time, v_optimisation_time, pop_stats = [], [], []
            # TODO: parallelize this for-loop amongst GPU workers ?
            for species_index, variables in species_buffers.items():
                obs, act, adv, td, old_log_probs, loc, pi, state_action_per_iter, samples_per_iter = variables
                if len(obs) < self.batch_size:
                    continue

                if self.normalize_advantages:
                    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

                for var, w in zip(self.ac.variables, weights[species_index]):
                    var.load(w)
                act_adv_logs = np.concatenate([act[:, None], adv[:, None], old_log_probs[:, None]], axis=-1)
                old_policy_loss, old_value_loss = None, None
                indices = np.arange(len(obs))
                with Timer() as pi_optimisation_timer:
                    for i in range(self.train_pi_iters):
                        result = self.ac.actor.fit(obs[indices], act_adv_logs[indices], batch_size=self.batch_size,
                                                   epochs=1, verbose=False)
                        if not i:
                            old_policy_loss = np.mean(result.history['loss'])
                        else:
                            np.random.shuffle(indices)
                        logits = self.ac.actor.predict(obs, batch_size=len(obs))
                        all_log_probs = np.log(scipy.special.softmax(logits, axis=1))
                        log_probs = all_log_probs[np.arange(len(act)), act.astype(np.int32)]
                        # a sample estimate on the kl divergence
                        kl = -np.mean(log_probs - old_log_probs)
                        if abs(kl) > 1.5 * self.target_kl:
                            print('stopped training at iter {}'.format(i))
                            break

                states_actions = np.empty((len(loc),) + state_action_per_iter.shape[1:])
                ptr = 0
                for samples, state_action in zip(samples_per_iter, state_action_per_iter):
                    states_actions[ptr: ptr+samples] = get_states_actions_for_locs(state_action, loc[ptr: ptr+samples],
                                                                                   self.env.n_rows, self.env.n_cols)
                    ptr += samples

                qs = self.ac.critic.predict(states_actions, batch_size=len(states_actions))
                q = qs[np.arange(len(act)), act]
                act_td = np.concatenate([act[:, None], td[:, None]], axis=-1)
                with Timer() as v_optimisation_timer:
                    for i in range(self.train_v_iters):
                        # TODO: make the fit do the shuffling
                        result = self.ac.critic.fit(states_actions[indices], act_td[indices],
                                                    batch_size=self.batch_size, epochs=1, verbose=False)
                        if not i:
                            old_value_loss = np.mean(result.history['loss'])
                        else:
                            np.random.shuffle(indices)

                weights[species_index] = self.ac.get_weights()
                weights_id_list[species_index] = ray.put(weights[species_index])
                checkpoint_path = os.path.join(generation_folder, str(species_index), str(episodes))
                self.ac.save_weights(checkpoint_path)

                samples_this_iter += len(obs)
                training_samples += len(obs)
                pi_optimisation_time += [pi_optimisation_timer.interval]
                v_optimisation_time += [v_optimisation_timer.interval]
                probs = scipy.special.softmax(logits, axis=1)
                entropy = np.mean(-np.sum(np.where(probs == 0, 0, probs*np.log(probs)), axis=1))
                explained_variance = get_explained_variance(td, q)
                key_value_pairs = [('LossV', old_value_loss), ('Explained Ret Variance', explained_variance),
                                   ('KL', kl), ('Entropy', entropy), ('LossPi', old_policy_loss),
                                   ('TD(lambda) mean', np.mean(td))]
                pop_stats.append({'%s_%s' % (species_index, k): v for k, v in key_value_pairs})

            # get ep_stats from samplers
            ep_metrics = self._concatenate_ep_stats(ray.get([s.get_ep_stats.remote() for s in self.samplers]))

            episodes += ep_metrics['EpisodesThisIter']
            ep_metrics.update({'Episodes Sampled': episodes, 'Training Samples': training_samples,
                               'Sampling time': sampling_time.interval,
                               'Pi optimisation time': np.mean(pi_optimisation_time),
                               'V optimisation time': np.mean(v_optimisation_time),
                               'Samples this iter': samples_this_iter})

            for stats in pop_stats:
                ep_metrics.update(stats)

            print('Epoch: ', epoch)
            with train_summary_writer.as_default():
                for k, v in ep_metrics.items():
                    print(k, v)
                    tf.summary.scalar(k, v, step=episodes)

            if epoch % save_freq == save_freq - 1:
                with open(os.path.join(generation_folder, 'variables.pkl'), 'wb') as f:
                    pickle.dump((self.filters, species_sampler, episodes, training_samples), f)
                print('Saved variables!')
            print('\n' * 2)

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
        for k, v in total_stats.items():
            metrics['Avg_' + k] = np.mean(v)
            if min_and_max:
                metrics['Min_' + k] = np.min(v)
                metrics['Max_' + k] = np.max(v)
            if include_std:
                metrics['Std_' + k] = np.std(v)
        return metrics

    def _load_generation(self, generation):
        generation_folder = './checkpoints/{}/{}'.format(self.env.name, generation)
        pickle_path = os.path.join(generation_folder, 'variables.pkl')
        with open(pickle_path, 'rb') as f:
            filters, species_sampler, episodes, training_samples = pickle.load(f)
        weights = [None]*self.population_size
        for species_index in range(self.population_size):
            species_folder = os.path.join(generation_folder, str(species_index))
            checkpoint_path = tf.train.latest_checkpoint(species_folder)
            self.ac.load_weights(checkpoint_path)
            weights[species_index] = self.ac.get_weights()
        return weights, filters, species_sampler, episodes, training_samples

    def _create_generation(self, generation):
        generation_folder = './checkpoints/{}/{}'.format(self.env.name, generation)
        tensorboard_folder = os.path.join(generation_folder, 'tensorboard', datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        os.makedirs(generation_folder)
        os.makedirs(tensorboard_folder)
        episodes, collected_samples = 0, 0
        weights = []
        if generation == 0:
            weights = [self.ac.get_weights()]*self.population_size
            for species_index in range(self.population_size):
                species_folder = os.path.join(generation_folder, str(species_index))
                os.makedirs(species_folder)
                checkpoint_path = os.path.join(species_folder, str(episodes))
                self.ac.save_weights(checkpoint_path)
        policy_sampler = SpeciesSampler(self.population_size)
        return weights, policy_sampler

    def _value_loss(self, acts_tds, qs):
        # a trick to input actions and td(lambda) through same API
        actions, td_lambda = [tf.squeeze(v) for v in tf.split(acts_tds, 2, axis=-1)]
        q = tf.reduce_sum(tf.one_hot(tf.cast(actions, tf.int32), depth=self.env.action_space.n)*qs, axis=-1)
        return kr.losses.mean_squared_error(td_lambda, q)

    def _surrogate_loss(self, acts_advs_logs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages, old_log_probs = [tf.squeeze(v) for v in tf.split(acts_advs_logs, 3, axis=-1)]
        actions = tf.cast(actions, tf.int32)
        all_log_probs = tf.nn.log_softmax(logits)
        probs = tf.exp(all_log_probs)
        log_probs = tf.reduce_sum(tf.one_hot(actions, depth=self.env.action_space.n) * all_log_probs, axis=-1)
        ratio = tf.exp(log_probs - old_log_probs)
        min_adv = tf.where(advantages > 0, (1 + self.clip_ratio) * advantages, (1 - self.clip_ratio) * advantages)
        entropy = tf.reduce_mean(-tf.reduce_sum(tf.where(probs == 0., tf.zeros_like(probs), probs*all_log_probs), axis=1))
        surrogate_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))
        return surrogate_loss - self.entropy_coeff*entropy

    def _initialise_buffers(self, sizes_dict):
        buffers = {}
        for species_index, size in sizes_dict.items():
            if species_index == 'global':
                obs_shape = list(self.env.critic_observation_shape)
                obs_shape[-1] += 1
                state_action_buf = np.empty((size,) + tuple(obs_shape), dtype=np.float32)
                samples_buf = np.empty(size, dtype=np.int32)
                buffers[species_index] = [state_action_buf, samples_buf]
            else:
                obs_buf = np.empty((size,) + self.env.observation_space.shape, dtype=np.float32)
                act_buf = np.empty((size,) + self.env.action_space.shape, dtype=np.int32)
                adv_buf = np.empty(size, dtype=np.float32)
                td_buf = np.empty(size, dtype=np.float32)
                log_probs_buf = np.empty(size, dtype=np.float32)
                loc_buf = np.empty((size, 2), dtype=np.int32)
                pi_buf = np.empty((size, self.env.action_space.n), dtype=np.float32)
                buffers[species_index] = [obs_buf, act_buf, adv_buf, td_buf, log_probs_buf, loc_buf, pi_buf]
        return buffers

    def _concatenate_samplers_results(self, trajectories_list, ext_species_buffers):
        """

        :param trajectories_list: list of dictionaries {species_index: [obs, act, adv, td, old_log_probs, loc, pi],
                                                      'global': [state_action_per_iter, samples_per_iter]}
        Note: obs dimensions are [n_entities*n_iters, #observation_space.shape]
            and state_action_per_iter dims are [n_iters, state_action.shape]
        :return: concatenated_species_buffer (dict) {species_index: [obs, act, adv, td, old_log_probs, loc, pi,
                                                     state_action_per_iter, samples_per_iter]}
        """
        assert len(trajectories_list) == self.n_workers
        # get sizes for buffers
        sizes_dict = defaultdict(int)
        for species_buffers in trajectories_list:
            for species_index, buffers in species_buffers.items():
                sizes_dict[species_index] += len(buffers[0])
        concatenated_bufs_this_iter = self._initialise_buffers(sizes_dict)

        # concatenate along species/global
        ptr_dict = defaultdict(int)
        for species_buffers in trajectories_list:
            for species_index, buffers in species_buffers.items():
                ptr = ptr_dict[species_index]
                size = len(buffers[0])
                for buf, result in zip(concatenated_bufs_this_iter[species_index], buffers):
                    buf[ptr:ptr + size] = result
                ptr_dict[species_index] += size

        # copy the global variables to each specie index
        for species_index, buffers in concatenated_bufs_this_iter.items():
            if species_index != 'global':
                concatenated_bufs_this_iter[species_index].extend(concatenated_bufs_this_iter['global'])
        del concatenated_bufs_this_iter['global']

        # concatenate with species buffers
        for species_index, buffers in concatenated_bufs_this_iter.items():
            if species_index in ext_species_buffers:
                ext_species_buffers[species_index] = [np.concatenate((old, present), axis=0) for old, present in
                                                      zip(ext_species_buffers[species_index], buffers)]
            else:
                ext_species_buffers[species_index] = buffers
