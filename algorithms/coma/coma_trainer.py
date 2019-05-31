'''
#TODO: implement distribution strategy
#TODO: document
#TODO: rename agents to entities
'''
from collections import defaultdict, namedtuple
import os
import pickle
from datetime import datetime
from time import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
import ray

from utils.misc import Timer, SpeciesSampler, SpeciesSamplerManager
from utils.filters import FilterManager, MeanStdFilter
from utils.coma_helper import get_states_actions_for_locs_and_dna
from utils.metrics import get_kl_metric, entropy, get_coma_explained_variance, EarlyStoppingKL
from algorithms.coma.sampler import Sampler


Weights = namedtuple('Weights', ('actor', 'critic'))


class MultiAgentCOMATrainer:
    def __init__(self, env_creator, ac_creator, population_size, update_target_freq=30, seed=0, n_workers=1,
                 sample_batch_size=500, batch_size=250, gamma=1., lamb=0.95, clip_ratio=0.2, pi_lr=3e-4, value_lr=1e-3,
                 train_pi_iters=80, train_v_iters=80, target_kl=0.01, save_freq=10, normalise_advantages=False,
                 normalise_observation=False, entropy_coeff=0.01):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.update_target_freq = update_target_freq
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

        self.algorithm_folder = os.path.dirname(os.path.abspath(__file__))
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

        kl = get_kl_metric(self.env.action_space.n)
        r2score = get_coma_explained_variance(self.env.action_space.n)
        self.actor_callbacks = [EarlyStoppingKL(self.target_kl)]
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            self.ac = ac_creator()
            self.ac.critic.compile(optimizer=kr.optimizers.Adam(learning_rate=value_lr), loss=self._value_loss,
                                   metrics=[r2score])
            self.ac.actor.compile(optimizer=kr.optimizers.Adam(learning_rate=pi_lr), loss=self._surrogate_loss,
                                  metrics=[kl, entropy])

        self.steps_per_worker = sample_batch_size // n_workers
        if sample_batch_size % n_workers:
            sample_batch_size = n_workers * self.steps_per_worker
            print('WARNING: the sample_batch_size was changed to: %d' % sample_batch_size)
        self.samplers = [Sampler.remote(self.steps_per_worker, gamma, lamb, env_creator, ac_creator, i,
                                        population_size, normalise_observation) for i in range(n_workers)]

    def train(self, epochs, generation, load=False):
        generation_folder = os.path.join(self.algorithm_folder, 'checkpoints', self.env.name, str(generation))
        tensorboard_folder = os.path.join(generation_folder, 'tensorboard', 'coma_%d' % time())
        if load:
            weights, self.filters, species_sampler, episodes, training_samples = self._load_generation(generation_folder)
        else:
            weights, species_sampler = self._create_generation(generation_folder, generation)
            episodes, training_samples = 0, 0
        target_weights = [w.critic.copy() for w in weights]
        weights_id_list = [ray.put(w) for w in weights]

        species_trained_epochs = defaultdict(int)
        species_buffers = {}
        train_summary_writer = tf.summary.create_file_writer(tensorboard_folder)
        for epoch in range(epochs):
            samples_this_iter = 0
            with Timer() as sampling_time:
                results_list = ray.get([worker.rollout.remote(weights_id_list) for worker in self.samplers])
            self.filter_manager.synchronize(self.filters, self.samplers)
            self._concatenate_samplers_results(results_list, species_buffers)

            self.species_sampler_manager.synchronize(species_sampler, self.samplers)
            pi_optimisation_time, v_optimisation_time, pop_stats = [], [], []
            processed_species = []
            for species_index, variables in species_buffers.items():
                obs, act, adv, td, old_log_probs, pi, states_actions = variables
                if len(obs) < self.batch_size:
                    continue
                for var, w in zip(self.ac.actor.variables, weights[species_index].actor):
                    var.load(w)
                for var, w in zip(self.ac.critic.variables, weights[species_index].critic):
                    var.load(w)
                processed_species.append(species_index)

                if self.normalize_advantages:
                    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
                act_adv_logs = np.concatenate([act[:, None], adv[:, None], old_log_probs[:, None]], axis=-1)
                with Timer() as pi_optimisation_timer:
                    result = self.ac.actor.fit(obs, act_adv_logs, batch_size=self.batch_size, shuffle=True,
                                               epochs=self.train_pi_iters, verbose=False,
                                               callbacks=self.actor_callbacks)
                    old_policy_loss = result.history['loss'][0]
                    old_entropy = result.history['entropy'][0]
                    kl = result.history['kl'][-1]

                act_td = np.concatenate([act[:, None], td[:, None]], axis=-1)
                with Timer() as v_optimisation_timer:
                    result = self.ac.critic.fit(states_actions, act_td, shuffle=True, batch_size=self.batch_size,
                                                verbose=False, epochs=self.train_v_iters)
                    old_value_loss = result.history['loss'][0]
                    value_loss = result.history['loss'][-1]
                    old_explained_variance = result.history['explained_variance'][0]
                    explained_variance = result.history['explained_variance'][-1]

                species_trained_epochs[species_index] += 1
                if not (species_trained_epochs[species_index] % self.update_target_freq):
                    target_weights[species_index] = self.ac.critic.get_weights()
                    print('Updated target weights!')
                weights[species_index] = Weights(actor=self.ac.actor.get_weights(), critic=self.ac.critic.get_weights())
                weights_id_list[species_index] = ray.put(Weights(weights[species_index].actor,
                                                                 target_weights[species_index]))
                if (species_trained_epochs[species_index] % self.save_freq) == self.save_freq - 1 or epoch == epochs-1:
                    checkpoint_path = os.path.join(generation_folder, str(species_index), str(episodes))
                    self.ac.save_weights(checkpoint_path)
                    print('Weights of species {} saved!'.format(species_index))

                samples_this_iter += len(obs)
                training_samples += len(obs)
                pi_optimisation_time += [pi_optimisation_timer.interval]
                v_optimisation_time += [v_optimisation_timer.interval]
                key_value_pairs = [('LossQ', old_value_loss), ('deltaQLoss', old_value_loss-value_loss),
                                   ('Old Explained Variance', old_explained_variance),
                                   ('Explained variance', explained_variance),
                                   ('KL', kl), ('Old entropy', old_entropy), ('LossPi', old_policy_loss),
                                   ('TD(lambda)', np.mean(td))]
                pop_stats.append({'%s_%s' % (species_index, k): v for k, v in key_value_pairs})

            for species_index in processed_species:
                del species_buffers[species_index]

            # get ep_stats from samplers
            ep_metrics = self._concatenate_ep_stats(ray.get([s.get_ep_stats.remote() for s in self.samplers]))

            episodes += ep_metrics['EpisodesThisIter']
            ep_metrics.update({'Episodes Sampled': episodes, 'Training Samples': training_samples,
                               'Sampling time': sampling_time.interval,
                               'Pi optimisation time': np.sum(pi_optimisation_time),
                               'V optimisation time': np.sum(v_optimisation_time),
                               'Samples this iter': samples_this_iter})
            if ep_metrics['EpisodesThisIter']:
                for stats in pop_stats:
                    ep_metrics.update(stats)

            print('Epoch: ', epoch)
            with train_summary_writer.as_default():
                for k, v in ep_metrics.items():
                    print(k, v)
                    tf.summary.scalar(k, v, step=episodes)

            if epoch % self.save_freq == self.save_freq - 1 or epoch == epochs-1:
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
        if metrics['EpisodesThisIter']:
            for k, v in total_stats.items():
                metrics['Avg_' + k] = np.mean(v)
                if min_and_max:
                    metrics['Min_' + k] = np.min(v)
                    metrics['Max_' + k] = np.max(v)
                if include_std:
                    metrics['Std_' + k] = np.std(v)
        return metrics

    def _load_generation(self, generation_folder):
        pickle_path = os.path.join(generation_folder, 'variables.pkl')
        with open(pickle_path, 'rb') as f:
            filters, species_sampler, episodes, training_samples = pickle.load(f)
        weights = [None]*self.population_size
        for species_index in range(self.population_size):
            species_folder = os.path.join(generation_folder, str(species_index))
            checkpoint_path = tf.train.latest_checkpoint(species_folder)
            self.ac.load_weights(checkpoint_path)
            weights[species_index] = Weights(self.ac.actor.get_weights(), self.ac.critic.get_weights())
        return weights, filters, species_sampler, episodes, training_samples

    def _create_generation(self, generation_folder, generation):
        tensorboard_folder = os.path.join(generation_folder, 'tensorboard', datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        os.makedirs(generation_folder, exist_ok=True)
        os.makedirs(tensorboard_folder, exist_ok=True)
        episodes, collected_samples = 0, 0
        weights = []
        if generation == 0:
            weights = [Weights(self.ac.actor.get_weights(), self.ac.critic.get_weights())]*self.population_size
            for species_index in range(self.population_size):
                species_folder = os.path.join(generation_folder, str(species_index))
                os.makedirs(species_folder, exist_ok=True)
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
        entropy = tf.reduce_mean(-tf.reduce_sum(tf.where(probs == 0., tf.zeros_like(probs), probs*all_log_probs),
                                                axis=1))
        surrogate_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))
        return surrogate_loss - self.entropy_coeff*entropy

    def _initialise_buffers(self, sizes_dict):
        buffers = {}
        obs_shape = list(self.env.critic_observation_shape)
        obs_shape[-1] += 1
        for species_index, size in sizes_dict.items():
            obs_buf = np.empty((size,) + self.env.observation_space.shape, dtype=np.float32)
            act_buf = np.empty((size,) + self.env.action_space.shape, dtype=np.int32)
            adv_buf = np.empty(size, dtype=np.float32)
            td_buf = np.empty(size, dtype=np.float32)
            log_probs_buf = np.empty(size, dtype=np.float32)
            pi_buf = np.empty((size, self.env.action_space.n), dtype=np.float32)
            state_action_buf = np.empty((size,) + tuple(obs_shape), dtype=np.float32)
            buffers[species_index] = [obs_buf, act_buf, adv_buf, td_buf, log_probs_buf, pi_buf, state_action_buf]
        return buffers

    def _concatenate_samplers_results(self, results_list, ext_species_buffers):
        """

        :param results_list: [(species_buffers_dict, global_dict), (species_buffers_dict, global_dict)]

        global_buffer: {(worker.index, iter.index): [state_action, number of samples]}
        """
        assert len(results_list) == self.n_workers

        species_dict_list, global_buffers_list = zip(*results_list)

        # get sizes for buffers
        sizes_dict = defaultdict(int)
        for species_dict in species_dict_list:
            for species_index, buffers in species_dict.items():
                sizes_dict[species_index] += len(buffers[0])
        concatenated_bufs_this_iter = self._initialise_buffers(sizes_dict)

        # concatenate along species
        ptr_dict = defaultdict(int)
        for species_dict, global_buffer in zip(species_dict_list, global_buffers_list):
            for species_index, buffers in species_dict.items():
                ptr = ptr_dict[species_index]
                size = len(buffers[0])
                buf_indices = np.arange(ptr, ptr+size)
                state_action_buf = concatenated_bufs_this_iter[species_index][-1]

                # remove locs, inds and dnas from the buffers and
                # pre_process state_actions
                buffers, (locs, dnas, inds) = buffers[:-3], buffers[-3:]
                unique_inds = np.unique(inds)
                for ind in unique_inds:
                    mask = inds == ind
                    buf_indices_to_fill = buf_indices[mask]
                    g_raw_state_action, n_samples = global_buffer[ind]
                    # assert np.sum(mask) == n_samples, "n_samples doesnt match samples received"
                    raw_states_actions = get_states_actions_for_locs_and_dna(g_raw_state_action, locs[mask], dnas[mask],
                                                                             self.env.n_rows, self.env.n_cols,
                                                                             self.env.State.DNA)
                    for buf_index, raw_state_action in zip(buf_indices_to_fill, raw_states_actions):
                        state_action_buf[buf_index][..., :-1] = self.filters['CriticObsFilter']\
                            (raw_state_action[..., :-1], update=False)
                        state_action_buf[buf_index][..., -1] = raw_state_action[..., -1]
                for buf, result in zip(concatenated_bufs_this_iter[species_index][:-1], buffers):
                    buf[ptr:ptr + size] = result
                ptr_dict[species_index] += size

        # concatenate with species buffers
        for species_index, buffers in concatenated_bufs_this_iter.items():
            if species_index in ext_species_buffers:
                ext_species_buffers[species_index] = [np.concatenate((old, present), axis=0) for old, present in
                                                      zip(ext_species_buffers[species_index], buffers)]
            else:
                ext_species_buffers[species_index] = buffers
