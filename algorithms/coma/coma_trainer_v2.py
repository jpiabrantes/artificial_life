'''
#TODO: implement distribution strategy
#TODO: document
#TODO: rename agents to entities
'''
from collections import defaultdict
import os
import pickle
from datetime import datetime
from time import time
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import ray

from utils.misc import Timer, SpeciesSampler, SpeciesSamplerManager, Weights
from utils.filters import FilterManager, MeanStdFilter
from utils.coma_helper import get_states_actions_for_locs_and_dna
from algorithms.coma.sampler import Sampler
from algorithms.coma.trainer import Trainer


def get_number_of_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x for x in local_device_protos if x.device_type == 'GPU'])


def load_generation(model, env, generation, population_size):
    # TODO: have a way for only the actor to load the weights
    algorithm_folder = os.path.dirname(os.path.abspath(__file__))
    generation_folder = os.path.join(algorithm_folder, 'checkpoints', env.name, str(generation))
    last_episode = max([int(s.split('_')[0]) for s in os.listdir(generation_folder) if 'variables' in s])
    pickle_path = os.path.join(generation_folder, '%d_variables.pkl' % last_episode)
    with open(pickle_path, 'rb') as f:
        filters, species_sampler, episodes, training_samples = pickle.load(f)
    weights = [None] * population_size
    for species_index in range(population_size):
        species_folder = os.path.join(generation_folder, str(species_index))
        checkpoint_path = tf.train.latest_checkpoint(species_folder)
        model.load_weights(checkpoint_path)
        weights[species_index] = Weights(model.actor.get_weights(), model.critic.get_weights())
    return weights, filters, species_sampler, episodes, training_samples


class MultiAgentCOMATrainer:
    def __init__(self, env_creator, ac_creator, population_size, update_target_freq=30, seed=0, n_workers=1,
                 sample_batch_size=500, batch_size=250, gamma=1., lamb=0.95, clip_ratio=0.2, pi_lr=3e-4, value_lr=1e-3,
                 train_pi_iters=80, train_v_iters=80, target_kl=0.01, save_freq=10, normalise_advantages=False,
                 normalise_observation=False, entropy_coeff=0.01, vf_clip_param=10, n_trainers=1):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.update_target_freq = update_target_freq
        self.n_workers = n_workers
        self.sample_batch_size = sample_batch_size
        self.batch_size = batch_size
        self.save_freq = save_freq
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
        with tf.device('/cpu:0'):
            self.ac = ac_creator()
        if get_number_of_gpus() > 0:
            trainer = ray.remote(num_gpus=1)(Trainer)
        else:
            trainer = ray.remote()(Trainer)
        self.trainers = [trainer.remote(ac_creator, batch_size, normalise_advantages, train_pi_iters, train_v_iters,
                                        value_lr, pi_lr, env_creator, target_kl, clip_ratio, vf_clip_param,
                                        entropy_coeff)
                         for _ in range(n_trainers)]

        self.steps_per_worker = sample_batch_size // n_workers
        if sample_batch_size % n_workers:
            sample_batch_size = n_workers * self.steps_per_worker
            print('WARNING: the sample_batch_size was changed to: %d' % sample_batch_size)
        self.samplers = [Sampler.remote(self.steps_per_worker, gamma, lamb, env_creator, ac_creator, i,
                                        population_size, normalise_observation) for i in range(n_workers)]

    def set_family_reward_coeffs(self, epoch):
        coeff = 1-np.exp(-epoch/250)
        coeff_dict = {k: coeff for k in range(1, self.population_size)}
        coeff_dict[0] = 0.
        for sampler in self.samplers:
            sampler.set_family_reward_coeff.remote(coeff_dict)

    def train(self, epochs, generation, load=False):
        with tf.device('/cpu:0'):
            generation_folder = os.path.join(self.algorithm_folder, 'checkpoints', self.env.name, str(generation))
            tensorboard_folder = os.path.join(generation_folder, 'tensorboard', 'coma_%d' % time())
            if load:
                weights, self.filters, species_sampler, episodes, training_samples = load_generation(self.ac, self.env,
                                                                                                     generation,
                                                                                                     self.population_size)
            else:
                weights, species_sampler, episodes, training_samples = self._create_generation(generation_folder, generation)
            target_weights = [w.critic.copy() for w in weights]
            weights_id_list = [ray.put(w) for w in weights]

            species_trained_epochs = defaultdict(int)
            species_buffers = {}
            train_summary_writer = tf.summary.create_file_writer(tensorboard_folder)
            for epoch in range(epochs):
                if generation > 0:
                    self.set_family_reward_coeffs(epoch)
                total_time = time()
                with Timer() as sampling_time:
                    results_list = ray.get([worker.rollout.remote(weights_id_list) for worker in self.samplers])
                self.filter_manager.synchronize(self.filters, self.samplers)
                self._concatenate_samplers_results(results_list, species_buffers)

                self.species_sampler_manager.synchronize(species_sampler, self.samplers)

                processed_species, training_results = [], []
                samples_this_iter = 0
                i = 0
                for species_index, variables in species_buffers.items():
                    obs, act, adv, ret, old_log_probs, pi, q_tak, states_actions = variables
                    if len(obs) < self.batch_size:
                        continue
                    samples_this_iter += len(obs)
                    training_samples += len(obs)
                    species_trained_epochs[species_index] += 1
                    processed_species.append(species_index)
                    training_results.append(self.trainers[i].train.remote(weights[species_index], variables, species_index))
                    i = (i + 1) % len(self.trainers)

                update_weights_list = ray.get(training_results)
                for species_index, updated_weights in zip(processed_species, update_weights_list):
                    del species_buffers[species_index]
                    if not (species_trained_epochs[species_index] % self.update_target_freq):
                        target_weights[species_index] = updated_weights.critic
                        print('Updated target weights!')
                    weights[species_index] = Weights(actor=updated_weights.actor, critic=updated_weights.critic)
                    weights_id_list[species_index] = ray.put(Weights(weights[species_index].actor,
                                                                     target_weights[species_index]))
                    if (species_trained_epochs[species_index] % self.save_freq) == self.save_freq - 1 or epoch == epochs-1:
                        self.ac.critic.set_weights(updated_weights.critic)
                        self.ac.actor.set_weights(updated_weights.actor)
                        checkpoint_path = os.path.join(generation_folder, str(species_index), str(episodes))
                        self.ac.save_weights(checkpoint_path)
                        print('Weights of species {} saved!'.format(species_index))

                pi_optimisation_time = sum(ray.get([trainer.pi_optimisation_time.remote() for trainer in self.trainers]))
                v_optimisation_time = sum(ray.get([trainer.v_optimisation_time.remote() for trainer in self.trainers]))
                species_stats = ray.get([trainer.species_stats.remote() for trainer in self.trainers])

                # get ep_stats from samplers
                ep_metrics = self._concatenate_ep_stats(ray.get([s.get_ep_stats.remote() for s in self.samplers]))

                episodes += ep_metrics['EpisodesThisIter']
                ep_metrics.update({'Episodes Sampled': episodes, 'Training Samples': training_samples,
                                   'Sampling time': sampling_time.interval,
                                   'Pi optimisation time': np.sum(pi_optimisation_time),
                                   'V optimisation time': np.sum(v_optimisation_time),
                                   'Total time': time()-total_time,
                                   'Samples this iter': samples_this_iter,
                                   'Epoch': epoch})
                if ep_metrics['EpisodesThisIter']:
                    for stats in species_stats:
                        ep_metrics.update(stats)

                print('Epoch: ', epoch)
                with train_summary_writer.as_default():
                    for k, v in ep_metrics.items():
                        print(k, v)
                        tf.summary.scalar(k, v, step=episodes)

                if epoch % self.save_freq == self.save_freq - 1 or epoch == epochs-1:
                    with open(os.path.join(generation_folder, '%d_variables.pkl' % episodes), 'wb') as f:
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

    def _create_generation(self, generation_folder, generation):
        tensorboard_folder = os.path.join(generation_folder, 'tensorboard', datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        os.makedirs(generation_folder, exist_ok=True)
        os.makedirs(tensorboard_folder, exist_ok=True)
        episodes, training_samples = 0, 0

        if generation == 0:
            species_sampler = SpeciesSampler(self.population_size, bias=1)
            weights = [Weights(self.ac.actor.get_weights(), self.ac.critic.get_weights())]*self.population_size
            for species_index in range(self.population_size):
                species_folder = os.path.join(generation_folder, str(species_index))
                os.makedirs(species_folder, exist_ok=True)
                checkpoint_path = os.path.join(species_folder, str(episodes))
                self.ac.save_weights(checkpoint_path)
        else:
            species_sampler = SpeciesSampler(self.population_size, bias=35)
            last_generation = generation - 1
            weights, self.filters, old_species_sampler, episodes, training_samples = load_generation(self.ac, self.env,
                                                                                                     last_generation,
                                                                                                     self.population_size)
            with open(os.path.join(generation_folder, '%d_variables.pkl' % episodes), 'wb') as f:
                pickle.dump((self.filters, species_sampler, episodes, training_samples), f)
            results = sorted([(i, v) for i, v in enumerate(old_species_sampler.rs.mean)], key=lambda tup: tup[1])
            species_indices = [None]*self.population_size
            species_indices[0] = results[-1][0]
            species_indices[1] = results[-1][0]
            species_indices[2] = results[-2][0]
            species_indices[3] = results[-3][0]
            species_indices[4] = results[-4][0]
            species_indices[5:] = old_species_sampler.sample(self.population_size-5)
            new_weights = [None]*self.population_size
            for new_species_index, species_index in enumerate(species_indices):
                species_folder = os.path.join(generation_folder, str(new_species_index))
                os.makedirs(species_folder, exist_ok=True)
                checkpoint_path = os.path.join(species_folder, str(episodes))
                for var, w in zip(self.ac.actor.variables, weights[species_index].actor):
                    var.load(w)
                for var, w in zip(self.ac.critic.variables, weights[species_index].critic):
                    var.load(w)
                self.ac.save_weights(checkpoint_path)
                new_weights[new_species_index] = deepcopy(weights[species_index])
            weights = new_weights

        return weights, species_sampler, episodes, training_samples

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
            q_tak_buf = np.empty(size, dtype=np.float32)
            state_action_buf = np.empty((size,) + tuple(obs_shape), dtype=np.float32)
            buffers[species_index] = [obs_buf, act_buf, adv_buf, td_buf, log_probs_buf, pi_buf, q_tak_buf,
                                      state_action_buf]
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
