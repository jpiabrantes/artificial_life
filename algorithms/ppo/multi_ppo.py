'''
#TODO: implement distribution strategy
#TODO: document
'''
from collections import defaultdict
import os
import pickle
from time import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
import ray

from utils.buffers import PPOBuffer, EpStats
from utils.filters import FilterManager, MeanStdFilter, apply_filters
from utils.misc import Timer
from utils.metrics import get_kl_metric, entropy, explained_variance, EarlyStoppingKL


@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, batch_size, gamma, lamb, env_creator, ac_creators, ac_mapping_fn, worker_index,
                 normalise_observation=False):
        self.index = worker_index
        self.batch_size = batch_size
        self.gamma = gamma
        self.lamb = lamb
        self.ac_mapping_fn = ac_mapping_fn
        seed = 10000 * self.index
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.env = env_creator()
        # TODO: make a filter per policy
        if normalise_observation:
            self.filters = {'MeanStdFilter': MeanStdFilter(shape=self.env.observation_space.shape)}
        else:
            self.filters = {}
        self.stats = EpStats()
        self.acs = {ac_name: ac_creator() for ac_name, ac_creator in ac_creators.items()}

    def rollout(self, weights_dict, avg_samples_per_ep):
        # TODO: this seems to add more ops to the graph. find another way to do it
        for ac_name, weights in weights_dict.items():
            self.acs[ac_name].set_weights(weights)
        ac_buffers = defaultdict(list)
        raw_obs_dict, done, ep_ret, ep_len, ac_mapping, agent_buffers = self.env.reset(), False, 0, 0, {}, {}
        collected_samples = 0
        episodes_sampled = 0
        while (episodes_sampled == 0) or (collected_samples < self.batch_size - avg_samples_per_ep):
            while True:
                del raw_obs_dict['state']
                # collect observations for each ac
                ac_info = defaultdict(lambda: {'obs': [], 'agents': []})
                for agent_name, raw_obs in raw_obs_dict.items():
                    if agent_name not in ac_mapping:
                        ac_mapping[agent_name] = self.ac_mapping_fn(agent_name)
                        agent_buffers[agent_name] = PPOBuffer(self.env.longevity, self.env.observation_space,
                                                              self.env.action_space, self.gamma, self.lamb)
                    ac_name = ac_mapping[agent_name]
                    ac_info[ac_name]['agents'].append(agent_name)
                    ac_info[ac_name]['obs'].append(apply_filters(raw_obs, self.filters))
                # compute actions for each agent and save (obs, act, rew, val_est, logp)
                action_dict = {}
                for ac_name, info in ac_info.items():
                    observation_arr = np.stack(info['obs'])
                    actions, values, log_probs = self.acs[ac_name].action_value_logprobs(observation_arr)
                    info['values'] = values
                    info['log_probs'] = log_probs
                    info['actions'] = actions
                    for agent_name, action in zip(info['agents'], actions):
                        action_dict[agent_name] = action
                # step
                n_raw_obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
                collected_samples += len(reward_dict)
                for ac_name, info in ac_info.items():
                    for agent_name, obs, action, value, log_prob in zip(*[info[f] for f in ('agents', 'obs', 'actions',
                                                                                            'values', 'log_probs')]):

                        if agent_name in done_dict:
                            buf, rew = agent_buffers[agent_name], reward_dict[agent_name]
                            buf.store(obs, action, rew, value, log_prob)
                            if done_dict[agent_name] or done_dict['__all__']:
                                buf.finnish_path(0)
                raw_obs_dict = n_raw_obs_dict

                # TODO: compute return per policy
                ep_ret += sum(reward_dict.values())
                ep_len += 1

                if done_dict['__all__']:
                    self._transfer_agent_buffers_to_ac(agent_buffers, ac_buffers, ac_mapping)
                    stats = {'Episode Return': ep_ret, 'Episode Length': ep_len}
                    stats.update(info_dict['__all__'])
                    self.stats.add(stats)
                    raw_obs_dict, done_dict, ep_ret, ep_len, ac_mapping, agent_buffers = (self.env.reset(), False, 0, 0, {},
                                                                                          {})
                    episodes_sampled += 1
                    break
        return ac_buffers

    @staticmethod
    def _transfer_agent_buffers_to_ac(agent_buffers, ac_buffers, ac_mapping):
        for agent_name, buf in agent_buffers.items():
            ac_name = ac_mapping[agent_name]
            if ac_name in ac_buffers:
                ac_buffers[ac_name] = [np.concatenate((present, new), axis=0) for present, new in
                                       zip(ac_buffers[ac_name], buf.get())]
            else:
                ac_buffers[ac_name] = buf.get()

    def get_ep_stats(self):
        return self.stats.get()

    def get_filters(self, flush_after=False):
        """Returns a snapshot of filters.

        Args:
            flush_after (bool): Clears the filter buffer state.

        Returns:
            return_filters (dict): Dict for serializable filters
        """
        return_filters = {}
        for k, f in self.filters.items():
            return_filters[k] = f.as_serializable()
            if flush_after:
                f.clear_buffer()
        return return_filters

    def sync_filters(self, new_filters):
        """Changes self's filter to given and rebases any accumulated delta.

        Args:
            new_filters (dict): Filters with new state to update local copy.
        """
        assert all(k in new_filters for k in self.filters)
        for k in self.filters:
            self.filters[k].sync(new_filters[k])


def load_models_and_filters(models, names, env, only_actors=False):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    checkpoint_folder = os.path.join(this_folder, 'checkpoints', env.name)

    # load filters
    filters_folder = os.path.join(checkpoint_folder, 'filters')
    epoch = max([int(s.split('.')[0]) for s in os.listdir(filters_folder)])
    with open(os.path.join(checkpoint_folder, 'filters', '%d.pkl' % epoch), 'rb') as f:
        filters = pickle.load(f)

    # load models
    for model, name in zip(models, names):
        model_folder = os.path.join(checkpoint_folder, name)
        if only_actors:
            model.load_weights(tf.train.latest_checkpoint(os.path.join(model_folder, 'actor')))
        else:
            model.actor.load_weights(tf.train.latest_checkpoint(os.path.join(model_folder, 'actor')))
            model.critic.load_weights(tf.train.latest_checkpoint(os.path.join(model_folder, 'critic')))
    return filters


class MultiAgentPPOTrainer:
    def __init__(self, env_creator, ac_creators, ac_mapping_fn, ac_kwargs=dict(), seed=0, n_workers=1, batch_size=100,
                 gamma=1., lamb=0.95, clip_ratio=0.2, pi_lr=3e-4, value_lr=1e-3, train_pi_iters=80, train_v_iters=80,
                 target_kl=0.01, save_freq=10, normalise_advantages=False, normalise_observation=False,
                 entropy_coeff=0.01):
        self.name = 'PPO'
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.clip_ratio = clip_ratio
        self.normalize_advantages = normalise_advantages
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.train_v_iters = train_v_iters
        self.train_pi_iters = train_pi_iters
        self.target_kl = target_kl
        self.entropy_coeff = entropy_coeff

        self.filter_manager = FilterManager()

        env = env_creator()
        self.env = env
        if normalise_observation:
            self.filters = {'MeanStdFilter': MeanStdFilter(shape=env.observation_space.shape)}
        else:
            self.filters = {}

        this_folder = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_folder = os.path.join(this_folder, 'checkpoints', self.env.name)
        os.makedirs(os.path.join(self.checkpoint_folder, 'filters'), exist_ok=True)

        # metrics
        kl = get_kl_metric(self.env.action_space.n)
        self.actor_callbacks = [EarlyStoppingKL(self.target_kl)]

        ac_kwargs['num_actions'] = env.action_space.n
        self.acs = {}
        for ac_name, ac_creator in ac_creators.items():
            ac = ac_creator()
            ac.critic.compile(optimizer=kr.optimizers.Adam(learning_rate=value_lr), loss=self._value_loss,
                              metrics=[explained_variance])
            ac.actor.compile(optimizer=kr.optimizers.Adam(learning_rate=pi_lr), loss=self._surrogate_loss,
                             metrics=[kl, entropy])
            self.acs[ac_name] = ac

        self.steps_per_worker = batch_size // n_workers
        if batch_size % n_workers:
            self.batch_size = n_workers * self.steps_per_worker
            print('WARNING: the batch_size was changed to: %d' % self.batch_size)
        self.workers = [Worker.remote(self.steps_per_worker, gamma, lamb, env_creator, ac_creators, ac_mapping_fn, i,
                                      normalise_observation) for i in range(n_workers)]

    def train(self, epochs, load=False):
        if load:
            names, models = zip(*[(ac_name, ac) for ac_name, ac in self.acs.items()])
            self.filters = load_models_and_filters(models, names, self.env)
            print('Loaded models!')
        avg_samples_per_ep = self.env.longevity*self.env.n_agents
        episodes, collected_samples = 0, 0
        train_summary_writer = tf.summary.create_file_writer(os.path.join(self.checkpoint_folder, 'tensorboard',
                                                                          'ppo_%d' % time()))
        with train_summary_writer.as_default():
            for epoch in range(epochs):
                samples_this_iter = 0
                weights_dict = {ac_name: ac.get_weights() for ac_name, ac in self.acs.items()}
                weights_dict_id = ray.put(weights_dict)
                with Timer() as sampling_time:
                    results_list = ray.get([worker.rollout.remote(weights_dict_id, avg_samples_per_ep) for worker in self.workers])
                ac_buffers_dict = self._concatenate_results(results_list)
                self.filter_manager.synchronize(self.filters, self.workers)
                pi_optimisation_time, v_optimisation_time, ac_stats = 0, 0, []
                for ac_name, (obs, act, adv, ret, old_log_probs) in ac_buffers_dict.items():
                    ac = self.acs[ac_name]

                    if self.normalize_advantages:
                        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

                    act_adv_logs = np.concatenate([act[:, None], adv[:, None], old_log_probs[:, None]], axis=-1)
                    with Timer() as pi_optimisation_timer:
                        result = ac.actor.fit(obs, act_adv_logs, batch_size=self.batch_size, shuffle=True,
                                              epochs=self.train_pi_iters, verbose=False, callbacks=self.actor_callbacks)
                        old_policy_loss = result.history['loss'][0]
                        old_entropy = result.history['entropy'][0]
                        kl = result.history['kl'][-1]

                    with Timer() as v_optimisation_timer:
                        result = ac.critic.fit(obs, ret[:, None], batch_size=self.batch_size, epochs=self.train_v_iters,
                                               verbose=False, shuffle=True)
                        old_value_loss = result.history['loss'][0]
                        value_loss = result.history['loss'][-1]
                        old_explained_variance = result.history['explained_variance'][0]
                        explained_variance = result.history['explained_variance'][-1]

                    samples_this_iter += len(obs)
                    pi_optimisation_time += pi_optimisation_timer.interval
                    v_optimisation_time += v_optimisation_timer.interval
                    key_value_pairs = [('LossV', old_value_loss), ('DeltaVLoss', old_value_loss-value_loss),
                                       ('Explained variance', explained_variance),
                                       ('Old explained variance', old_explained_variance),
                                       ('KL', kl), ('Entropy', old_entropy), ('LossPi', old_policy_loss)]
                    ac_stats.append({'%s_%s' % (ac_name, k): v for k, v in key_value_pairs})

                # Metrics
                ep_metrics = ray.get([worker.get_ep_stats.remote() for worker in self.workers])
                ep_metrics = self._concatenate_ep_stats(ep_metrics, min_and_max=True)
                episodes += ep_metrics['EpisodesThisIter']
                collected_samples += samples_this_iter
                avg_samples_per_ep = samples_this_iter/ep_metrics['EpisodesThisIter']
                for stats in ac_stats:
                    ep_metrics.update(stats)
                ep_metrics.update({'Episodes': episodes, 'Collected Samples': collected_samples,
                                   'Sampling time': sampling_time.interval,
                                   'Pi optimisation time': pi_optimisation_time,
                                   'V optimisation time': v_optimisation_time,
                                   'Samples this iter': samples_this_iter})
                print('Epoch ', epoch)
                for k, v in ep_metrics.items():
                    print(k, v)
                print('\n' * 3)

                # Log on Tensorboard
                for k, v in ep_metrics.items():
                    tf.summary.scalar(k, v, step=epoch)
                if (epoch % self.save_freq == 0) or (epoch == epochs - 1):
                    for ac_name, ac in self.acs.items():
                        ac.actor.save_weights(os.path.join(self.checkpoint_folder, ac_name, 'actor', '%d.ckpt' % epoch))
                        ac.critic.save_weights(os.path.join(self.checkpoint_folder, ac_name, 'critic', '%d.ckpt' % epoch))
                        filters_to_save = {k: v.as_serializable() for k, v in self.filters.items()}
                        with open(os.path.join(self.checkpoint_folder, 'filters', '%d.pkl' % epoch), 'wb') as f:
                            pickle.dump(filters_to_save, f)
                        print('Saved model in %s' % self.checkpoint_folder)

    @staticmethod
    def _concatenate_ep_stats(stats_list, min_and_max=False, include_std=False):
        total_stats = None
        for stats in stats_list:
            if total_stats is None:
                total_stats = stats
            else:
                for k, v in stats.items():
                    total_stats[k].extend(v)

        metrics = {}
        for k, v in total_stats.items():
            metrics['Avg' + k] = np.mean(v)
            if min_and_max:
                metrics['Min' + k] = np.min(v)
                metrics['Max' + k] = np.max(v)
            if include_std:
                metrics['Std' + k] = np.std(v)
        metrics['EpisodesThisIter'] = len(v)
        return metrics

    @staticmethod
    def _value_loss(returns, values):
        return kr.losses.mean_squared_error(returns, values)

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
        for ac_name, size in sizes_dict.items():
            obs_buf = np.empty((size,) + self.env.observation_space.shape, dtype=np.float32)
            act_buf = np.empty((size,) + self.env.action_space.shape, dtype=np.float32)
            adv_buf = np.empty(size, dtype=np.float32)
            ret_buf = np.empty(size, dtype=np.float32)
            log_probs_buf = np.empty(size, dtype=np.float32)
            buffers[ac_name] = [obs_buf, act_buf, adv_buf, ret_buf, log_probs_buf]
        return buffers

    def _concatenate_results(self, results):
        assert len(results) == self.n_workers
        # get sizes for buffers
        sizes_dict = defaultdict(int)
        for result_dict in results:
            for ac_name, result_list in result_dict.items():
                sizes_dict[ac_name] += len(result_list[0])
        ac_buffers_dict = self._initialise_buffers(sizes_dict)
        ptr_dict = defaultdict(int)
        for result_dict in results:
            for ac_name, result_list in result_dict.items():
                ptr = ptr_dict[ac_name]
                size = len(result_list[0])
                for buf, result in zip(ac_buffers_dict[ac_name], result_list):
                    buf[ptr:ptr + size] = result
                ptr_dict[ac_name] += size
        return ac_buffers_dict
