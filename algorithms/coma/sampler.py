"""
# TODO: create docstrings
"""
from collections import defaultdict

import ray
import numpy as np
import tensorflow as tf

from utils.filters import MeanStdFilter
from utils.buffers import EntityBuffer, EpStats
from utils.coma_helper import get_states_actions_for_locs
from utils.misc import SpeciesSampler


def agent_name_to_species_index_fn(agent_name):
    return int(agent_name.split('_')[0])


@ray.remote(num_cpus=1)
class Sampler:
    def __init__(self, sample_batch_size, gamma, lamb, env_creator, ac_creator, worker_index, population_size,
                 normalise_observation=False):
        self.index = worker_index
        self.sample_batch_size = sample_batch_size
        self.gamma = gamma
        self.lamb = lamb

        seed = 10000 * self.index
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.env = env_creator()
        self.n_acs = self.env.n_agents
        self.acs = {'ac_%d' % i: ac_creator() for i in range(self.n_acs)}

        if normalise_observation:
            self.filters = {'ActorObsFilter': MeanStdFilter(shape=self.env.observation_space.shape),
                            'CriticObsFilter': MeanStdFilter(shape=self.env.critic_observation_shape)}
        else:
            self.filters = {}

        self.species_sampler = SpeciesSampler(population_size)
        self.in_an_episode = False
        self.state = None
        self.ep_stats = EpStats()

    def rollout(self, weight_id_list):
        """
        n_raw_obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
        action_dict.agents == reward_dict.agents == done_dict.agents

        """
        collected_samples = 0
        episodes_sampled = 0
        species_buffers = {}

        if self.in_an_episode:  # restore state
            species_indices, raw_obs_dict, ep_ret, ep_len = [self.state[s] for s in
                                                             ('species_indices', 'raw_obs_dict', 'ep_ret', 'ep_len')]
            species_ac_map = self._load_species(species_indices, weight_id_list)
            n_active_agents = len(raw_obs_dict)
        else:  # new episode
            species_indices = self.species_sampler.sample(self.n_acs)
            species_ac_map = self._load_species(species_indices, weight_id_list)
            raw_obs_dict, done_dict, ep_ret, ep_len, = self.env.reset(species_indices), {'__all__': False}, 0, 0
            n_active_agents = len(species_indices)

        states_actions_buf, samples_buf = [], []
        agent_name_ac, agent_buffers, done_dict = {}, {}, {'__all__': False}
        while collected_samples < self.sample_batch_size - n_active_agents:
            if not self.in_an_episode:
                self.in_an_episode = True
            global_raw_obs = raw_obs_dict['state']
            global_obs = self.filters['CriticObsFilter'](global_raw_obs)
            del raw_obs_dict['state']

            # collect and filter observations for each ac
            ac_info = self._collect_and_filter_obs(raw_obs_dict, agent_name_ac, agent_buffers, species_ac_map)

            # compute actions for each agent and build global_actions
            action_dict, global_actions = self._get_action_dict_and_global_action(ac_info, raw_obs_dict)

            # compute state-action value and advantage
            # input: log_prob, action, global-action-state
            state_action = np.concatenate((global_obs, global_actions[..., None]), axis=-1)
            val_map = self._compute_vals_and_advs(ac_info, state_action)

            # step
            n_raw_obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)

            # make sure we have a reward and a done for every action sent
            for set_, label in zip((set(reward_dict.keys()), set(done_dict.keys()).difference({'__all__'})),
                                   ('reward', 'done')):
                assert set(action_dict.keys()) == set_, "You should have a {} for each agent that sent an action." \
                                                        "\n{} vs {}".format(label, set_, action_dict.keys())

            raw_obs_dict = n_raw_obs_dict
            samples_buf.append(len(reward_dict))
            collected_samples += len(reward_dict)
            ep_len += 1
            n_active_agents = len(raw_obs_dict)
            done_sampling = not (collected_samples < self.sample_batch_size - n_active_agents)

            # store relevant info in the entities buffers
            for ac_name, info in ac_info.items():
                for agent_name, obs, action, val, log_prob, adv, pi, loc in \
                        zip(*[info[f] for f in ('agents', 'obs', 'actions', 'vals', 'log_probs', 'advs', 'pis',
                                                'locs')]):
                    if agent_name in done_dict:
                        buf, rew = agent_buffers[agent_name], reward_dict[agent_name]
                        ep_ret += rew
                        buf.store(obs, action, rew, val, adv, log_prob, pi, loc)
                        if done_dict[agent_name]:  # if entity died
                            kinship_map = self.env.get_kinship_map(agent_name)
                            if np.any(kinship_map > 0):
                                last_value = 1/np.sum(kinship_map)*(kinship_map*val_map).sum()
                            else:
                                last_value = 0
                            buf.finnish_path(last_value)
                        # if entity is alive but episode is over or we're done sampling
                        elif done_dict['__all__'] or done_sampling:
                            buf.finnish_path(val)

            states_actions_buf.append(state_action)
            if done_dict['__all__']:
                species_results = defaultdict(list)
                for species_index, result in zip(species_indices, info_dict['founders_results']):
                    species_results[species_index].append(result)
                stats = {'ep_len': ep_len, 'ep_ret': ep_ret}
                stats.update(info_dict['__all__'])
                stats.update({'%s_survivers' % i: np.mean(r) for i, r in species_results.items()})
                self.ep_stats.add(stats)
                episodes_sampled += 1
                self.in_an_episode = False
                self._collect_entity_buffers_into_species_buffers(agent_buffers, species_buffers)
                if not done_sampling:
                    self.species_sampler.show_results(species_indices, info_dict['founders_results'])
                    species_indices = self.species_sampler.sample(self.n_acs)
                    species_ac_map = self._load_species(species_indices, weight_id_list)
                    raw_obs_dict, done_dict, ep_ret, ep_len, = self.env.reset(species_indices), {'__all__': False}, 0, 0
                    agent_name_ac, agent_buffers = {}, {}
                    n_active_agents = len(species_indices)

        if self.in_an_episode:
            # We stopped in the middle of an episode! Let's store the state so that we can carry on later.
            keys = ('species_indices', 'raw_obs_dict', 'ep_ret', 'ep_len')
            values = (species_indices, raw_obs_dict, ep_ret, ep_len)
            self.state = {k: v for k, v in zip(keys, values)}

        self._collect_entity_buffers_into_species_buffers(agent_buffers, species_buffers)
        species_buffers['global'] = [states_actions_buf, samples_buf]
        return species_buffers

    def _load_species(self, species_indices, weight_id_list):
        species_ac_map = {}
        ac_names = list(self.acs.keys())
        j = 0
        for species_index in species_indices:
            if species_index not in species_ac_map:
                ac_name = ac_names[j]
                species_ac_map[species_index] = ac_name
                # TODO: allow to have shared weights and non-shared weights
                self.acs[ac_name].set_weights(ray.get(weight_id_list[species_index]))
                j += 1
        return species_ac_map

    def _collect_and_filter_obs(self, raw_obs_dict, agent_name_ac, agent_buffers, species_ac_map):
        ac_info = defaultdict(lambda: {'obs': [], 'agents': [], 'locs': []})
        for agent_name, raw_obs in raw_obs_dict.items():
            if agent_name not in agent_name_ac:
                species_index = agent_name_to_species_index_fn(agent_name)
                agent_name_ac[agent_name] = species_ac_map[species_index]
                agent_buffers[agent_name] = EntityBuffer(self.env.longevity, self.env.observation_space,
                                                         self.env.action_space, self.gamma, self.lamb)
            ac_name = agent_name_ac[agent_name]
            agent = self.env.agents[agent_name]
            ac_info[ac_name]['locs'].append((agent.row, agent.col))
            ac_info[ac_name]['agents'].append(agent_name)
            ac_info[ac_name]['obs'].append(self.filters['ActorObsFilter'](raw_obs))
        return ac_info

    def _get_action_dict_and_global_action(self, ac_info, raw_obs_dict):
        global_action = np.zeros((self.env.n_rows, self.env.n_cols), np.int32)
        action_dict = {}
        for ac_name, info in ac_info.items():
            observation_arr = np.stack(info['obs'])
            actions, log_probs, pi = self.acs[ac_name].action_logp_pi(observation_arr)
            info['log_probs'] = log_probs
            info['actions'] = actions
            info['pis'] = pi
            for agent_name, action, (row, col) in zip(info['agents'], actions, info['locs']):
                action_dict[agent_name] = action
                global_action[row, col] = action

        assert set(action_dict.keys()) == set(raw_obs_dict.keys()), \
            "You should have an action for each agent that received an observation." \
            " {} vs {}".format(action_dict.keys(), raw_obs_dict.keys())
        return action_dict, global_action

    def _compute_vals_and_advs(self, ac_info, state_action):
        val_map = np.zeros((self.env.n_rows, self.env.n_cols), np.float32)
        for ac_name, info in ac_info.items():
            states_actions = get_states_actions_for_locs(state_action, info['locs'], self.env.n_rows, self.env.n_cols)
            vals, advs = self.acs[ac_name].val_and_adv(states_actions, info['actions'], info['pis'])
            info['vals'], info['advs'] = vals, advs
            for val, (row, col) in zip(vals, info['locs']):
                val_map[row, col] = val
        return val_map

    @staticmethod
    def _collect_entity_buffers_into_species_buffers(agent_buffers, species_buffers):
        for agent_name, buf in agent_buffers.items():
            species_index = agent_name_to_species_index_fn(agent_name)
            if species_index in species_buffers:
                species_buffers[species_index] = [np.concatenate((present, new), axis=0) for present, new in
                                                  zip(species_buffers[species_index], buf.get())]
            else:
                species_buffers[species_index] = buf.get()

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

    def get_species_sampler(self):
        return self.species_sampler

    def sync_species_sampler(self, new_species_sampler):
        self.species_sampler.sync(new_species_sampler)

    def get_ep_stats(self):
        return self.ep_stats.get()
