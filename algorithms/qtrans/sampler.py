from collections import defaultdict

import numpy as np
import ray

from utils.buffers import QTranReplayBuffer, EpStats
from utils.misc import agent_name_to_species_index_fn
from utils.filters import MeanStdFilter
from models.base import Qtran


@ray.remote(num_cpus=1)
class Sampler:
    def __init__(self, env_creator, num_of_envs, num_of_steps_per_env, brain_kwargs):
        self.num_of_steps_per_env = num_of_steps_per_env
        self.envs = [env_creator() for _ in range(num_of_envs)]
        self.state = [{'raw_obs_dict': env.reset(), 'ep_len': 0, 'ep_rew': 0} for env in self.envs]
        # TODO: check random seed
        self.filters = {'ActorObsFilter': MeanStdFilter(shape=self.envs[0].observation_space.shape)}
        self.qn = Qtran(**brain_kwargs)
        self.ep_stats = EpStats()

    def rollout(self, weights, eps_dict):
        qn = self.qn
        species_indices = list(weights.keys())
        species_buffers = {species_index: QTranReplayBuffer() for species_index in species_indices}
        for env_i, env in enumerate(self.envs):
            state = self.state[env_i]
            raw_obs_dict, done_dict, ep_len, ep_rew = state['raw_obs_dict'], {'__all__': False}, state['ep_len'],\
                                                      state['ep_rew']

            species_info = None
            for step in range(self.num_of_steps_per_env):
                state_action_species = np.ones((env.n_rows, env.n_cols, len(env.State) + 2), np.float32) * -1
                state_action_species[:, :, :-2] = raw_obs_dict['state']

                # collect observations for each species
                if species_info is None:
                    species_info = self._collect_observations_for_each_species(raw_obs_dict)

                # compute actions of each species
                action_dict = {}
                for species_index, info in species_info.items():
                    qn.set_weights(weights[species_index])
                    eps = eps_dict[species_index]
                    observation_arr = np.stack(info['obs'])
                    actions = qn.get_actions(observation_arr, eps)
                    info['actions'] = actions
                    for agent_name, action in zip(info['agents'], actions):
                        action_dict[agent_name] = action
                        agent = env.agents[agent_name]
                        info['locs'].append((agent.row, agent.col))

                # step
                n_raw_obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
                # make sure we have a reward and a done for every action sent
                for set_, label in zip((set(reward_dict.keys()), set(done_dict.keys()).difference({'__all__'})),
                                       ('reward', 'done')):
                    assert set(action_dict.keys()) == set_, "You should have a {} for each agent that sent an action." \
                                                            "\n{} vs {}".format(label, set_, action_dict.keys())

                # Save the experience in each species episode buffer
                species_indices = list(species_info.keys())
                step_buffer = defaultdict(list)
                for species_index, info in species_info.items():
                    for agent_name, obs, action in zip(info['agents'], info['obs'], info['actions']):
                        rew, done = reward_dict[agent_name], done_dict[agent_name]
                        step_buffer[species_index].append((obs, action, rew, done))

                # compute n_state_action_star
                n_state_action = np.ones((env.n_rows, env.n_cols, len(env.State) + 1), np.float32) * -1
                n_state_action[:, :, :-1] = n_raw_obs_dict['state']
                # collect n_observations for each species
                species_info = self._collect_observations_for_each_species(n_raw_obs_dict)
                # compute n_actions_star of each species
                for species_index, info in species_info.items():
                    qn.set_weights(weights[species_index])
                    observation_arr = np.stack(info['obs'])
                    actions = qn.get_actions(observation_arr, 0)
                    for agent_name, action in zip(info['agents'], actions):
                        agent = env.agents[agent_name]
                        row, col = agent.row, agent.col
                        n_state_action[row, col, -1] = action
                for species_index in species_indices:
                    n_sta_act = n_state_action.copy()
                    n_sta_act[:, :, env.State.DNA] = n_sta_act[:, :, env.State.DNA] == species_index
                    species_buffers[species_index].add_step(step_buffer[species_index], state_action_species,
                                                            species_info[species_index]['locs'], n_sta_act)

                raw_obs_dict = n_raw_obs_dict
                ep_rew += sum(reward_dict.values())
                ep_len += 1
                if done_dict['__all__']:
                    stats = {'ep_len': ep_len, 'ep_rew': ep_rew}
                    stats.update(info_dict['__all__'])
                    self.ep_stats.add(stats)
                    raw_obs_dict, ep_len, ep_rew = env.reset(), 0, 0

            self.state[env_i] = {'raw_obs_dict': raw_obs_dict, 'ep_len': ep_len, 'ep_rew': ep_rew}
        return species_buffers

    def _collect_observations_for_each_species(self, raw_obs_dict):
        species_info = defaultdict(lambda: {'obs': [], 'agents': [], 'locs': []})
        for agent_name, raw_obs in raw_obs_dict.items():
            if agent_name != 'state':
                species_index = agent_name_to_species_index_fn(agent_name)
                species_info[species_index]['agents'].append(agent_name)
                filter_ = self.filters['ActorObsFilter']
                species_info[species_index]['obs'].append(filter_(raw_obs))
        return species_info

    def get_ep_stats(self):
        return self.ep_stats.get()

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
