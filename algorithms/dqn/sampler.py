from collections import defaultdict

import numpy as np
import ray

from utils.buffers import VDNReplayBuffer, EpStats
from utils.misc import agent_name_to_species_index_fn
from utils.filters import MeanStdFilter


@ray.remote(num_cpus=1)
class Sampler:
    def __init__(self, env_creator, num_of_envs, num_of_steps_per_env, qn_creator):
        self.env_creator = env_creator
        self.num_of_steps_per_env = num_of_steps_per_env
        self.envs = [env_creator() for _ in range(num_of_envs)]
        self.max_iters = self.envs[0].max_iters
        [self._randomise_episode_length(env) for env in self.envs]
        self.state = [{'raw_obs_dict': env.reset(self._sample_species_indices()), 'ep_len': 0, 'ep_rew': 0}
                      for env in self.envs]
        # TODO: check random seed
        self.filters = {'ActorObsFilter': MeanStdFilter(shape=self.envs[0].observation_space.shape)}
        self.qn = qn_creator()
        self.ep_stats = EpStats()

    def _sample_species_indices(self):
        return np.random.randint(5, size=5)

    def _randomise_episode_length(self, env):
        noise = int(self.max_iters*0.1)
        env.max_iters = self.max_iters + np.random.randint(-noise, noise + 1)

    def rollout(self, weights, eps_dict=None, training=True):
        qn = self.qn
        species_indices = list(weights.keys())
        species_buffers = {species_index: VDNReplayBuffer() for species_index in species_indices}
        envs = self.envs if training else [self.env_creator()]
        num_of_steps = self.num_of_steps_per_env if training else envs[0].max_iters
        eps_dict = defaultdict(float) if eps_dict is None else eps_dict

        for env_i, env in enumerate(envs):
            state = self.state[env_i]
            if training:
                raw_obs_dict, done_dict, ep_len, ep_rew = state['raw_obs_dict'], {'__all__': False}, state['ep_len'],\
                                                          state['ep_rew']
            else:
                raw_obs_dict, done_dict, ep_len, ep_rew = env.reset(), {'__all__': False}, 0, 0
            for step in range(num_of_steps):
                del raw_obs_dict['state']
                # collect observations for each species
                species_info = defaultdict(lambda: {'obs': [], 'agents': [], 'dna': []})
                for agent_name, raw_obs in raw_obs_dict.items():
                    species_index = agent_name_to_species_index_fn(agent_name)
                    species_info[species_index]['agents'].append(agent_name)
                    filter_ = self.filters['ActorObsFilter']
                    species_info[species_index]['obs'].append(filter_(raw_obs, self.filters))
                    species_info[species_index]['dna'].append(env.agents[agent_name].dna)

                # compute actions of each species
                action_dict = {}
                for species_index, info in species_info.items():
                    qn.set_weights(weights[species_index])
                    eps = eps_dict[species_index]
                    observation_arr = np.stack(info['obs']).astype(np.float32)
                    actions = qn.get_actions(observation_arr, eps)
                    info['actions'] = actions
                    for agent_name, action in zip(info['agents'], actions):
                        action_dict[agent_name] = action

                # step
                n_raw_obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
                # make sure we have a reward and a done for every action sent
                for set_, label in zip((set(reward_dict.keys()), set(done_dict.keys()).difference({'__all__'})),
                                       ('reward', 'done')):
                    assert set(action_dict.keys()) == set_, "You should have a {} for each agent that sent an action." \
                                                            "\n{} vs {}".format(label, set_, action_dict.keys())

                if training:
                    # Save the experience in each species episode buffer
                    step_buffer = defaultdict(lambda: defaultdict(list))
                    for species_index, info in species_info.items():
                        for agent_name, obs, action, dna in zip(*[info[f] for f in ('agents', 'obs', 'actions', 'dna')]):
                            rew, done = reward_dict[agent_name], done_dict[agent_name]
                            n_obs = n_raw_obs_dict.get(agent_name, None)
                            if n_obs is not None:
                                n_obs = self.filters['ActorObsFilter'](n_obs, update=False)
                            else:
                                n_obs = obs * np.nan
                            step_buffer[species_index][dna].append((obs, action, rew, n_obs, done))
                        for dna_step in step_buffer[species_index].values():
                            species_buffers[species_index].add_step(dna_step)

                raw_obs_dict = n_raw_obs_dict
                ep_rew += sum(reward_dict.values())
                ep_len += 1
                if done_dict['__all__']:
                    stats = {'ep_len': ep_len, 'ep_rew': ep_rew}
                    stats.update(info_dict['__all__'])
                    self.ep_stats.add(stats)
                    if training:
                        raw_obs_dict, ep_len, ep_rew = env.reset(self._sample_species_indices()), 0, 0
                        self._randomise_episode_length(env)
                    else:
                        break
            if training:
                self.state[env_i] = {'raw_obs_dict': raw_obs_dict, 'ep_len': ep_len, 'ep_rew': ep_rew}
        if training:
            return species_buffers
        else:
            return self.get_ep_stats()

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
