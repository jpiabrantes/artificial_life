from collections import defaultdict

import ray
import tensorflow as tf
import numpy as np

from utils.filters import MeanStdFilter, apply_filters


def agent_name_to_policy_index(agent_name):
    return int(agent_name.split('_')[0])


@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, worker_index, env_creator, policy_creators, n_rollouts, normalise_observation=False):
        self.n_rollouts = n_rollouts
        self.index = worker_index
        seed = 10000 * self.index
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.env = env_creator()
        if normalise_observation:
            self.filters = {'MeanStdFilter': MeanStdFilter(shape=self.env.observation_space.shape)}
        else:
            self.filters = {}
        self.policies = [p_creator() for p_creator in policy_creators]

    def rollout(self, weights_ids, pop_indices):
        for policy, weight in zip(self.policies, ray.get(weights_ids)):
            policy.load_flat_array(weight)
        mean_fitness = defaultdict(int)
        mean_len = 0
        for n_rollout in range(self.n_rollouts):
            raw_obs_dict, done_dict = self.env.reset(pop_indices), {'__all__': False}
            ep_len = 0
            while not done_dict['__all__']:
                del raw_obs_dict['state']
                # collect observations for each policy
                policy_info = defaultdict(lambda: {'obs': [], 'agents': []})
                for agent_name, raw_obs in raw_obs_dict.items():
                    policy_index = pop_indices.index(agent_name_to_policy_index(agent_name))
                    policy_info[policy_index]['agents'].append(agent_name)
                    policy_info[policy_index]['obs'].append(apply_filters(raw_obs, self.filters))

                # compute actions for each agent
                action_dict = {}
                for policy_index, info in policy_info.items():
                    observation_arr = np.stack(info['obs'])
                    actions = self.policies[policy_index].get_actions(observation_arr)
                    info['actions'] = actions
                    for agent_name, action in zip(info['agents'], actions):
                        action_dict[agent_name] = action

                assert set(action_dict.keys()) == set(raw_obs_dict.keys()),\
                    "You should have an action for each agent that received an observation." \
                    " {} vs {}".format(action_dict.keys(), raw_obs_dict.keys())

                # step
                n_raw_obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
                # assert set(action_dict.keys()) == set(reward_dict.keys()),\
                #     "You should have a reward for each agent that sent an action." \
                #     " {} vs {}".format(reward_dict.keys(), action_dict.keys())

                assert set(action_dict.keys()).issubset(set(done_dict.keys())),\
                    "You should have a done for each agent that sent an action." \
                    " {} vs {}".format(done_dict.keys(), action_dict.keys())

                raw_obs_dict = n_raw_obs_dict
                ep_len += 1
            for score, pop_index in zip(info_dict['founders_total_results'], pop_indices):
                mean_fitness[pop_index] += score * 1/self.n_rollouts
            mean_len += ep_len * 1/self.n_rollouts
        return mean_fitness, mean_len

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
