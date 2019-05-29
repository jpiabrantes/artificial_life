import os
import pickle
from collections import defaultdict

import numpy as np

from utils.misc import agent_name_to_policy_index


def rollout(env, exp_name, policies, species_indices, obs_filter):
    """
    Executes one rollout

    :param env:
    :param exp_name: (string)
    :param policies: (dict) {species_index: DiscreteActor}
    :param obs_filter: (Filter)
    :param species_indices: (list)
    :return: ep_len, population_integral
    """
    dicts, population_integral, ep_len = [], 0, 0
    raw_obs_dict, done_dict = env.reset(species_indices), {'__all__': False}
    while not done_dict['__all__']:
        episode_dict = env.to_dict()
        del raw_obs_dict['state']
        # collect observations for each policy
        species_info = defaultdict(lambda: {'obs': [], 'agents': []})
        for agent_name, raw_obs in raw_obs_dict.items():
            species_index = species_indices.index(agent_name_to_policy_index(agent_name))
            species_info[species_index]['agents'].append(agent_name)
            species_info[species_index]['obs'].append(obs_filter(raw_obs))

        # compute actions for each agent
        action_dict = {}
        for species_index, info in species_info.items():
            observation_arr = np.stack(info['obs'])
            actions, logp, probs = policies[species_index].action_logp_pi(observation_arr)
            for agent_name, action_probs, action in zip(info['agents'], probs, actions):
                action_dict[agent_name] = action
                episode_dict['agents'][agent_name]['action_probs'] = action_probs

        population_integral += len(env.agents)
        # step
        n_raw_obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)

        raw_obs_dict = n_raw_obs_dict
        ep_len += 1
        dicts.append(episode_dict)

    with open(os.path.join('./dicts', '%s.pkl' % exp_name), 'wb') as f:
        pickle.dump(dicts, f)
    return ep_len, population_integral
