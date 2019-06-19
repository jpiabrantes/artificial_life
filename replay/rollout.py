import os
import pickle
from collections import defaultdict

import pandas as pd
import numpy as np

from utils.misc import agent_name_to_species_index_fn

this_folder = os.path.dirname(os.path.abspath(__file__))
columns = ['age', 'health', 'sugar', 'family_size', 'attacked', 'kill', 'victim', 'cannibal_attack', 'cannibal_kill',
           'cannibal_victim', 'species_index']


def write_stats(agents, exp_name):
    data = []
    for agent in agents:
        data.append([agent.__dict__[key] for key in columns])
    df = pd.DataFrame(data, columns=columns)
    path = os.path.join(this_folder, 'data', exp_name+'.csv')
    if os.path.isfile(path):
        df.to_csv(path, mode='a', header=False)
    else:
        df.to_csv(path)


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
            species_index = agent_name_to_species_index_fn(agent_name)
            species_info[species_index]['agents'].append(agent_name)
            species_info[species_index]['obs'].append(obs_filter(raw_obs))

        # competitive stats
        agent_names = list(raw_obs_dict.keys())
        np.random.shuffle(agent_names)
        for agent_name in agent_names:
            agent = env.agents[agent_name]
            location = agent.tile.find_best_tile()
            if location is not None:
                env.competitive_scenario.add_best_tile(agent.to_dict(), location)

        # compute actions for each agent
        action_dict = {}
        for species_index, info in species_info.items():
            observation_arr = np.stack(info['obs'])
            actions, logp, probs = policies[species_index].action_logp_pi(observation_arr)
            for agent_name, action_probs, action in zip(info['agents'], probs, actions):
                action_dict[agent_name] = action
                episode_dict['agents'][agent_name]['action_probs'] = action_probs

        # competitive stats
        env.competitive_scenario.add_rows(action_dict)

        population_integral += len(env.agents)
        # step
        n_raw_obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
        write_stats(list(env.agents.values()), exp_name)

        raw_obs_dict = n_raw_obs_dict
        ep_len += 1
        dicts.append(episode_dict)

    with open(os.path.join('./dicts', '%s.pkl' % exp_name), 'wb') as f:
        pickle.dump(dicts, f)
    env.competitive_scenario.save()
    return ep_len, population_integral
