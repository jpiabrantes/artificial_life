import os
import pickle
from collections import defaultdict
from time import time

import pandas as pd
import numpy as np

from models.base import VDNMixer
from utils.misc import agent_name_to_species_index_fn

this_folder = os.path.dirname(os.path.abspath(__file__))
columns = ['age', 'health', 'sugar', 'family_size', 'dna']
enemy_columns = ['e_%s' % f for f in columns]


class StatsWriter:
    def __init__(self):
        self.last_data = {}

    def add_stats(self, agents, exp_name, iter_):
        rows = []
        data = {}
        for agent in agents:
            data[agent.id] = [agent.__dict__[key] for key in columns]+[len(agents), iter_]
            if agent.age:
                row = self.last_data[agent.id][:]
                if agent.attacked:
                    e_data = self.last_data[agent.attacked][:-2]
                else:
                    e_data = [0]*len(columns)
                row.extend(e_data)
                rows.append(row)
        self.last_data = data
        if rows:
            df = pd.DataFrame(rows, columns=columns+['population', 'iteration']+enemy_columns)
            path = os.path.join(this_folder, 'data', exp_name+'.csv')
            if os.path.isfile(path):
                df.to_csv(path, mode='a', header=False)
            else:
                df.to_csv(path)

iteration = 0
def rollout(env, exp_name, policies, species_indices, obs_filter, save_dict=True):
    """
    Executes one rollout

    :param env:
    :param exp_name: (string)
    :param policies: (dict) {species_index: DiscreteActor}
    :param obs_filter: (Filter)
    :param species_indices: (list)
    :return: ep_len, population_integral
    """
    global iteration
    stats_writer = StatsWriter()
    dicts, population_integral, ep_len = [], 0, 0
    raw_obs_dict, done_dict = env.reset(species_indices), {'__all__': False}

    while not done_dict['__all__']:
        stats_writer.add_stats(list(env.agents.values()), exp_name, iteration)
        iteration += 1
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
            if type(policies[species_index]) == VDNMixer:
                actions = policies[species_index].get_actions(observation_arr, 0)
                for agent_name, action in zip(info['agents'], actions):
                    action_dict[agent_name] = action
            else:
                actions, logp, probs = policies[species_index].action_logp_pi(observation_arr)
                for agent_name, action_probs, action in zip(info['agents'], probs, actions):
                    action_dict[agent_name] = action
                    episode_dict['agents'][agent_name]['action_probs'] = action_probs

        # competitive stats
        env.competitive_scenario.add_rows(action_dict)

        population_integral += len(env.agents)
        # step
        n_raw_obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)

        raw_obs_dict = n_raw_obs_dict

        ep_len += 1
        dicts.append(episode_dict)
    if save_dict:
        with open(os.path.join('./dicts', '%s.pkl' % exp_name), 'wb') as f:
            pickle.dump(dicts, f)
    env.competitive_scenario.save()
    return ep_len, population_integral
