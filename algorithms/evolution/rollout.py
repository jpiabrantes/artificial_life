import pickle
from collections import defaultdict

import numpy as np
import scipy

from envs.bacteria_colony.bacteria_colony import BacteriaColony, Terrain
from envs.bacteria_colony.env_config import env_default_config
from algorithms.evolution.policies import DiscreteVisionAndFcPolicy
from algorithms.evolution.helpers import load_variables
from algorithms.evolution.es import CMAES
from utils.filters import apply_filters


env = BacteriaColony(env_default_config)
_, mu0, stds, _, _, filters = load_variables(env)
opts = {'popsize': 5, 'seed': 0}
opts['CMA_stds'] = stds
std0 = 1.
es = CMAES(mu0, sigma0=std0, opts=opts)
solutions = es.ask()
policy_args = {'conv_sizes': [(32, (3, 3), 1), (32, (3, 3), 1)],
               'fc_sizes': [16],
               'last_fc_sizes': [32],
               'conv_input_shape': [env.vision*2 + 1, env.vision*2 + 1, Terrain.size],
               'fc_input_length': (np.prod(env.observation_space.shape) -
                                   np.prod([env.vision*2 + 1, env.vision*2 + 1, Terrain.size])),
               'num_outputs': env.action_space.n}
policies = [DiscreteVisionAndFcPolicy(**policy_args) for _ in range(5)]
for policy, solution in zip(policies, solutions):
    policy.load_flat_array(solution)

# rollout
n_episodes = 1
save_dicts = True
pop_indices = list(range(5))


def agent_name_to_policy_index(agent_name):
    return int(agent_name.split('_')[0])


for episode in range(n_episodes):
    raw_obs_dict, done_dict = env.reset(pop_indices), {'__all__': False}
    ep_len = 0
    dicts = []
    while not done_dict['__all__']:
        if save_dicts and not episode:
            episode_dict = env.to_dict()
        # collect observations for each policy
        policy_info = defaultdict(lambda: {'obs': [], 'agents': []})
        for agent_name, raw_obs in raw_obs_dict.items():
            policy_index = pop_indices.index(agent_name_to_policy_index(agent_name))
            policy_info[policy_index]['agents'].append(agent_name)
            policy_info[policy_index]['obs'].append(apply_filters(raw_obs, filters))

        # compute actions for each agent
        action_dict = {}
        for policy_index, info in policy_info.items():
            observation_arr = np.stack(info['obs'])
            actions = policies[policy_index].get_actions(observation_arr)
            logits = policies[policy_index].predict(observation_arr)
            action_probs = scipy.special.softmax(logits, axis=1)
            info['actions'] = actions
            for agent_name, action, action_p in zip(info['agents'], actions, action_probs):
                if save_dicts and not episode:
                    episode_dict['agents'][agent_name]['action_probs'] = action_p
                action_dict[agent_name] = action

        assert set(action_dict.keys()) == set(raw_obs_dict.keys()), \
            "You should have an action for each agent that received an observation." \
            " {} vs {}".format(action_dict.keys(), raw_obs_dict.keys())

        # step
        n_raw_obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
        # assert set(action_dict.keys()) == set(reward_dict.keys()),\
        #     "You should have a reward for each agent that sent an action." \
        #     " {} vs {}".format(reward_dict.keys(), action_dict.keys())

        assert set(action_dict.keys()).issubset(set(done_dict.keys())), \
            "You should have a done for each agent that sent an action." \
            " {} vs {}".format(done_dict.keys(), action_dict.keys())

        raw_obs_dict = n_raw_obs_dict
        ep_len += 1
        if save_dicts and not episode:
            dicts.append(episode_dict)
    print(ep_len)
    if save_dicts and not episode:
        with open('./dicts.pkl', 'wb') as f:
            pickle.dump(dicts, f)
