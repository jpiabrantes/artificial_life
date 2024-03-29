from collections import namedtuple
import tensorflow as tf
import numpy as np

# from envs.bacteria_colony.bacteria_colony import BacteriaColony
# from envs.bacteria_colony.env_config import env_default_config

from envs.deadly_colony.deadly_colony import DeadlyColony
from envs.deadly_colony.env_config import env_default_config

from models.base import create_vision_and_fc_network, COMAActorCritic, VDNMixer, VDNMixer_2
from replay.rollout import rollout

from algorithms.evolution.helpers import load_variables
from algorithms.ppo.multi_ppo import load_models_and_filters
from algorithms.coma.coma_trainer import load_generation
from algorithms.central_ppo.central_ppo import load_generation as central_load_generation
from algorithms.dqn.p_vdn_train import load as vdn_load
from algorithms.maeq.maeq import load as evdn_load


Weights = namedtuple('Weights', ('main', 'target'))


# env
# env = BacteriaColony(env_default_config)
config = env_default_config.copy()
ep_len = 500
config['max_iters'] = ep_len
config['update_stats'] = True
env = DeadlyColony(config)


# actor
policy_args = {'conv_sizes': [(32, (3, 3), 1)],
               'fc_sizes': [16],
               'last_fc_sizes': [64, 32],
               'conv_input_shape': env.actor_terrain_obs_shape,
               'fc_input_length': np.prod(env.observation_space.shape) - np.prod(env.actor_terrain_obs_shape),
               'num_outputs': env.action_space.n,
               'obs_input_shape': env.observation_space.shape}
policy_creator = lambda: create_vision_and_fc_network(**policy_args)

# coma actor-critic
rows, cols, depth = env.critic_observation_shape
depth += 1  # will give state-actions
critic_args = {'conv_sizes': [(32, (6, 6), (3, 3)), (64, (4, 4), (2, 2)), (64, (3, 3), (1, 1))],
               'fc_sizes': [512],
               'input_shape': (rows, cols, depth),
               'num_outputs': 1}

ac_kwarg = {'actor_args': policy_args, 'critic_args': critic_args, 'observation_space': env.observation_space}
ac_creator = lambda: COMAActorCritic(**ac_kwarg)


exp_name = 'EVDN'
if exp_name == 'EvolutionStrategies':
    last_generation, mu0_list, stds_list, filters = load_variables(env)
    obs_filter = filters['MeanStdFilter']
    species_indices = list(range(len(mu0_list)))
    policies = {}
    for species_index, mu0, stds in zip(species_indices, mu0_list, stds_list):
        actor = policy_creator()
        actor.load_flat_array(mu0)
        policies[species_index] = actor
elif exp_name == 'MultiPPO':
    checkpoint_folder = '../examples/checkpoints/{}/5'
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_folder)
    species_indices = list(range(5))
    policies = [policy_creator() for _ in species_indices]
    filters = load_models_and_filters(policies, [str(i) for i in species_indices], env, only_actors=True)
    obs_filter = filters['MeanStdFilter']
    policies = {i: a for i, a in zip(species_indices, policies)}
elif exp_name == 'COMA':
    ac = ac_creator()
    weights, filters, species_sampler, episodes, training_samples = load_generation(ac_creator(), env, 0, 10)
    species_indices = species_sampler.sample_steps(5).tolist()
    policies = {i: policy_creator() for i in species_indices}
    for species_index, policy in policies.items():
        policy.set_weights(weights[species_index].actor)
    obs_filter = filters['ActorObsFilter']
    print(policies)
    print(species_indices)
elif exp_name == 'CENTRAL_PPO':
    ac = ac_creator()
    weights, filters, species_sampler, episodes, training_samples = central_load_generation(ac_creator(), env, 0, 9)
    species_indices = species_sampler.sample_steps(5).tolist()
    policies = {i: policy_creator() for i in species_indices}
    for species_index, policy in policies.items():
        policy.set_weights(weights[species_index].actor)
    obs_filter = filters['ActorObsFilter']
    print(policies)
    print(species_indices)
elif exp_name == 'EVDN':
    from envs.sexual_colony.sexual_colony import SexualColony
    from envs.sexual_colony.env_config import env_default_config

    env = SexualColony(env_default_config)
    q_kwargs = {'conv_sizes': [(32, (3, 3), 1)],
                'fc_sizes': [16],
                'last_fc_sizes': [64, 32],
                'conv_input_shape': env.actor_terrain_obs_shape,
                'fc_input_length': np.prod(env.observation_space.shape) - np.prod(env.actor_terrain_obs_shape),
                'action_space': env.action_space,
                'obs_input_shape': env.observation_space.shape}
    policies = [VDNMixer_2(**q_kwargs)]

    # q_kwargs = {'hidden_units': [512, 256, 128],
    #             'observation_space': env.observation_space,
    #             'action_space': env.action_space}
    # policies = [VDNMixer(**q_kwargs)]
    filters, weights = evdn_load(env, 'basic')
    obs_filter = filters['ActorObsFilter']
    for policy, (i, w) in zip(policies, weights.items()):
        policy.set_weights(w.main)
    species_indices = list(range(5))
    print(policies)
    print(species_indices)
elif exp_name == 'VDN':
    # q_kwargs = {'conv_sizes': [(32, (3, 3), 1)],
    #             'fc_sizes': [16],
    #             'last_fc_sizes': [64, 32],
    #             'conv_input_shape': env.actor_terrain_obs_shape,
    #             'fc_input_length': np.prod(env.observation_space.shape) - np.prod(env.actor_terrain_obs_shape),
    #             'action_space': env.action_space,
    #             'obs_input_shape': env.observation_space.shape}
    # policies = [VDNMixer_2(**q_kwargs) for _ in range(5)]

    q_kwargs = {'hidden_units': [512, 256, 128],
                'observation_space': env.observation_space,
                'action_space': env.action_space}
    policies = [VDNMixer(**q_kwargs) for _ in range(5)]
    filters, weights = vdn_load(env, 'basic')
    obs_filter = filters['ActorObsFilter']
    for policy, (i, w) in zip(policies, weights.items()):
        policy.set_weights(w.main)
    species_indices = list(range(5))

    print(policies)
    print(species_indices)
elif exp_name == 'Fight':
    last_generation, mu0_list, stds_list, filters = load_variables(env)
    species_indices = list(range(len(mu0_list)))
    policies = {}
    for species_index, mu0, stds in zip(species_indices, mu0_list, stds_list):
        actor = policy_creator()
        actor.load_flat_array(mu0)
        policies[species_index] = actor
    policies = [policies[1], policies[2]]
    obs_filter = {0: filters['MeanStdFilter'], 1: filters['MeanStdFilter']}

    q_kwargs = {'conv_sizes': [(32, (3, 3), 1)],
                'fc_sizes': [16],
                'last_fc_sizes': [64, 32],
                'conv_input_shape': env.actor_terrain_obs_shape,
                'fc_input_length': np.prod(env.observation_space.shape) - np.prod(env.actor_terrain_obs_shape),
                'action_space': env.action_space,
                'obs_input_shape': env.observation_space.shape}
    policies.extend([VDNMixer_2(**q_kwargs) for _ in range(2)])
    filters, weights = vdn_load(env, 'basic')
    weights = {k: weights[k] for k in [1, 3]}
    obs_filter[2] = filters['ActorObsFilter']
    obs_filter[3] = filters['ActorObsFilter']
    for policy, (i, w) in zip(policies[2:], weights.items()):
        policy.set_weights(w.main)
    species_indices = list(range(4))

else:
    print('passing')

# allele_counts_exp = np.zeros((90, 5, ep_len))
for i in range(90):
    print(i)
    ep_len, population_integral = rollout(env, 'VDN_%d' % i, policies, species_indices, obs_filter, save_dict=True)
    # allele_counts_exp[i, :, :] = allele_counts
    print('Episode length: ', ep_len)
    print('Population integral: ', population_integral)
