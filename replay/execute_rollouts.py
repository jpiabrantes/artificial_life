import tensorflow as tf
import numpy as np

from envs.bacteria_colony.bacteria_colony import BacteriaColony
from envs.bacteria_colony.env_config import env_default_config
from models.base import create_vision_and_fc_network
from replay.rollout import rollout

from algorithms.evolution.helpers import load_variables
from algorithms.ppo.multi_ppo import load_models_and_filters


# env
env = BacteriaColony(env_default_config)

# actor
policy_args = {'conv_sizes': [(32, (3, 3), 1), (32, (3, 3), 1)],
               'fc_sizes': [16],
               'last_fc_sizes': [32],
               'conv_input_shape': env.actor_terrain_obs_shape,
               'fc_input_length': np.prod(env.observation_space.shape) - np.prod(env.actor_terrain_obs_shape),
               'num_outputs': env.action_space.n,
               'obs_input_shape': env.observation_space.shape}
policy_creator = lambda: create_vision_and_fc_network(**policy_args)

exp_name = 'EvolutionStrategies'

if exp_name == 'EvolutionStrategies':
    last_generation, mu0_list, stds_list, horizons_list, returns_list, filters = load_variables(env)
    obs_filter = filters['MeanStdFilter']
    species_indices = list(range(len(mu0_list)))
    policies = {}
    for species_index, mu0, stds in zip(species_indices, mu0_list, stds_list):
        actor = policy_creator()
        actor.load_flat_array(mu0)
        policies[species_index] = actor
elif exp_name == 'MultiPPO':
    checkpoint_folder = '../examples/checkpoints/{}/1'
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_folder)
    species_indices = list(range(5))
    policies = [policy_creator() for _ in species_indices]
    filters = load_models_and_filters(policies, [str(i) for i in species_indices], env, only_actors=True)
    obs_filter = filters['MeanStdFilter']
    policies = {i: a for i, a in zip(species_indices, policies)}
else:
    raise Exception()

ep_len, population_integral = rollout(env, exp_name, policies, species_indices, obs_filter)
print('Episode length: ', ep_len)
print('Population integral: ', population_integral)
