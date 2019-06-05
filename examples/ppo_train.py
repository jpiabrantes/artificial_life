import ray
import tensorflow as tf
import numpy as np

from envs.bacteria_colony.bacteria_colony import BacteriaColony
from envs.bacteria_colony.env_config import env_default_config
from models.base import PPOActorCritic
from algorithms.ppo.multi_ppo import MultiAgentPPOTrainer

EAGER = True
if not EAGER:
    tf.compat.v1.disable_eager_execution()

# training session
load = False
generation = 0
epochs = 5000
save_freq = 100

# env
config = env_default_config.copy()
config['greedy_reward'] = True
env_creator = lambda: BacteriaColony(config)
env = env_creator()

# algorithm
gamma = 0.99
lamb = 0.8  # for GAE
seed = 0
sample_batch_size = 260
batch_size = 250
entropy_coeff = 0.01
population_size = 1

# parallelism
n_workers = 1
DEBUG = n_workers == 1
ray.init(local_mode=DEBUG)

# actor critic
ac_kwargs = {'network_args': {'conv_sizes': [(32, (3, 3), 1), (32, (3, 3), 1)],
                              'obs_input_shape': env.observation_space.shape,
                              'fc_sizes': [16],
                              'last_fc_sizes': [32],
                              'conv_input_shape': env.actor_terrain_obs_shape,
                              'fc_input_length': np.prod(env.observation_space.shape) -
                                                 np.prod(env.actor_terrain_obs_shape)},
             'num_actions': env.action_space.n}

ac_creators = {str(i): lambda: PPOActorCritic(**ac_kwargs) for i in range(5)}
ac_mapping_fn = lambda agent_name: agent_name.split('_')[0]

# train
trainer = MultiAgentPPOTrainer(env_creator, ac_creators, ac_mapping_fn, seed=seed, gamma=gamma, lamb=lamb,
                               n_workers=n_workers, batch_size=batch_size, normalise_observation=True,
                               entropy_coeff=entropy_coeff, normalise_advantages=True, save_freq=save_freq)
trainer.train(epochs, load=load)
