import os
import ray
import tensorflow as tf
import numpy as np

from envs.bacteria_colony.bacteria_colony import BacteriaColony
from envs.bacteria_colony.env_config import env_default_config
from models.base import COMAActorCritic
from algorithms.coma.coma_trainer import MultiAgentCOMATrainer

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
tf.debugging.set_log_device_placement(True)

EAGER = True
if not EAGER:
    tf.compat.v1.disable_eager_execution()

# training session
generation = 0
epochs = 50

# env
env_creator = lambda: BacteriaColony(env_default_config)
env = env_creator()

# algorithm
gamma = 0.99
lamb = 0.8  # lambda for TD(lambda)
seed = 0
sample_batch_size = 500
batch_size = 250
entropy_coeff = 0.01
population_size = 10

# parallelism
n_workers = 1
DEBUG = n_workers == 1
ray.init(local_mode=DEBUG)

# actor critic
actor_args = {'conv_sizes': [(32, (3, 3), 1), (32, (3, 3), 1)],
              'fc_sizes': [32],
              'last_fc_sizes': [32],
              'conv_input_shape': env.actor_terrain_obs_shape,
              'fc_input_length': np.prod(env.observation_space.shape) - np.prod(env.actor_terrain_obs_shape),
              'num_outputs': env.action_space.n}

rows, cols, depth = env.critic_observation_shape
depth += env.action_space.n
critic_args = {'conv_sizes': [(32, (2, 2), 1), (32, (2, 2), 1), (32, (2, 2), 1)],
               'fc_sizes': [64, 32],
               'conv_input_shape': (rows, cols, depth),
               'num_outputs': env.action_space.n}

ac_kwarg = {'actor_args': actor_args, 'critic_args': critic_args, 'observation_space': env.observation_space}
ac_creator = lambda: COMAActorCritic(**ac_kwarg)

# train
trainer = MultiAgentCOMATrainer(env_creator, ac_creator, population_size, seed=seed, gamma=gamma, lamb=lamb,
                                n_workers=n_workers, batch_size=batch_size, normalise_observation=True,
                                sample_batch_size=sample_batch_size, entropy_coeff=entropy_coeff)
trainer.train(epochs, generation)