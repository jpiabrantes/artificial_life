import tensorflow as tf
import tensorflow.keras as kr
import tensorflow.keras.layers as kl
import numpy as np
from models.base import COMAActorCritic
from envs.bacteria_colony.bacteria_colony import BacteriaColony
from envs.bacteria_colony.env_config import env_default_config

env = BacteriaColony(env_default_config)

actor_args = {'conv_sizes': [(32, (3, 3), 1), (32, (3, 3), 1)],
              'obs_input_shape': env.observation_space.shape,
              'fc_sizes': [16],
              'last_fc_sizes': [32],
              'conv_input_shape': env.actor_terrain_obs_shape,
              'fc_input_length': np.prod(env.observation_space.shape) - np.prod(env.actor_terrain_obs_shape),
              'num_outputs': env.action_space.n}

rows, cols, depth = env.critic_observation_shape
depth += 1  # will give state-actions
critic_args = {'conv_sizes': [(32, (2, 2), 1), (16, (2, 2), 1), (4, (2, 2), 1)],
               'fc_sizes': [128, 32],
               'input_shape': (rows, cols, depth),
               'num_outputs': env.action_space.n}

ac_kwarg = {'actor_args': actor_args, 'critic_args': critic_args, 'observation_space': env.observation_space}



EAGER = False
if not EAGER:
    tf.compat.v1.disable_eager_execution()


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = kr.Sequential([kl.Dense(5, input_shape=(5,)), kl.Dense(1)])
    ac = COMAActorCritic(**ac_kwarg)

print(ac.get_weights())
