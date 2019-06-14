import ray
import numpy as np
from multiprocessing import cpu_count

from envs.deadly_colony.deadly_colony import DeadlyColony
from envs.deadly_colony.env_config import env_default_config
from models.base import CentralPPOActorCritic
from algorithms.central_ppo.central_ppo import CentralPPOTrainer

# training session
generation = 0
epochs = 10000
save_freq = 30
load = False

# env
config = env_default_config.copy()
config['greedy_reward'] = True
env_creator = lambda: DeadlyColony(config)
env = env_creator()

# algorithm
gamma = 0.99
lamb = 0.8  # lambda for TD(lambda)
seed = 0
sample_batch_size = 300*10
batch_size = 250
entropy_coeff = 0.01
population_size = 9
update_target_freq = 1
uniform_sample = True
vf_clip_param = 50

# parallelism
n_trainers = 4
n_workers = 6
assert n_workers <= cpu_count(), 'Number of workers is too high'
DEBUG = n_workers == 1
ray.init(local_mode=DEBUG)

# actor critic
actor_args = {'conv_sizes': [(32, (3, 3), 1), (32, (3, 3), 1)],
              'obs_input_shape': env.observation_space.shape,
              'fc_sizes': [16],
              'last_fc_sizes': [32],
              'conv_input_shape': env.actor_terrain_obs_shape,
              'fc_input_length': np.prod(env.observation_space.shape) - np.prod(env.actor_terrain_obs_shape),
              'num_outputs': env.action_space.n}

rows, cols, depth = env.critic_observation_shape
depth += 1  # will give state-actions
critic_args = {'conv_sizes': [(32, (6, 6), (3, 3)), (64, (4, 4), (2, 2)), (64, (3, 3), (1, 1))],
               'fc_sizes': [512],
               'input_shape': (rows, cols, depth),
               'num_outputs': 1}

ac_kwarg = {'actor_args': actor_args, 'critic_args': critic_args}
ac_creator = lambda: CentralPPOActorCritic(**ac_kwarg)

# train
trainer = CentralPPOTrainer(env_creator, ac_creator, population_size, seed=seed, gamma=gamma, lamb=lamb,
                            n_workers=n_workers, batch_size=batch_size, normalise_observation=True,
                            sample_batch_size=sample_batch_size, entropy_coeff=entropy_coeff,
                            normalise_advantages=True, update_target_freq=update_target_freq, save_freq=save_freq,
                            n_trainers=n_trainers, vf_clip_param=vf_clip_param, uniform_sample=uniform_sample)
trainer.train(epochs, generation, load=load)
