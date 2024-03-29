import ray
import numpy as np
from multiprocessing import cpu_count

# from envs.bacteria_colony.bacteria_colony import BacteriaColony
# from envs.bacteria_colony.env_config import env_default_config
from envs.deadly_colony.deadly_colony import DeadlyColony
from envs.deadly_colony.env_config import env_default_config
from models.base import COMAActorCritic
from algorithms.coma.coma_trainer_v2 import MultiAgentCOMATrainer

# training session
generation = 0
epochs = 5000
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
sample_batch_size = 30*10
batch_size = 25
entropy_coeff = 0.08
population_size = 10
update_target_freq = 1
vf_clip_param = 50

# parallelism
n_trainers = 1
n_workers = 1
assert n_workers <= cpu_count(), 'Number of workers is too high'
DEBUG = n_workers == 1
ray.init(local_mode=DEBUG)

# actor critic
# TODO: make the actor a dense network
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
               'num_outputs': env.action_space.n}

ac_kwarg = {'actor_args': actor_args, 'critic_args': critic_args, 'observation_space': env.observation_space}
ac_creator = lambda: COMAActorCritic(**ac_kwarg)

# train
trainer = MultiAgentCOMATrainer(env_creator, ac_creator, population_size, seed=seed, gamma=gamma, lamb=lamb,
                                n_workers=n_workers, batch_size=batch_size, normalise_observation=True,
                                sample_batch_size=sample_batch_size, entropy_coeff=entropy_coeff,
                                normalise_advantages=True, update_target_freq=update_target_freq, save_freq=save_freq,
                                vf_clip_param=vf_clip_param, n_trainers=n_trainers)
trainer.train(epochs, generation, load=load)
