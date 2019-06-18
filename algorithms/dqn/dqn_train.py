import os
from time import time
from collections import defaultdict, namedtuple

import numpy as np
import tensorflow as tf

from utils.buffers import ReplayBuffer
from utils.filters import MeanStdFilter


def agent_name_to_species_index_fn(agent_name):
    return int(agent_name.split('_')[0])


Networks = namedtuple('Networks', ('main', 'target'))
Weights = namedtuple('Weights', ('main', 'target'))


class DQN_trainer:
    def __init__(self, env_creator,  brain_creator, population_size, batch_size=32, update_freq=4, gamma=0.99,
                 start_eps=1, end_eps=0.1, annealing_steps=50000, num_epochs=5000, pre_train_steps=10000, tau=0.001):
        env = env_creator()
        self.env = env

        exp_name = 'basic'
        algorithm_folder = os.path.dirname(os.path.abspath(__file__))
        exp_folder = os.path.join(algorithm_folder, 'checkpoints', env.name, exp_name)
        tensorboard_folder = os.path.join(exp_folder, 'tensorboard', 'dqn_%d' % int(time()))

        self.weights = {}
        for species_index in range(population_size):
            brain = brain_creator()
            weights = brain.get_weights()
            self.weights[species_index] = Weights(weights, weights)

        main_qn = brain_creator()
        target_qn = brain_creator()

        self.filters = {'ActorObsFilter': MeanStdFilter(shape=self.env.observation_space.shape),
                        'CriticObsFilter': MeanStdFilter(shape=self.env.critic_observation_shape)}

        species_dict = {species_index: {'steps': 0, 'last_update': 0, 'eps': start_eps, 'buffer': ReplayBuffer(),
                                        'optimiser': tf.keras.optimizers.Adam(learning_rate=0.001)}
                        for species_index in range(population_size)}
        species_indices = list(range(5))

        # Set the rate of random action decrease.

        train_summary_writer = tf.summary.create_file_writer(tensorboard_folder)
        episodes = 0
        training_losses = []
        total_steps = 0
        while total_steps < 100000:
            episode_buffers = {species_index: ReplayBuffer() for species_index in species_indices}
            # Reset environment and get first new observation
            raw_obs_dict, done_dict = env.reset(), {'__all__': False}

            episodes += 1
            ep_len = 0
            total_rew = 0
            while not done_dict['__all__']:
                del raw_obs_dict['state']
                # collect observations for each species
                species_info = defaultdict(lambda: {'obs': [], 'agents': []})
                for agent_name, raw_obs in raw_obs_dict.items():
                    species_index = agent_name_to_species_index_fn(agent_name)
                    species_info[species_index]['agents'].append(agent_name)
                    filter_ = self.filters['ActorObsFilter']
                    species_info[species_index]['obs'].append(filter_(raw_obs, self.filters))

                # compute actions of each species
                action_dict = {}
                for species_index, info in species_info.items():
                    main_qn.set_weights(self.weights[species_index].main)
                    eps = species_dict[species_index]['eps']
                    observation_arr = np.stack(info['obs'])
                    actions = main_qn.get_actions(observation_arr, eps)
                    info['actions'] = actions
                    for agent_name, action in zip(info['agents'], actions):
                        action_dict[agent_name] = action

                # step
                n_raw_obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
                total_steps += 1

                # Save the experience in each species episode buffer
                for species_index, info in species_info.items():
                    buffer = episode_buffers[species_index]
                    species_dict[species_index]['steps'] += len(info['agents'])
                    for agent_name, obs, action in zip(info['agents'], info['obs'], info['actions']):
                        rew, done = reward_dict[agent_name], done_dict[agent_name]
                        n_obs = n_raw_obs_dict.get(agent_name, None)
                        if n_obs is not None:
                            n_obs = self.filters['ActorObsFilter'](n_obs, update=False)
                        else:
                            n_obs = obs*np.nan
                        buffer.add((obs, action, rew, n_obs, done, species_dict[species_index]['steps']))

                raw_obs_dict = n_raw_obs_dict
                total_rew += sum(reward_dict.values())
                ep_len += 1

                for species_index in species_info:
                    dict_ = species_dict[species_index]
                    if dict_['steps'] > pre_train_steps:
                        if dict_['steps'] > annealing_steps:
                            coeff = (dict_['steps']-annealing_steps)/annealing_steps
                            dict_['eps'] = max(coeff * 0.01 + (1 - coeff) * end_eps, 0.01)
                        else:
                            coeff = dict_['steps']/annealing_steps
                            dict_['eps'] = max(coeff*end_eps+(1-coeff)*start_eps, end_eps)

                        if dict_['steps'] - dict_['last_update'] >= update_freq:
                            dict_['last_update'] = dict_['steps']
                            #  obs: batch_size, #agents, obs_dim
                            obs, actions, rew, n_obs, done, steps = dict_['buffer'].sample(batch_size)  # Get a random batch of experiences.

                            main_qn.set_weights(self.weights[species_index].main)
                            target_qn.set_weights(self.weights[species_index].target)

                            # Below we perform the Double-DQN update to the target Q-values
                            list_of_obs = []
                            list_of_act = []
                            unique_steps = np.unique(steps)
                            target_q = np.zeros((len(unique_steps),), np.float32)
                            for i, step in enumerate(unique_steps):
                                mask = step == steps
                                s_rew, s_done, s_obs, s_act = rew[mask], done[mask], obs[mask], actions[mask]
                                list_of_obs.append(s_obs)
                                list_of_act.append(s_act)
                                assert len(np.unique(s_rew)) == 1

                                target_q[i] = s_rew[0]
                                alive_mask = np.logical_not(s_done)
                                if np.any(alive_mask):
                                    # compute Qtot for alive agents
                                    n_actions = main_qn.get_actions(s_obs[alive_mask], 0)
                                    q_out = target_qn.q.predict(s_obs[alive_mask])
                                    double_q = q_out[range(len(n_actions)), n_actions]
                                    target_q[i] += gamma * np.sum(double_q)

                            # Update the network with our target values.
                            with tf.GradientTape() as t:
                                loss = self.td_loss(main_qn(list_of_obs, list_of_act), target_q)
                            grads = t.gradient(loss, main_qn.variables)
                            training_losses.append(loss)
                            dict_['optimiser'].apply_gradients(zip(grads, main_qn.variables))
                            #result = main_qn.train_on_batch(x=list_of_obs, y=target_q)

                            target_w = []
                            for mw, tw in zip(main_qn.get_weights(), target_qn.get_weights()):
                                target_w.append(tau * mw + (1-tau)*tw)
                            self.weights[species_index] = Weights(main_qn.get_weights(), target_w)
            for species_index, buffer in episode_buffers.items():
                species_dict[species_index]['buffer'].add(buffer.buffer)

            training_loss = 0 if not len(training_losses) else np.mean(training_losses)
            training_losses = []
            metrics = {'EpLen': ep_len, 'Reward': total_rew, 'QLoss': training_loss}
            for species_index, dict_ in species_dict.items():
                for key in ('steps', 'eps'):
                    metrics['%d_%s' % (species_index, key)] = dict_[key]
            metrics.update(info_dict['__all__'])
            print('\n\nEpisodes: ', episodes)
            with train_summary_writer.as_default():
                for k, v in metrics.items():
                    print(k, v)
                    tf.summary.scalar(k, v, step=total_steps)

    def td_loss(self, targets, qtot):
        return tf.keras.losses.mean_squared_error(targets, qtot)


if __name__ == '__main__':
    from models.base import VDNMixer
    from envs.deadly_colony.deadly_colony import DeadlyColony
    from envs.deadly_colony.env_config import env_default_config

    config = env_default_config.copy()
    config['greedy_reward'] = True
    env_creator = lambda: DeadlyColony(config)
    env = env_creator()
    q_kwargs = {'hidden_units': [512, 256, 128],
                'observation_space': env.observation_space,
                'action_space': env.action_space}
    brain_creator = lambda: VDNMixer(**q_kwargs)
    trainer = DQN_trainer(env_creator, brain_creator, population_size=5)
