import os
from time import time
from collections import defaultdict, namedtuple

import numpy as np
import tensorflow as tf

from utils.buffers import QTranReplayBuffer
from utils.filters import MeanStdFilter


def agent_name_to_species_index_fn(agent_name):
    return int(agent_name.split('_')[0])


Networks = namedtuple('Networks', ('main', 'target'))
Weights = namedtuple('Weights', ('main', 'target'))


class DQN_trainer:
    def __init__(self, env_creator,  brain_creator, population_size, batch_size=32, update_freq=4, gamma=0.99,
                 start_eps=1, end_eps=0.1, annealing_steps=250000, num_epochs=5000, pre_train_steps=250, tau=0.001,
                 opt_coeff=1., nopt_coeff=1.):
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

        species_dict = {species_index: {'steps': 0, 'last_update': 0, 'eps': start_eps, 'buffer': QTranReplayBuffer(),
                                        'optimiser': tf.keras.optimizers.Adam(learning_rate=0.0005)}
                        for species_index in range(population_size)}
        species_indices = list(range(5))

        # Set the rate of random action decrease.
        train_summary_writer = tf.summary.create_file_writer(tensorboard_folder)
        episodes = 0
        training_losses = []
        total_steps = 0
        while True:
            episode_buffers = {species_index: QTranReplayBuffer() for species_index in species_indices}
            # Reset environment and get first new observation
            raw_obs_dict, done_dict = env.reset(), {'__all__': False}

            episodes += 1
            ep_len = 0
            total_rew = 0
            while not done_dict['__all__']:
                state_action_species = np.ones((env.n_rows, env.n_cols, len(env.State) + 2))*-1
                state_action_species[:, :, :-2] = raw_obs_dict['state']

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
                        agent = env.agents[agent_name]
                        row, col = agent.row, agent.col
                        state_action_species[row, col, -2] = action
                        state_action_species[row, col, -1] = species_index

                # step
                n_raw_obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
                total_steps += 1

                # Save the experience in each species episode buffer
                step_buffer = defaultdict(list)
                for species_index, info in species_info.items():
                    for agent_name, obs, action in zip(info['agents'], info['obs'], info['actions']):
                        rew, done = reward_dict[agent_name], done_dict[agent_name]
                        n_obs = n_raw_obs_dict.get(agent_name, None)
                        if n_obs is not None:
                            n_obs = self.filters['ActorObsFilter'](n_obs, update=False)
                        else:
                            n_obs = obs*np.nan
                        step_buffer[species_index].append((obs, action, rew, n_obs, done))
                    species_dict[species_index]['steps'] += len(info['agents'])

                for species_index in species_info:
                    species_dict[species_index]['buffer'].add_step(step_buffer[species_index], state_action_species)

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
                            '''
                            steps: list of size batch_size with a list of size n_agents with experiences
                            g_steps: list of size batch_size with state_action_species
                            '''

                            dict_['last_update'] = dict_['steps']

                            steps, g_steps = dict_['buffer'].sample_steps(batch_size)  # Get a random batch of experiences.

                            main_qn.set_weights(self.weights[species_index].main)
                            target_qn.set_weights(self.weights[species_index].target)

                            # Below we perform the Double-DQN update to the target Q-values
                            list_of_obs = [None]*len(steps)
                            list_of_act = [None]*len(steps)
                            target_q = np.zeros((len(steps),), np.float32)
                            for i, (step, state_action_species) in enumerate(zip(steps, g_steps)):
                                # step: list of experiences of size #n_agents
                                step = np.stack(step)  # n_agents, 5
                                obs, act, rew, n_obs, done = [np.array(x.tolist(), t)
                                                              for x, t in zip(step.T,
                                                                              (np.float32, np.int32, np.float32,
                                                                               np.float32, np.bool))]
                                # obs: n_agents, obs_dim
                                list_of_obs[i] = obs
                                list_of_act[i] = act
                                assert len(np.unique(rew)) == 1

                                target_q[i] = rew[0]
                                alive_mask = np.logical_not(done)
                                if np.any(alive_mask):
                                    # compute Qtot for alive agents
                                    n_actions = main_qn.get_actions(n_obs[alive_mask], 0)
                                    q_out = target_qn.q.predict(n_obs[alive_mask])
                                    double_q = q_out[range(len(n_actions)), n_actions]
                                    target_q[i] += gamma * np.sum(double_q)

                                state_action = state_action_species[:, :, :-1]
                                state = state_action.copy()
                                mask = state_action_species[:, :, -1] == species_index
                                state[mask, -1] = -1  # hide our species actions
                                Q = main_qn.Q(state_action)
                                V = main_qn.V(state)



                            # Update the network with our target values.
                            with tf.GradientTape() as t:
                                Q = main_qn(states_actions)
                                Qhat = tf.stop_gradient(Q)
                                Qprime = main_qn(list_of_obs, list_of_act)
                                V = main_qn.V(states)
                                td_loss = self.td_loss(main_qn(states_actions), target_q)
                                opt_loss = self.td_loss(Qprime+V, Qhat)
                                nopt_loss = tf.square(tf.minimum(Qprime+V-Qhat, 0))
                                loss = td_loss + opt_coeff*opt_loss + nopt_coeff*nopt_loss
                            grads = t.gradient(loss, main_qn.variables)
                            training_losses.append(loss)
                            dict_['optimiser'].apply_gradients(zip(grads, main_qn.variables))
                            #result = main_qn.train_on_batch(x=list_of_obs, y=target_q)

                            target_w = []
                            for mw, tw in zip(main_qn.get_weights(), target_qn.get_weights()):
                                target_w.append(tau * mw + (1-tau)*tw)
                            self.weights[species_index] = Weights(main_qn.get_weights(), target_w)
            for species_index, buffer in episode_buffers.items():
                species_dict[species_index]['buffer'].add_buffer(buffer)

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
