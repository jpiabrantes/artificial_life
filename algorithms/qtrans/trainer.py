import ray
import pickle
import numpy as np
import tensorflow as tf


@ray.remote(num_cpus=1)
class Trainer:
    def __init__(self, brain_creator, gamma, learning_rate, opt_coeff, nopt_coeff):
        self.main_qn, self.target_qn = brain_creator(), brain_creator()
        self.gamma = gamma
        self.opt_coeff = opt_coeff
        self.nopt_coeff = nopt_coeff
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train(self, weights, steps, g_steps, optimiser_weights, species_index):
        main_qn, target_qn = self.main_qn, self.target_qn
        if optimiser_weights:
            self.optimiser.set_weights(pickle.loads(optimiser_weights))

        main_qn.set_weights(weights.main)
        target_qn.Q.set_weights(weights.target)

        # Below we perform the Double-DQN update to the target Q-values
        state_action_shape = g_steps[0].shape
        state_action_shape[-1] -= 1
        states_actions = np.empty((len(steps),) + state_action_shape)
        states = np.empty((len(steps),) + state_action_shape)
        list_of_obs = [None] * len(steps)
        list_of_act = [None] * len(steps)
        target_q = np.zeros((len(steps),), np.float32)
        for i, (step, state_action_species) in enumerate(zip(steps, g_steps)):
            # step: list of experiences of size #n_agents
            step = np.stack(step)  # n_agents, 5
            obs, act, rew, n_obs, done = [np.array(x.tolist(), t)
                                          for x, t in zip(step.T, (np.float32, np.int32, np.float32,
                                                                   np.float32, np.bool))]
            assert len(np.unique(rew)) == 1
            # obs: n_agents, obs_dim
            list_of_obs[i] = obs
            list_of_act[i] = act

            state_action = state_action_species[:, :, :-1]
            state = state_action.copy()
            mask = state_action_species[:, :, -1] == species_index
            state[mask, -1] = -1  # hide our species actions
            states_actions[i] = state_action
            states[i] = state
            target_q[i] = rew[0]
            alive_mask = np.logical_not(done)
            if np.any(alive_mask):
                # compute Qtot for alive agents
                target_q[i] += self.gamma * target_qn.Q(state_action)

        # Update the network with our target values.
        with tf.GradientTape() as t:
            Q = main_qn(states_actions)
            Qhat = tf.stop_gradient(Q)
            Qprime = main_qn(list_of_obs, list_of_act)
            V = main_qn.V(states)
            td_loss = tf.keras.losses.mean_squared_error(main_qn(states_actions), target_q)
            opt_loss = tf.keras.losses.mean_squared_error(Qprime + V, Qhat)
            nopt_loss = tf.reduce_mean(tf.square(tf.minimum(Qprime + V - Qhat, 0)))
            loss = td_loss + self.opt_coeff * opt_loss + self.nopt_coeff * nopt_loss
        grads = t.gradient(loss, main_qn.variables)
        self.optimiser.apply_gradients(zip(grads, main_qn.variables))
        return main_qn.get_weights(), pickle.dumps(self.optimiser.get_weights()), loss.__array__()
