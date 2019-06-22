import ray
import pickle
import numpy as np
import tensorflow as tf

from utils.filters import MeanStdFilter
from models.base import Qtran


@ray.remote(num_cpus=1)
class Trainer:
    def __init__(self, brain_kwargs, gamma, learning_rate, opt_coeff, nopt_coeff, critic_observation_shape):
        self.main_qn, self.target_qn = Qtran(**brain_kwargs), Qtran(**brain_kwargs)
        self.gamma = gamma
        self.opt_coeff = opt_coeff
        self.nopt_coeff = nopt_coeff
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.filters = {'CriticObsFilter': MeanStdFilter(shape=critic_observation_shape)}

    def train(self, weights, steps, sta_act_spe, locs, n_sta_act_star, optimiser_weights, species_index):
        obs_filter = self.filters['CriticObsFilter']
        main_qn, target_qn = self.main_qn, self.target_qn
        if optimiser_weights:
            self.optimiser.set_weights(pickle.loads(optimiser_weights))

        main_qn.set_weights(weights.main)
        target_qn.Q.set_weights(weights.target)

        # Below we perform the Double-DQN update to the target Q-values
        state_action_shape = tuple(main_qn.Q.input.shape)[1:]
        n_states_actions_star = np.empty((len(steps),) + state_action_shape, np.float32)
        states_actions = np.empty((len(steps),) + state_action_shape, np.float32)
        states_actions_star = np.empty((len(steps),) + state_action_shape, np.float32)
        states = np.empty((len(steps),) + state_action_shape, np.float32)

        list_of_obs = [None] * len(steps)
        list_of_act = [None] * len(steps)
        list_of_act_star = [None] * len(steps)
        target_q = np.empty((len(steps),), np.float32)
        not_terminated_mask = np.empty((len(steps)), np.bool)
        for i, (step, state_action_species, loc, n_state_action_star) in enumerate(zip(steps, sta_act_spe, locs,
                                                                                       n_sta_act_star)):
            # step: list of experiences of size #n_agents
            step = np.stack(step)  # n_agents, 5
            obs, act, rew, n_obs, done = [np.array(x.tolist(), t)
                                          for x, t in zip(step.T, (np.float32, np.int32, np.float32,
                                                                   np.float32, np.bool))]
            assert len(np.unique(rew)) == 1
            # obs: n_agents, obs_dim
            list_of_obs[i] = obs
            list_of_act[i] = act
            list_of_act_star[i] = main_qn.get_actions(obs, 0)

            # state_action
            state_action = state_action_species[:, :, :-1]
            state_action[:, :, :-1] = obs_filter(state_action[:, :, :-1])
            states_actions[i] = state_action

            # state
            state = state_action.copy()
            mask = state_action_species[:, :, -1] == species_index
            state[mask, -1] = -1  # hide our species actions
            states[i] = state

            # state_action_star
            # TODO: filter g_states
            state_action_star = state_action.copy()
            for (row, col), act in zip(loc, list_of_act_star[i]):
                state_action_star[row, col, -1] = act
            states_actions_star[i] = state_action_star
            n_states_actions_star[i, :, :, :-1] = obs_filter(n_state_action_star[:, :, :-1], update=False)
            n_states_actions_star[i, :, :, -1] = n_state_action_star[:, :, -1]
            target_q[i] = rew[0]
            alive_mask = np.logical_not(done)
            not_terminated_mask[i] = np.any(alive_mask)

        target_q[not_terminated_mask] += self.gamma * target_qn.Q.predict(n_states_actions_star[not_terminated_mask]).ravel()
        # Update the network with our target values.
        with tf.GradientTape() as t:
            Q = tf.squeeze(main_qn.Q(states_actions), axis=1)
            Qhat = tf.stop_gradient(Q)
            Q_hat_star = tf.stop_gradient(tf.squeeze(main_qn.Q(states_actions_star), axis=1))
            Qprime = main_qn(list_of_obs, list_of_act)
            Qprime_star = main_qn(list_of_obs, list_of_act_star)
            V = tf.squeeze(main_qn.V(states), axis=1)
            td_loss = tf.keras.losses.mean_squared_error(Q, target_q)
            opt_loss = tf.keras.losses.mean_squared_error(Qprime_star + V, Q_hat_star)
            nopt_loss = tf.reduce_mean(tf.square(tf.minimum(Qprime + V - Qhat, 0)))
            loss = td_loss + self.opt_coeff * opt_loss + self.nopt_coeff * nopt_loss
        grads = t.gradient(loss, main_qn.variables)
        self.optimiser.apply_gradients(zip(grads, main_qn.variables))
        return main_qn.get_weights(), pickle.dumps(self.optimiser.get_weights()), loss.__array__()

    def get_filters(self, flush_after=False):
        """Returns a snapshot of filters.

        Args:
            flush_after (bool): Clears the filter buffer state.

        Returns:
            return_filters (dict): Dict for serializable filters
        """
        return_filters = {}
        for k, f in self.filters.items():
            return_filters[k] = f.as_serializable()
            if flush_after:
                f.clear_buffer()
        return return_filters

    def sync_filters(self, new_filters):
        """Changes self's filter to given and rebases any accumulated delta.

        Args:
            new_filters (dict): Filters with new state to update local copy.
        """
        assert all(k in new_filters for k in self.filters)
        for k in self.filters:
            self.filters[k].sync(new_filters[k])
