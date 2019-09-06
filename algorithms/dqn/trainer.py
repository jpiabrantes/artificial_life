import ray
import pickle
import numpy as np
import tensorflow as tf


@ray.remote(num_cpus=1)
class Trainer:
    def __init__(self, brain_creator, gamma, learning_rate):
        self.main_qn, self.target_qn = brain_creator(), brain_creator()
        self.gamma = gamma
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train(self, weights, steps, optimiser_weights):
        main_qn, target_qn = self.main_qn, self.target_qn
        if optimiser_weights:
            self.optimiser.set_weights(pickle.loads(optimiser_weights))
        main_qn.set_weights(weights.main)
        target_qn.set_weights(weights.target)

        # Below we perform the Double-DQN update to the target Q-values
        list_of_obs = [None] * len(steps)
        list_of_act = [None] * len(steps)
        target_q = np.zeros((len(steps),), np.float32)
        for i, step in enumerate(steps):
            # step: list of experiences of size #n_agents
            step = np.stack(step)  # n_agents, 5
            obs, act, rew, n_obs, done = [np.array(x.tolist(), t) for x, t in zip(step.T, (np.float32, np.int32,
                                                                                           np.float32, np.float32,
                                                                                           np.bool))]
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
                target_q[i] += self.gamma * np.mean(double_q)

        # Update the network with our target values.
        with tf.GradientTape() as t:
            loss = tf.keras.losses.mean_squared_error(main_qn(list_of_obs, list_of_act), target_q)
        grads = t.gradient(loss, main_qn.variables)
        self.optimiser.apply_gradients(zip(grads, main_qn.variables))
        return main_qn.get_weights(), pickle.dumps(self.optimiser.get_weights()), loss.__array__()
