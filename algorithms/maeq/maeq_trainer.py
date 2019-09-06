import ray
import pickle
import numpy as np
import tensorflow as tf


@ray.remote(num_cpus=1)
class Trainer:
    def __init__(self, brain_creator, gamma, learning_rate, action_space_dim):
        self.action_space_dim = action_space_dim
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
        loss = 0
        with tf.GradientTape() as t:
            for step_i, step in enumerate(steps):
                # step: list of experiences of size #n_agents
                step = np.stack(step)  # n_agents, 5
                n_agents = step.shape[0]
                obs, act, rew, n_obs, done, dna = [np.array(x.tolist(), t) for x, t in zip(step.T,
                                                                                           (np.float32, np.int32,
                                                                                            np.float32, np.float32,
                                                                                            np.bool, np.uint32))]
                alive_mask = np.logical_not(done)
                if np.any(alive_mask):
                    n_actions = main_qn.get_actions(n_obs[alive_mask], 0)
                    n_q_out = target_qn.q.predict(n_obs[alive_mask])
                    double_q = n_q_out[range(len(n_actions)), n_actions]

                q_out = main_qn.q(obs)
                q = tf.reduce_sum(tf.one_hot(tf.cast(act, tf.int32), self.action_space_dim)*q_out, axis=1)

                for i in range(n_agents):
                    agent_dna = dna[i]
                    kinship = np.mean((agent_dna == dna), axis=-1)
                    n = np.sum(kinship)
                    assert n > 0, 'n needs to be greater than zero'
                    Q = 1 / n * tf.reduce_sum(kinship * q)

                    n_kinship = np.mean((agent_dna == dna[alive_mask]), axis=-1)
                    n_n = np.sum(n_kinship)

                    if alive_mask[i]:
                        assert n_n > 0, 'n_n needs to be greater than zero'
                        target_q = rew[i] + self.gamma * 1/n_n * np.sum(n_kinship*double_q)
                    else:
                        target_q = 0 if n_n == 0 else 1/n_n*np.sum(n_kinship*double_q)
                    loss += tf.square(Q-target_q)

        grads = t.gradient(loss, main_qn.variables)
        self.optimiser.apply_gradients(zip(grads, main_qn.variables))
        return main_qn.get_weights(), pickle.dumps(self.optimiser.get_weights()), loss.__array__()
