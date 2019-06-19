import tensorflow as tf
import tensorflow.keras as kr
import tensorflow.keras.layers as kl

import numpy as np


class LSTM_Q(kr.Model):
    def __init__(self, hidden_units, lstm_units, observation_shape, action_dim):
        super().__init__('LSTM_Q')
        self.lstm_units = lstm_units
        self.action_dim = action_dim
        input_layer = kl.Input(shape=(None,) + observation_shape)
        dense = input_layer
        for h in hidden_units:
            dense = kl.TimeDistributed(kl.Dense(h, activation='relu'))(dense)
        self.lstm = kl.LSTM(lstm_units, activation='relu', return_sequences=True, return_state=True)
        lstm_out, hidden_states, lstm_states = self.lstm(dense)
        stream_adv, stream_val = tf.split(dense, 2, axis=2)
        advantage = kl.TimeDistributed(kl.Dense(action_dim, activation=None, use_bias=None))(stream_adv)
        advantage = tf.subtract(advantage, tf.reduce_mean(advantage, axis=2, keepdims=True))
        value = kl.TimeDistributed(kl.Dense(1, activation=None, use_bias=None))(stream_val)
        Qout = value + advantage
        self.net = kr.Model(inputs=input_layer, outputs=[Qout, lstm_states])

    def __call__(self, inputs, states=None):
        if states is None:
            self.lstm.get_initial_state = self.get_zero_initial_state
        else:
            self.initial_state = tf.convert_to_tensor(states)
            self.lstm.get_initial_state = self.get_initial_state
        return self.net(inputs)

    def get_zero_initial_state(self, inputs):
        batch_size = inputs.shape[0]
        return [tf.zeros((batch_size, self.lstm_units)), tf.zeros((batch_size, self.lstm_units))]

    def get_initial_state(self, inputs):
        return [self.initial_state, self.initial_state]


class VDNMixer(kr.Model):
    def __init__(self, hidden_units, lstm_units, observation_shape, action_dim):
        super().__init__('VDNMixer')
        self.lstm_units = lstm_units
        self.action_dim = action_dim
        self.q = LSTM_Q(hidden_units, lstm_units, observation_shape, action_dim)

    def __call__(self, list_of_obs, list_of_act, list_of_states=None, training=True):
        if list_of_states is None:
            list_of_states = [None]*len(list_of_obs)
        result = []
        for obs, act, cell_state in zip(list_of_obs, list_of_act, list_of_states):
            # n_agents, obs_shape
            qout, cell_state = self.q(obs, cell_state)
            result.append(tf.reduce_sum(tf.reduce_sum(tf.one_hot(tf.cast(act, tf.int32), self.action_dim) * qout,
                                                      axis=-1), axis=0))
        return tf.stack(result)

    def get_actions(self, obs, cell_states, eps):
        batch_size = len(obs)
        actions = np.empty((batch_size,), np.int32)
        random_mask = np.random.rand(batch_size) < eps
        actions[random_mask] = np.random.randint(0, self.action_space.n, size=sum(random_mask))
        non_random_mask = np.logical_not(random_mask)
        qout, cell_states[non_random_mask] = self.q(obs[non_random_mask], cell_states[non_random_mask])
        actions[non_random_mask] = tf.argmax(qout, axis=1)
        return actions, cell_states, non_random_mask
