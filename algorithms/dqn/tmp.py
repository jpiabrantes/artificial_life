import tensorflow as tf
import tensorflow.keras as kr
import tensorflow.keras.layers as kl
import numpy as np

from models.base import MLP


def td_loss(targets, qtot):
    return tf.keras.losses.mean_squared_error(targets, qtot)


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
    def __init__(self, hidden_units, observation_shape, action_dim):
        super().__init__('VDNMixer')
        self.action_dim = action_dim
        self.q = LSTM_Q(hidden_units, 6, observation_shape, action_dim)

    def __call__(self, list_of_obs, list_of_act, list_of_states=None, training=True):
        """
        :param list_of_obs_act: list of n arrays with dimensions [None, obs_dim+1]
        :param training: (bool)
        :return: n
        """
        if list_of_states is None:
            list_of_states = [None]*len(list_of_obs)
        result = []
        for obs, act, cell_state in zip(list_of_obs, list_of_act, list_of_states):
            # n_agents, obs_shape
            qout, cell_state = self.q(obs, cell_state)
            result.append(tf.reduce_sum(tf.reduce_sum(tf.one_hot(tf.cast(act, tf.int32), self.action_dim) * qout,
                                                      axis=-1), axis=0))
        return tf.stack(result)


# model params
observation_shape = (100, )
action_dim = 5
hidden_sizes = [512, 256, 128]
q_kwargs = {'hidden_units': hidden_sizes,
            'observation_shape': observation_shape,
            'action_dim': action_dim}
net = VDNMixer(**q_kwargs)

# create dummy data
batch_size = 10
agents = np.random.randint(0, 8, size=batch_size)
list_of_obs = [np.random.rand(a, 2, observation_shape[0]).astype(np.float32) for a in agents]
list_of_act = [np.random.randint(0, action_dim, size=(a, 2)) for a in agents]
t_targets = np.random.rand(batch_size)

#print(net(list_of_obs, list_of_act, ))
out, states = net.q(list_of_obs[1], states=np.random.rand(list_of_obs[1].shape[0], 6).astype(np.float32))

