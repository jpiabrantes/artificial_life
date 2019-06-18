import tensorflow as tf
import tensorflow.keras as kr
import tensorflow.keras.layers as kl
import numpy as np

from models.base import MLP


def td_loss(targets, qtot):
    return tf.keras.losses.mean_squared_error(targets, qtot)


class VDNMixer(kr.Model):
    def __init__(self, hidden_units, observation_shape, action_dim):
        super().__init__('VDNMixer')
        self.action_dim = action_dim
        input_layer = kl.Input(shape=observation_shape)
        dense = MLP(hidden_units, 0, observation_shape)(input_layer)
        stream_adv, stream_val = tf.split(dense, 2, axis=1)
        advantage = kl.Dense(action_dim, activation=None, use_bias=None)(stream_adv)
        advantage = tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
        value = kl.Dense(1, activation=None, use_bias=None)(stream_val)
        Qout = value + advantage
        self.q = kr.Model(inputs=input_layer, outputs=[Qout])

    def __call__(self, list_of_obs_act, training=True):
        """
        :param list_of_obs_act: list of n arrays with dimensions [None, obs_dim+1]
        :param training: (bool)
        :return: n
        """
        result = []
        for obs in list_of_obs_act:
            # n_agents, obs_shape
            act = obs[:, -1]
            obs = obs[:, :-1]
            qout = self.q(obs)
            result.append(tf.reduce_sum(tf.one_hot(tf.cast(act, tf.int32), self.action_dim) * qout))
        return tf.stack(result)


# model params
observation_shape = (100, )
action_dim = 5
q_kwargs = {'hidden_units': [512, 256, 128],
            'observation_shape': observation_shape,
            'action_dim': action_dim}
net = VDNMixer(**q_kwargs)

# create dummy data
batch_size = 10
agents = np.random.randint(0, 8, size=batch_size)
list_of_obs_act = [np.hstack((np.random.rand(a, observation_shape[0]), np.random.randint(0, action_dim, size=(a, 1))))
                   for a in agents]
t_targets = np.random.rand(batch_size)


