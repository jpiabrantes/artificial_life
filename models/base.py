import tensorflow as tf
import tensorflow.keras as kr
import tensorflow.keras.layers as kl
import numpy as np
import scipy


class ProbabilityDistribution(kr.Model):
    @staticmethod
    def call(logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class MLP(kr.Model):
    def __init__(self, hidden_sizes, num_outputs, input_shape):
        super().__init__('mlp')
        self.num_outputs = num_outputs
        self.dense_layers = [kl.Dense(hidden_sizes[0], activation='relu', input_shape=input_shape)]
        self.dense_layers += [kl.Dense(size, activation='relu') for size in hidden_sizes[1:]]
        if num_outputs:
            self.output_layer = kl.Dense(num_outputs)

    def call(self, inputs):
        for layer in self.dense_layers:
            inputs = layer(inputs)
        if self.num_outputs:
            return self.output_layer(inputs)
        else:
            return inputs


class DiscreteActor(kr.Model):
    def __init__(self, inputs, outputs):
        super(DiscreteActor, self).__init__(name='discrete_vision_and_fc_policy', inputs=inputs, outputs=outputs)
        self.action_dist = ProbabilityDistribution()

    def action_logp_pi(self, obs):
        logits = self.predict(obs)
        actions = self.action_dist.predict(logits)
        pi = scipy.special.softmax(logits, axis=1)
        logp = np.log(pi[np.arange(len(actions)), actions])
        return actions, logp, pi

    def get_actions(self, inputs):
        logits = self.predict(inputs)
        return self.action_dist.predict(logits)

    def load_flat_array(self, arr):
        ptr = 0
        weights = []
        for var in self.variables:
            var_size = np.prod(var.shape)
            weights.append(np.reshape(arr[ptr: ptr + var_size], var.shape))
            ptr += var_size
        self.set_weights(weights)


class COMAActorCritic(kr.Model):
    def __init__(self, actor_args, critic_args, observation_space):
        super().__init__('coma_ac')
        self.actor = create_vision_and_fc_network(**actor_args)
        self.critic = create_global_critic(**critic_args)

    def action_logp_pi(self, obs):
        return self.actor.action_logp_pi(obs)

    def val_and_adv(self, states_actions, actions, pi):
        qs = self.critic.predict(states_actions)
        val = np.sum(qs*pi, axis=-1)
        adv = qs[np.arange(len(actions)), actions]-val
        return val, adv


class CentralPPOActorCritic(kr.Model):
    def __init__(self, actor_args, critic_args):
        super().__init__('coma_ac')
        self.actor = create_vision_and_fc_network(**actor_args)
        self.critic = create_global_critic(**critic_args)

    def action_logp(self, obs):
        action, logp, _ = self.actor.action_logp_pi(obs)
        return action, logp


class PPOActorCritic(kr.Model):
    def __init__(self, network_args, num_actions):
        super().__init__('ppo_ac')
        self.actor = create_vision_and_fc_network(**network_args, num_outputs=num_actions)
        self.critic = create_vision_and_fc_network(**network_args, num_outputs=1, actor=False)

    def call(self, inputs):
        return self.actor(inputs), self.critic(inputs)

    def action_value_logprobs(self, obs):
        logits, value = self.predict(obs)
        actions = self.actor.action_dist.predict(logits)
        all_log_probs = np.log(scipy.special.softmax(logits, axis=1))
        log_probs = all_log_probs[np.arange(len(actions)), actions]
        return actions, np.squeeze(value, axis=-1), log_probs


class Qtran(kr.Model):
    def __init__(self, q_kwargs, Q_kwargs, V_kwargs, action_space):
        super().__init__('VDNMixer')
        self.action_space = action_space
        self.q = self._build_q(**q_kwargs, action_space=action_space)
        self.Q = self._build_Q(**Q_kwargs)
        self.V = self._build_V(**V_kwargs)

    @staticmethod
    def _build_Q(input_shape, conv_sizes, fc_sizes):
        return create_global_critic(input_shape, conv_sizes, fc_sizes, num_outputs=1)

    @staticmethod
    def _build_V(input_shape, conv_sizes, fc_sizes):
        return create_global_critic(input_shape, conv_sizes, fc_sizes, num_outputs=1)

    @staticmethod
    def _build_q(hidden_units, observation_space, action_space):
        input_layer = kl.Input(shape=observation_space.shape)
        dense = MLP(hidden_units, 0, observation_space.shape)(input_layer)
        stream_adv, stream_val = tf.split(dense, 2, axis=1)
        advantage = kl.Dense(action_space.n, activation=None, use_bias=None)(stream_adv)
        advantage = tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
        value = kl.Dense(1, activation=None, use_bias=None)(stream_val)
        Qout = value + advantage
        return kr.Model(inputs=input_layer, outputs=[Qout])

    def __call__(self, list_of_obs, list_of_act, training=True):
        """
        :param list_of_obs_act: list of n arrays with dimensions [None, obs_dim+1]
        :param training: (bool)
        :return: n
        """
        result = []
        for obs, act in zip(list_of_obs, list_of_act):
            # n_agents, obs_shape
            qout = self.q(obs)
            result.append(tf.reduce_sum(tf.one_hot(tf.cast(act, tf.int32), self.action_space.n)*qout))
        return tf.stack(result)

    def get_actions(self, obs, eps):
        batch_size = len(obs)
        actions = np.zeros((batch_size,), np.int32)
        random_mask = np.random.rand(batch_size) < eps
        actions[random_mask] = np.random.randint(0, self.action_space.n, size=sum(random_mask))
        non_random_mask = np.logical_not(random_mask)
        actions[non_random_mask] = tf.argmax(self.q(obs[non_random_mask]), axis=1)
        return actions


class VDNMixer_2(kr.Model):
    def __init__(self, obs_input_shape, conv_sizes, fc_sizes, last_fc_sizes, conv_input_shape, fc_input_length,
                 action_space):
        super().__init__('VDNMixer')
        self.action_space = action_space
        input_layer = kl.Input(shape=obs_input_shape)

        width, height, depth = conv_input_shape
        conv_input = tf.reshape(tf.slice(input_layer, (0, 0), (-1, height * width * depth)), [-1, height, width, depth])
        for filters, kernel, stride in conv_sizes:
            conv_input = kl.Conv2D(filters, kernel, stride, activation='relu')(conv_input)
        flatten = kl.Flatten()(conv_input)
        fc_input = tf.slice(input_layer, (0, height * width * depth), (-1, fc_input_length))
        fc = MLP(fc_sizes, 0, (fc_input_length,))(fc_input)
        concat = kl.Concatenate(axis=-1)([flatten, fc])
        out = MLP(last_fc_sizes, 0, (concat.shape[1],))(concat)
        stream_adv, stream_val = tf.split(out, 2, axis=1)
        advantage = kl.Dense(action_space.n, activation=None, use_bias=None)(stream_adv)
        advantage = tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
        value = kl.Dense(1, activation=None, use_bias=None)(stream_val)
        Qout = value + advantage
        self.q = kr.Model(inputs=input_layer, outputs=[Qout])

    def __call__(self, list_of_obs, list_of_act, training=True):
        """
        :param list_of_obs_act: list of n arrays with dimensions [None, obs_dim+1]
        :param training: (bool)
        :return: n
        """
        result = []
        for obs, act in zip(list_of_obs, list_of_act):
            # n_agents, obs_shape
            qout = self.q(obs)
            result.append(tf.reduce_sum(tf.one_hot(tf.cast(act, tf.int32), self.action_space.n)*qout))
        return tf.stack(result)

    def get_actions(self, obs, eps):
        batch_size = len(obs)
        actions = np.zeros((batch_size,), np.int32)
        random_mask = np.random.rand(batch_size) < eps
        actions[random_mask] = np.random.randint(0, self.action_space.n, size=sum(random_mask))
        non_random_mask = np.logical_not(random_mask)
        actions[non_random_mask] = tf.argmax(self.q(obs[non_random_mask]), axis=1)
        return actions


class VDNMixer(kr.Model):
    def __init__(self, hidden_units, observation_space, action_space):
        super().__init__('VDNMixer')
        self.action_space = action_space
        input_layer = kl.Input(shape=observation_space.shape)
        dense = MLP(hidden_units, 0, observation_space.shape)(input_layer)
        stream_adv, stream_val = tf.split(dense, 2, axis=1)
        advantage = kl.Dense(action_space.n, activation=None, use_bias=None)(stream_adv)
        advantage = tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
        value = kl.Dense(1, activation=None, use_bias=None)(stream_val)
        Qout = value + advantage
        self.q = kr.Model(inputs=input_layer, outputs=[Qout])

    def __call__(self, list_of_obs, list_of_act, training=True):
        """
        :param list_of_obs_act: list of n arrays with dimensions [None, obs_dim+1]
        :param training: (bool)
        :return: n
        """
        result = []
        for obs, act in zip(list_of_obs, list_of_act):
            # n_agents, obs_shape
            qout = self.q(obs)
            result.append(tf.reduce_sum(tf.one_hot(tf.cast(act, tf.int32), self.action_space.n)*qout))
        return tf.stack(result)

    def get_actions(self, obs, eps):
        batch_size = len(obs)
        actions = np.zeros((batch_size,), np.int32)
        random_mask = np.random.rand(batch_size) < eps
        actions[random_mask] = np.random.randint(0, self.action_space.n, size=sum(random_mask))
        non_random_mask = np.logical_not(random_mask)
        actions[non_random_mask] = tf.argmax(self.q(obs[non_random_mask]), axis=1)
        return actions


def create_vision_and_fc_network(obs_input_shape, conv_sizes, fc_sizes, last_fc_sizes, num_outputs, conv_input_shape,
                                 fc_input_length, actor=True):
    input_layer = kl.Input(shape=obs_input_shape)

    width, height, depth = conv_input_shape
    conv_input = tf.reshape(tf.slice(input_layer, (0, 0), (-1, height*width*depth)), [-1, height, width, depth])
    for filters, kernel, stride in conv_sizes:
        conv_input = kl.Conv2D(filters, kernel, stride, activation='relu')(conv_input)
    flatten = kl.Flatten()(conv_input)

    fc_input = tf.slice(input_layer, (0, height*width*depth), (-1, fc_input_length))
    fc = MLP(fc_sizes, 0, (None, fc_input_length))(fc_input)

    concat = kl.Concatenate(axis=-1)([flatten, fc])
    out = MLP(last_fc_sizes, num_outputs, (None, concat.shape[1]))(concat)
    if actor:
        return DiscreteActor(inputs=input_layer, outputs=[out])
    else:
        return kr.Model(inputs=input_layer, outputs=[out])


def create_global_critic(input_shape, conv_sizes, fc_sizes, num_outputs):
    num_outputs = num_outputs
    rows, cols, depth = input_shape
    input_layer = kl.Input(shape=(rows, cols, depth))
    actions = tf.squeeze(tf.slice(input_layer, [0, 0, 0, depth - 1], [-1, rows, cols, 1]), axis=-1)
    non_actions = tf.slice(input_layer, [0, 0, 0, 0], [-1, rows, cols, depth - 1])
    one_hot = kl.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), num_outputs),
                        input_shape=(rows, cols))(actions)
    # concat = kl.Concatenate(axis=-1)([non_actions, tf.reshape(one_hot, (-1, rows, cols, num_outputs))])
    concat = kl.Concatenate(axis=-1)([non_actions, one_hot])
    vision_layer = concat
    for i, (filters, kernel, stride) in enumerate(conv_sizes):
        vision_layer = kl.Conv2D(filters, kernel, stride, activation='relu')(vision_layer)

    flatten = kl.Flatten()(vision_layer)
    dense = MLP(fc_sizes, num_outputs, (None, flatten.shape[1]))(flatten)
    return kr.Model(inputs=input_layer, outputs=[dense])

