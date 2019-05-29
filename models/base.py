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


class DiscreteActionAC(kr.Model):
    def __init__(self, name, actor, critic, observation_space):
        super().__init__(name)
        self.actor = actor
        self.critic = critic
        self.action_dist = ProbabilityDistribution()
        #  necessary to figure out the input for the `action_dist` layer
        self.action_logp_pi(np.random.rand(*observation_space.shape)[None, :].astype(np.float32))

    def call(self, inputs):
        return self.actor(inputs), self.critic(inputs)

    def action_logp_pi(self, obs):
        logits = self.actor.predict(obs)
        actions = self.action_dist.predict(logits)
        pi = scipy.special.softmax(logits, axis=1)
        logp = np.log(pi[np.arange(len(actions)), actions])
        return actions, logp, pi


class COMAActorCritic(DiscreteActionAC):
    def __init__(self, actor_args, critic_args, observation_space):
        actor = create_vision_and_fc_model(**actor_args)
        critic = create_model(**critic_args)
        super().__init__('vision_fc_ac', actor, critic, observation_space)

    def val_and_adv(self, states_actions, actions, pi):
        qs = self.critic.predict(states_actions)
        val = np.sum(qs*pi, axis=-1)
        adv = qs[np.arange(len(actions)), actions]-val
        return val, adv


def create_vision_and_fc_model(obs_input_shape, conv_sizes, fc_sizes, last_fc_sizes, num_outputs, conv_input_shape,
                               fc_input_length):
    input_layer = kl.Input(shape=obs_input_shape)

    width, height, depth = conv_input_shape
    conv_input = tf.reshape(tf.slice(input_layer, (0, 0), (-1, height*width*depth)), [-1, height, width, depth])
    for filters, kernel, stride in conv_sizes:
        conv_input = kl.Conv2D(filters, kernel, stride, activation='relu')(conv_input)
    flatten = kl.Flatten()(conv_input)

    fc_input = tf.slice(input_layer, (0, height*width*depth), (-1, fc_input_length))
    fc = MLP(fc_sizes, 0, (None, fc_input_length))(fc_input)

    concat = kl.Concatenate(axis=-1)([flatten, fc])
    out = MLP(last_fc_sizes, num_outputs, (None, None))(concat)
    return kr.Model(inputs=input_layer, outputs=[out])


def create_model(input_shape, conv_sizes, fc_sizes, num_outputs):
    num_outputs = num_outputs
    rows, cols, depth = input_shape
    input_layer = kl.Input(shape=(rows, cols, depth))
    actions = tf.slice(input_layer, [0, 0, 0, depth - 1], [-1, rows, cols, 1])
    non_actions = tf.slice(input_layer, [0, 0, 0, 0], [-1, rows, cols, depth - 1])
    one_hot = kl.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), num_outputs),
                        input_shape=(rows, cols))(actions)
    concat = kl.Concatenate(axis=-1)([non_actions, tf.reshape(one_hot, (-1, rows, cols, num_outputs))])
    vision_layer = concat
    for i, (filters, kernel, stride) in enumerate(conv_sizes):
        vision_layer = kl.Conv2D(filters, kernel, stride, activation='relu')(vision_layer)
        vision_layer = kl.MaxPool2D(pool_size=(2, 2))(vision_layer)

    flatten = kl.Flatten()(vision_layer)
    dense = MLP(fc_sizes, num_outputs, (None, None))(flatten)
    return kr.Model(inputs=input_layer, outputs=[dense])

