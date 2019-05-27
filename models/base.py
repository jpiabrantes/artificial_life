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
        actor = VisionAndFc(**actor_args)
        critic = GlobalVisionCritic(**critic_args)
        super().__init__('vision_fc_ac', actor, critic, observation_space)

    def val_and_adv(self, states_actions, actions, pi):
        qs = self.critic.predict(states_actions)
        val = np.sum(qs*pi, axis=-1)
        adv = qs[np.arange(len(actions)), actions]-val
        return val, adv


class VisionAndFc(kr.Model):
    def __init__(self, conv_sizes, fc_sizes, last_fc_sizes, num_outputs, conv_input_shape, fc_input_length):
        super().__init__('vision_and_fc')
        self.conv_input_shape = conv_input_shape
        self.fc = MLP(fc_sizes, 0, (None, fc_input_length))
        filters, kernel, stride = conv_sizes[0]
        vision_layers = [kl.Conv2D(filters, kernel, stride, activation='relu', input_shape=conv_input_shape)]
        vision_layers += [kl.Conv2D(filters, kernel, stride, activation='relu') for filters, kernel, stride
                          in conv_sizes[1:]]
        self.vision = kr.Sequential(vision_layers)
        self.flatten = kl.Flatten()
        self.concat = kl.Concatenate(axis=-1)
        self.out = MLP(last_fc_sizes, num_outputs, (None, None))

    def call(self, inputs):
        width, height, depth = self.conv_input_shape
        sight = tf.reshape(inputs[:, :height*width*depth], [-1, height, width, depth])
        fc_input = inputs[:, height*width*depth:]
        fc_out = self.fc(fc_input)
        vision_out = self.vision(sight)
        concat = self.concat([self.flatten(vision_out), fc_out])
        return self.out(concat)


class GlobalVisionCritic(kr.Model):
    def __init__(self, input_shape, conv_sizes, fc_sizes, num_outputs):
        super().__init__('global_vision_critic')
        self.num_outputs = num_outputs

        rows, cols, depth = input_shape
        self.one_hot = kl.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), num_outputs), input_shape=(None, rows, cols))
        self.concat = kl.Concatenate(axis=-1)
        vision_layers = []
        for i, (filters, kernel, stride) in enumerate(conv_sizes):
            if not i:
                depth += num_outputs - 1
                vision_layers += [kl.Conv2D(filters, kernel, stride, activation='relu',
                                            input_shape=(rows, cols, depth))]
            else:
                vision_layers += [kl.Conv2D(filters, kernel, stride, activation='relu')]
            vision_layers += [kl.MaxPool2D(pool_size=(2, 2))]

        flatten = kl.Flatten()
        fc = MLP(fc_sizes, num_outputs, (None, None))
        self.net = kr.Sequential(vision_layers+[flatten]+[fc])
        self.build(input_shape=(None, ) + input_shape)

    def call(self, inputs):
        one_hot = self.one_hot(inputs[:, :, :, -1])
        return self.net(self.concat([inputs[:, :, :, :-1], one_hot]))
