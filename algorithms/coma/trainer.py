import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

from utils.misc import Timer, Weights
from utils.metrics import get_kl_metric, get_coma_explained_variance, EarlyStoppingKL, entropy


class Trainer:
    def __init__(self, ac_creator, batch_size, normalize_advantages, train_pi_iters, train_v_iters, value_lr, pi_lr,
                 env_creator, target_kl, clip_ratio, vf_clip_param, entropy_coeff):
        self.batch_size = batch_size
        self.normalize_advantages = normalize_advantages
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.clip_ratio = clip_ratio
        self.vf_clip_param = vf_clip_param
        self.env = env_creator()
        self.entropy_coeff = entropy_coeff

        kl = get_kl_metric(self.env.action_space.n)
        r2score = get_coma_explained_variance(self.env.action_space.n)
        self.actor_callbacks = [EarlyStoppingKL(target_kl)]

        self._pi_optimisation_time, self._v_optimisation_time, self._species_stats = None, None, None
        self.ac = ac_creator()
        self.ac.critic.compile(optimizer=kr.optimizers.Adam(learning_rate=value_lr), loss=self._value_loss,
                               metrics=[r2score])
        self.ac.actor.compile(optimizer=kr.optimizers.Adam(learning_rate=pi_lr), loss=self._surrogate_loss,
                              metrics=[kl, entropy])

    def train(self, weights, variables, species_index):
        obs, act, adv, ret, old_log_probs, pi, q_tak, states_actions = variables

        for var, w in zip(self.ac.actor.variables, weights.actor):
            var.load(w)
        for var, w in zip(self.ac.critic.variables, weights.critic):
            var.load(w)

        if self.normalize_advantages:
            adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8) # adv / (np.std(adv) + 1e-8)
        act_adv_logs = np.concatenate([act[:, None], adv[:, None], old_log_probs[:, None]], axis=-1)
        with Timer() as pi_optimisation_timer:
            result = self.ac.actor.fit(obs, act_adv_logs, batch_size=self.batch_size, shuffle=True,
                                       epochs=self.train_pi_iters, verbose=False,
                                       callbacks=self.actor_callbacks)
            old_policy_loss = result.history['loss'][0]
            old_entropy = result.history['entropy'][0]
            kl = result.history['kl'][-1]

        qtak_act_ret = np.concatenate([q_tak[:, None], act[:, None], ret[:, None]], axis=-1)
        with Timer() as v_optimisation_timer:
            result = self.ac.critic.fit(states_actions, qtak_act_ret, shuffle=True, batch_size=self.batch_size,
                                        verbose=False, epochs=self.train_v_iters)
            old_value_loss = result.history['loss'][0]
            value_loss = result.history['loss'][-1]
            old_explained_variance = result.history['explained_variance'][0]
            explained_variance = result.history['explained_variance'][-1]

        self._pi_optimisation_time = pi_optimisation_timer.interval
        self._v_optimisation_time = v_optimisation_timer.interval
        key_value_pairs = [('LossQ', old_value_loss), ('deltaQLoss', old_value_loss - value_loss),
                           ('Old Explained Variance', old_explained_variance),
                           ('Explained variance', explained_variance),
                           ('KL', kl), ('Old entropy', old_entropy), ('LossPi', old_policy_loss),
                           ('Return', np.mean(ret))]
        self._species_stats = {'%s_%s' % (species_index, k): v for k, v in key_value_pairs}
        return Weights(self.ac.actor.get_weights(), self.ac.critic.get_weights())

    def _value_loss(self, qtak_acts_rets, qs):
        old_q_taken, actions, ret = [tf.squeeze(v) for v in tf.split(qtak_acts_rets, 3, axis=-1)]
        q = tf.reduce_sum(tf.one_hot(tf.cast(actions, tf.int32), depth=self.env.action_space.n)*qs, axis=-1)
        loss1 = tf.square(q - ret)
        q_clipped = old_q_taken + tf.clip_by_value(q - old_q_taken, -self.vf_clip_param, self.vf_clip_param)
        loss2 = tf.square(q_clipped - ret)
        loss = tf.maximum(loss1, loss2)
        mean_loss = tf.reduce_mean(loss)
        return mean_loss

    def _surrogate_loss(self, acts_advs_logs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages, old_log_probs = [tf.squeeze(v) for v in tf.split(acts_advs_logs, 3, axis=-1)]
        actions = tf.cast(actions, tf.int32)
        all_log_probs = tf.nn.log_softmax(logits)
        probs = tf.exp(all_log_probs)
        log_probs = tf.reduce_sum(tf.one_hot(actions, depth=self.env.action_space.n) * all_log_probs, axis=-1)
        ratio = tf.exp(log_probs - old_log_probs)
        min_adv = tf.where(advantages > 0, (1 + self.clip_ratio) * advantages, (1 - self.clip_ratio) * advantages)
        entropy = tf.reduce_mean(-tf.reduce_sum(tf.where(probs == 0., tf.zeros_like(probs), probs*all_log_probs),
                                                axis=1))
        surrogate_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))
        return surrogate_loss - self.entropy_coeff*entropy

    def pi_optimisation_time(self):
        return self._pi_optimisation_time

    def v_optimisation_time(self):
        return self._v_optimisation_time

    def species_stats(self):
        return self._species_stats
