import tensorflow as tf


def get_kl_metric(n_actions):
    def kl(acts_advs_logs, logits):
        actions, advantages, old_log_probs = [tf.squeeze(v) for v in tf.split(acts_advs_logs, 3, axis=-1)]
        actions = tf.cast(actions, tf.int32)
        all_log_probs = tf.nn.log_softmax(logits)
        log_probs = tf.reduce_sum(tf.one_hot(actions, depth=n_actions) * all_log_probs, axis=-1)
        return -tf.reduce_mean(log_probs - old_log_probs)
    return kl


def entropy(acts_advs_logs, logits):
    all_log_probs = tf.nn.log_softmax(logits)
    probs = tf.exp(all_log_probs)
    return tf.reduce_mean(-tf.reduce_sum(tf.where(probs == 0., tf.zeros_like(probs), probs * all_log_probs), axis=1))


def get_coma_explained_variance(n_actions):
    def explained_variance(qtak_acts_tds, qs):
        # a trick to input actions and td(lambda) through same API
        _, actions, y = [tf.squeeze(v) for v in tf.split(qtak_acts_tds, 3, axis=-1)]
        prediction = tf.reduce_sum(tf.one_hot(tf.cast(actions, tf.int32), depth=n_actions)*qs, axis=-1)

        error = y - prediction
        return 1 - variance(error)/variance(y)
    return explained_variance


def ppo_explained_variance(ret_val, values):
    # a trick to input actions and td(lambda) through same API
    returns, _ = [tf.squeeze(v) for v in tf.split(ret_val, 2, axis=-1)]
    return explained_variance(returns, values)


def explained_variance(y, y_hat):
    error = y - y_hat
    return 1 - variance(error)/variance(y)


def variance(x):
    return tf.reduce_mean(tf.square(x - tf.reduce_mean(x)))


# CALLBACKS
class EarlyStoppingKL(tf.keras.callbacks.Callback):
    def __init__(self, target_kl):
        super(EarlyStoppingKL, self).__init__()

        self.target_kl = target_kl
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        kl = logs.get('kl')
        if abs(kl) > 1.5*self.target_kl:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
