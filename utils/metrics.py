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


def get_r2score(n_actions):
    def r2score(acts_tds, qs):
        # a trick to input actions and td(lambda) through same API
        actions, y = [tf.squeeze(v) for v in tf.split(acts_tds, 2, axis=-1)]
        prediction = tf.reduce_sum(tf.one_hot(tf.cast(actions, tf.int32), depth=n_actions)*qs, axis=-1)

        total_error = tf.reduce_sum(tf.square(y - tf.reduce_mean(y)))
        unexplained_error = tf.reduce_sum(tf.square(y - prediction))
        return 1 - unexplained_error/total_error
    return r2score
