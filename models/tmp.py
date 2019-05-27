from models.base import GlobalVisionCritic
import tensorflow as tf
import tensorflow.keras as kr

model_args = {'conv_sizes': [(32, (2, 2), 1), (32, (2, 2), 1), (32, (2, 2), 1)],
               'fc_sizes': [64, 32],
               'input_shape': (50, 50, 6),
               'num_outputs': 5}


def dummy_loss(values, targets):
    return tf.reduce_sum(values-targets, axis=-1)


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = GlobalVisionCritic(**model_args)
    model.compile(optimizer=kr.optimizers.Adam(learning_rate=0.01), loss=dummy_loss)
