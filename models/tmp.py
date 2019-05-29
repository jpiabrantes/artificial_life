from models.base import GlobalVisionCritic, create_model
import tensorflow as tf
import tensorflow.keras as kr
import numpy as np

model_args = {'conv_sizes': [(32, (2, 2), 1), (32, (2, 2), 1), (16, (2, 2), 1)],
               'fc_sizes': [64, 32],
               'input_shape': (50, 50, 6),
               'num_outputs': 5}


def dummy_loss(values, targets):
    return tf.reduce_sum(values-targets, axis=-1)


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = create_model(**model_args)
    model.compile(optimizer=kr.optimizers.Adam(learning_rate=0.01), loss=dummy_loss)


import tensorflow.keras.layers as kl
conv_sizes = [(32, (2, 2), 1), (32, (2, 2), 1), (16, (2, 2), 1)]
num_outputs = 5
rows, cols, depth = (50, 50, 6)
input_layer = kl.Input(shape=(rows, cols, depth))
one_hot = kl.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), num_outputs),
                    input_shape=(rows, cols))(tf.slice(input_layer, [0, 0, 0, depth-1], [-1, rows, cols, 1]))
concat = kl.Concatenate(axis=-1)([tf.slice(input_layer, (0, 0, 0 , 0), (-1, rows, cols, depth-1)), tf.reshape(one_hot, (-1, rows, cols, num_outputs))])
vision_layer = concat
for i, (filters, kernel, stride) in enumerate(conv_sizes):
    vision_layer = kl.Conv2D(filters, kernel, stride, activation='relu')(vision_layer)
    vision_layer = kl.MaxPool2D(pool_size=(2, 2))(vision_layer)

flatten = kl.Flatten()(vision_layer)
dense = MLP(fc_sizes, num_outputs, (None, None))(flatten)
self.net = kr.Model(inputs=self.input_layer, outputs=[dense])