from models.base import VisionAndFc, ProbabilityDistribution

import tensorflow.keras as kr
import numpy as np


class DiscreteVisionAndFcPolicy(kr.Model):
    def __init__(self, conv_sizes, fc_sizes, last_fc_sizes, num_outputs, conv_input_shape, fc_input_length):
        super().__init__('discrete_vision_and_fc_policy')
        self.net = VisionAndFc(conv_sizes, fc_sizes, last_fc_sizes, num_outputs, conv_input_shape, fc_input_length)
        self.action_dist = ProbabilityDistribution()
        self.get_actions(np.random.rand(np.prod(conv_input_shape)+fc_input_length)[None, :].astype(np.float32))

    def call(self, inputs):
        return self.net(inputs)

    def get_actions(self, inputs):
        logits = self.predict(inputs)
        return self.action_dist.predict(logits)

    def load_flat_array(self, arr):
        ptr = 0
        weight_list = []
        for var in self.variables:
            var_size = np.prod(var.shape)
            weight_list.append(np.reshape(arr[ptr: ptr + var_size], var.shape))
            ptr += var_size
        self.set_weights(weight_list)
