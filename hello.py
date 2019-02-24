#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""learn dssm model"""


import numpy as np
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.merge import concatenate, dot
from keras.layers.convolutional import Convolution1D
from keras.models import Model


USER_SIZE = 1000
ITEM_SIZE = 100
INIT_VECTOR_SIZE = 128
WINDOW_SIZE = 3
FILTER_LENGTH = 1
K, L, J = 300, 128, 4


# user_input = Input(shape=(1, INIT_VECTOR_SIZE))
# pos_item_input = Input(shape=(1, INIT_VECTOR_SIZE))
# neg_item_inputs = [Input(shape=(1, INIT_VECTOR_SIZE)) for j in range(J)]

# user_conv = Convolution1D(K,
#                           FILTER_LENGTH,
#                           padding="same",
#                           input_shape=(1, INIT_VECTOR_SIZE),
#                           activation="tanh")(user_input)
# user_sem = Dense(L, activation="tanh", input_dim=K)(user_conv)
#
# item_conv = Convolution1D(K,
#                           FILTER_LENGTH,
#                           padding="same",
#                           input_shape=(1, INIT_VECTOR_SIZE),
#                           activation="tanh")
# item_sem = Dense(L, activation="tanh", input_dim=K)
#
# pos_item_conv = item_conv(pos_item_input)
# neg_item_convs = [item_conv(neg_item_input) for neg_item_input in neg_item_inputs]
# pos_item_sem = item_sem(pos_item_conv)
# neg_item_sems = [item_sem(neg_item_conv) for neg_item_conv in neg_item_convs]
#
# user_item_pos_output = dot([user_sem, pos_item_sem], axes=1, normalize=True)
# user_item_neg_outputs = [dot([user_sem, neg_item_sem], axes=1, normalize=True) for neg_item_sem in neg_item_sems]
# outputs = concatenate([user_item_pos_output] + user_item_neg_outputs)
# outputs = Reshape((J + 1, 1))(outputs)
#
# weight = np.array([1]).reshape(1, 1, 1)
# with_gamma = Convolution1D(1,
#                            1,
#                            padding="same",
#                            input_shape=(J + 1, 1),
#                            activation="linear",
#                            use_bias=False,
#                            weights=[weight])(outputs)
# prob = Activation("softmax")(with_gamma)
#
# model = Model(inputs=[user_input, pos_item_input] + neg_item_inputs, outputs=prob)
# model.compile(optimizer="adadelta", loss="categorical_crossentropy")
#

def main():
    # user_vector = np.random.random((USER_SIZE, INIT_VECTOR_SIZE))
    # item_vector = np.random.random((ITEM_SIZE, INIT_VECTOR_SIZE))
    # user_input = np.random.random((1, INIT_VECTOR_SIZE))
    # print user_input
    print [0, 1] + [2, 3, 4, 5]
