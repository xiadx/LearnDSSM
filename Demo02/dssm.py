#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""learn dssm model"""


import numpy as np
from keras import backend
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.merge import concatenate, dot
from keras.layers.convolutional import Convolution1D
from keras.models import Model


# model parameter
USER_SIZE = 1000
ITEM_SIZE = 100
INIT_VECTOR_SIZE = 128
WINDOW_SIZE = 3
FILTER_LENGTH = 1
K, L, J = 300, 128, 4
BATCH_SIZE = 64
EPOCHS = 1


# dssm model
user_input = Input(shape=(INIT_VECTOR_SIZE,), name="user_input")
pos_item_input = Input(shape=(INIT_VECTOR_SIZE,), name="item_input")
neg_item_inputs = [Input(shape=(INIT_VECTOR_SIZE,)) for j in range(J)]

reshape_user_input = Reshape((1, INIT_VECTOR_SIZE))(user_input)
reshape_pos_item_input = Reshape((1, INIT_VECTOR_SIZE))(pos_item_input)
reshape_neg_item_inputs = [Reshape((1, INIT_VECTOR_SIZE))(neg_item_input) for neg_item_input in neg_item_inputs]

user_conv = Convolution1D(K,
                          FILTER_LENGTH,
                          padding="same",
                          input_shape=(1, INIT_VECTOR_SIZE),
                          activation="tanh")(reshape_user_input)
user_max = Lambda(lambda x: backend.max(x, axis=1), output_shape=(K,))(user_conv)
user_sem = Dense(L, activation="tanh", input_dim=K, name="user_sem")(user_max)

item_conv = Convolution1D(K,
                          FILTER_LENGTH,
                          padding="same",
                          input_shape=(1, INIT_VECTOR_SIZE),
                          activation="tanh")
item_max = Lambda(lambda x: backend.max(x, axis=1), output_shape=(K,))
item_sem = Dense(L, activation="tanh", input_dim=K, name="item_sem")

pos_item_conv = item_conv(reshape_pos_item_input)
neg_item_convs = [item_conv(reshape_neg_item_input) for reshape_neg_item_input in reshape_neg_item_inputs]
pos_item_max = item_max(pos_item_conv)
neg_item_maxs = [item_max(neg_item_conv) for neg_item_conv in neg_item_convs]
pos_item_sem = item_sem(pos_item_max)
neg_item_sems = [item_sem(neg_item_max) for neg_item_max in neg_item_maxs]

user_item_pos_output = dot([user_sem, pos_item_sem], axes=1, normalize=True)
user_item_neg_outputs = [dot([user_sem, neg_item_sem], axes=1, normalize=True) for neg_item_sem in neg_item_sems]
outputs = concatenate([user_item_pos_output] + user_item_neg_outputs)
outputs = Reshape((J + 1, 1))(outputs)

weight = np.array([1]).reshape(1, 1, 1)
with_gamma = Convolution1D(1,
                           1,
                           padding="same",
                           input_shape=(J + 1, 1),
                           activation="linear",
                           use_bias=False,
                           weights=[weight])(outputs)
with_gamma = Reshape((J + 1,))(with_gamma)

prob = Activation("softmax")(with_gamma)

model = Model(inputs=[user_input, pos_item_input] + neg_item_inputs, outputs=prob)
model.compile(optimizer="adadelta", loss="categorical_crossentropy")


# train data
user_train = np.random.random((100, INIT_VECTOR_SIZE))
pos_item_train = np.random.random((100, INIT_VECTOR_SIZE))
neg_item_inputs = [np.random.random((100, INIT_VECTOR_SIZE)) for _ in range(J)]
y_train = np.zeros((100, J + 1))
y_train[:, 0] = 1


# dssm model fit
model.fit([user_train, pos_item_train] + neg_item_inputs, y_train)


# save user_sem_model item_sem_model
user_sem_model = Model(inputs=model.get_layer("user_input").input, outputs=model.get_layer("user_sem").output)
item_sem_model = Model(inputs=model.get_layer("item_input").input, outputs=model.get_layer("item_sem").get_output_at(0))
user_sem_model.save("Demo02/user_sem_model.h5")
item_sem_model.save("Demo02/item_sem_model.h5")


# save user item deep structured semantic representation
print user_sem_model.predict(np.random.random((1, INIT_VECTOR_SIZE)))
print item_sem_model.predict(np.random.random((1, INIT_VECTOR_SIZE)))