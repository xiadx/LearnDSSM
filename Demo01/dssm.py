#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""learn dssm model"""


import numpy as np
from keras import backend
from keras.layers import Activation, Input, Embedding
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.merge import concatenate, dot
from keras.layers.convolutional import Convolution1D
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from preprocessing import *


# model parameter
USER_WORDS = 3
ITEM_WORDS = 17
NUM_WORDS = 10000
INIT_VECTOR_SIZE = 128
FILTER_LENGTH = 1
K, L, J = 300, 128, 4
BATCH_SIZE = 64
EPOCHS = 1


def load_dssm_train_data(path):
    """
    load dssm train data
    Args:
        path: file path
    Return:
        user_document: [user_document1, user_document2]
        positive_item_document: [positive_item_document1, positive_item_document2]
        negtive_item_documents: [[negtive_item_document1, negtive_item_document2], [...], ...]
    """
    user_document, positive_item_document, negtive_item_documents = [], [], [[] for _ in range(J)]
    with open(path, "rb") as fp:
        for line in fp:
            documents = line.strip().split(",")
            user_document.append(documents[0])
            positive_item_document.append(documents[1])
            for i in range(J):
                negtive_item_documents[i].append(documents[i + 2])
    return user_document, positive_item_document, negtive_item_documents


# load dssm train data
dssm_train_data_file = "dssm_train.data"
user_document, positive_item_document, negtive_item_documents = \
    load_dssm_train_data(dssm_train_data_file)
texts = user_document + \
        positive_item_document + \
        [document for documents in negtive_item_documents for document in documents]


# preprocessing embedding tokenizer pad_sequences
tk = Tokenizer(num_words=NUM_WORDS, lower=True)
tk.fit_on_texts(texts)


user_document = pad_sequences(tk.texts_to_sequences(user_document))
positive_item_document = pad_sequences(tk.texts_to_sequences(positive_item_document))
negtive_item_documents = [pad_sequences(tk.texts_to_sequences(negtive_item_document))
                          for negtive_item_document in negtive_item_documents]


# dssm model
user_input = Input(shape=(USER_WORDS,), name="user_input")
pos_item_input = Input(shape=(ITEM_WORDS,), name="item_input")
neg_item_inputs = [Input(shape=(ITEM_WORDS,)) for j in range(J)]

user_input_embedding = Embedding(NUM_WORDS, INIT_VECTOR_SIZE)(user_input)
pos_item_embedding = Embedding(NUM_WORDS, INIT_VECTOR_SIZE)(pos_item_input)
neg_item_emdeddings = [Embedding(NUM_WORDS, INIT_VECTOR_SIZE)(neg_item_input) for neg_item_input in neg_item_inputs]

user_conv = Convolution1D(K,
                          FILTER_LENGTH,
                          padding="same",
                          input_shape=(None, INIT_VECTOR_SIZE),
                          activation="tanh")(user_input_embedding)
user_max = Lambda(lambda x: backend.max(x, axis=1), output_shape=(K,))(user_conv)
user_sem = Dense(L, activation="tanh", input_dim=K, name="user_sem")(user_max)

item_conv = Convolution1D(K,
                          FILTER_LENGTH,
                          padding="same",
                          input_shape=(None, INIT_VECTOR_SIZE),
                          activation="tanh")
item_max = Lambda(lambda x: backend.max(x, axis=1), output_shape=(K,))
item_sem = Dense(L, activation="tanh", input_dim=K, name="item_sem")

pos_item_conv = item_conv(pos_item_embedding)
neg_item_convs = [item_conv(neg_item_emdedding) for neg_item_emdedding in neg_item_emdeddings]
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


y_train = np.zeros((len(user_document), J + 1))
y_train[:, 0] = 1


# dssm model fit
model.fit([user_document, positive_item_document] + negtive_item_documents,
          y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)


# save user_sem_model item_sem_model
user_sem_model = Model(inputs=model.get_layer("user_input").input, outputs=model.get_layer("user_sem").output)
item_sem_model = Model(inputs=model.get_layer("item_input").input, outputs=model.get_layer("item_sem").get_output_at(0))
user_sem_model.save("Demo01/user_sem_model.h5")
item_sem_model.save("Demo01/item_sem_model.h5")


# # save user item deep structured semantic representation
# from preprocessing import *
# user_path = "../ml-100k/u.user"
# fcut_words_user_info(user_path)


def main():
    pass


if __name__ == "__main__":
    main()





