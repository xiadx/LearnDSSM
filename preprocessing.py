#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""preprocessing ml-100k dataset"""


from __future__ import division
import os
from collections import defaultdict
import numpy as np


def get_user_action(path):
    """
    get user action
    Args:
        path: file path
    Return:
        user_action: {userid1: [itemid1, itemid2]}
    """
    if not os.path.exists(path):
        return {}
    user_action = {}
    with open(path, "rb") as fp:
        for line in fp:
            userid, itemid, rating, timestamp = line.strip().split()
            if userid not in user_action:
                user_action["user_"+userid] = []
            if float(rating) >= 3.0:
                user_action["user_"+userid].append("item_"+itemid)
    return user_action


def get_genre_dict(path):
    """
    get genre dict
    Args:
        path: file path
    Return:
        genre_dict: {0: "unknown", 1: "Action"}
    """
    genre_dict = {}
    with open(path, "rb") as fp:
        for line in fp:
            if line.strip():
                genres = line.strip().split("|")
                genre_dict[int(genres[1])] = genres[0]
    return genre_dict


def get_item_info(path, genre_dict):
    """
    get item info
    Args:
        path: file path
        genre_dict: genre dict
    Return:
        item_info: {itemid: [title, genre]}
    """
    item_info = {}
    with open(path, "rb") as fp:
        for line in fp:
            if line.strip():
                items = line.strip().split("|")
                itemid = items[0]
                title = items[1]
                genres = []
                for i, c in enumerate(items[-19:]):
                    if c == "1":
                        genres.append(genre_dict[i])
                genres = "|".join(genres)
                item_info[itemid] = [title, genres]
    return item_info


def get_item_vector(path):
    """
    get item vector
    Args:
        path: file path
    """
    if not os.path.exists(path):
        os.system("./train.sh")
    records = {}
    with open(path, "rb") as fp:
        for line in fp:
            items = line.strip().split()
            if len(items) < 129:
                continue
            if items[0] == "</s>":
                continue
            records[items[0]] = np.array([float(_) for _ in items[1:]])
    return records


def get_user_vector(user_action, item_vector):
    """
    get user vector
    Args:
        user_action: {userid1: [(itemid1, timestamp)], userid2: [(timeid2, timestamp)]}
        item_vector: {itemid1: np.ndarray([v1, v2, v3])}
    """
    records = defaultdict(lambda: np.zeros((128,)))
    for userid, itemids in user_action.items():
        if not itemids:
            records[userid] = np.random.random((128,))
        else:
            for itemid in itemids:
                if itemid in item_vector:
                    records[userid] += item_vector[itemid]
            records[userid] /= len(itemids)
    return records


def save_user_vector(path, user_vector):
    """
    save user vector
    Args:
        path: file path
    """
    with open(path, "wb") as fp:
        for userid, vectors in user_vector.items():
            fp.write(userid + " " + " ".join([str(_) for _ in vectors]) + "\n")


def get_item2vec_train_data(input_file, output_file):
    """
    get train data for item2vec model
    Args:
        input_file: input file path
        output_file: output file path
    """
    if not os.path.exists(input_file):
        return
    records = defaultdict(lambda: [])
    with open(input_file, "rb") as fp:
        for line in fp:
            userid, itemid, rating, timestamp = line.strip().split()
            if float(rating) >= 3.0:
                records["user_"+userid].append("item_"+itemid)
    with open(output_file, "w") as fp:
        for userid, items in records.items():
            fp.write(" ".join(items) + "\n")


def get_dssm_train_data():
    pass


def main():
    input_file = "ml-100k/u.data"
    output_file = "train.data"
    get_item2vec_train_data(input_file, output_file)
    user_action_path = "ml-100k/u.data"
    item_vector_path = "item_vector.data"
    item_vector = get_item_vector(item_vector_path)
    user_action = get_user_action(user_action_path)
    user_vector = get_user_vector(user_action, item_vector)
    user_vector_path = "user_vector.data"
    save_user_vector(user_vector_path, user_vector)



if __name__ == "__main__":
    main()