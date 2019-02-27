#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""preprocessing ml-100k dataset"""


from __future__ import division
import os
import random
import numpy as np


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


def cut_words_item_info(item_info):
    """
    cut words for item info
    Args:
        item_info: {itemid: [title, genre]}
    Return:
        cut_words: {itemid: "word1 word2 word3"}
    """
    cut_words = {}
    stopwords = '!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n'
    for itemid, infos in item_info.items():
        title, genre = infos[0], infos[1]
        title = " ".join("".join([c for c in title if c not in stopwords]).split())
        genre = " ".join("".join([c for c in genre if c not in stopwords]).split())
        cut_words[itemid] = title + " " + genre.replace("|", " ")
    return cut_words


def cut_words_user_info(path):
    """
    Args:
        path: file path
    Return:
        cut_words: {userid: "word1 word2 word3"}
    """
    if not os.path.exists(path):
        return
    cut_words = {}
    with open(path, "rb") as fp:
        for line in fp:
            infos = line.strip().split("|")
            userid, age, gender, occupation = infos[0], infos[1], infos[2], infos[3]
            cut_words[userid] = " ".join([age, gender, occupation])
    return cut_words


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
                user_action[userid] = []
            if float(rating) >= 3.0:
                user_action[userid].append(itemid)
    return user_action


def get_dssm_train_data(user_action, user_cut_words, item_cut_words, path, size=10, J=4):
    """
    get dssm train data
    Args:
        size: get dssm train data size for once action default 10
        J: dssm model negative sampling
        user_action: user action
        user_cut_words: user cut words
        item_cut_words: item cut words
        path: save dssm train data file
    Return:
        np.ndarray
    """
    with open(path, "wb") as fp:
        for userid, itemids in user_action.items():
            for itemid in itemids:
                for _ in range(size):
                    negative_itemids = random.sample(set(item_cut_words.keys()).difference(set(itemids)), J)
                    user_document = user_cut_words[userid]
                    positive_item_document = item_cut_words[itemid]
                    fp.write(user_document + "," +
                             positive_item_document + "," +
                             ",".join([item_cut_words[itemid] for itemid in negative_itemids]) + "\n")


def main():
    genre_path = "../ml-100k/u.genre"
    item_path = "../ml-100k/u.item"
    user_path = "../ml-100k/u.user"
    user_action_path = "../ml-100k/u.data"
    dssm_train_data_path = "dssm_train.data"
    genre_dict = get_genre_dict(genre_path)
    item_info = get_item_info(item_path, genre_dict)
    item_cut_words = cut_words_item_info(item_info)
    user_cut_words = cut_words_user_info(user_path)
    user_action = get_user_action(user_action_path)
    get_dssm_train_data(user_action, user_cut_words, item_cut_words, dssm_train_data_path)


if __name__ == "__main__":
    main()