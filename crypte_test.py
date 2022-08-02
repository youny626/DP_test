from crypte_src import crypte
import unittest
import numpy as np
import random
import resource
import time
import phe.paillier as paillier
from dataclasses import dataclass
# import util
import json
from crypte_src import provision
# from crypte import Crypte
import pandas as pd
import time
import diffprivlib as dp
from tqdm import tqdm


class CrypteTest:
    def __init__(self, attributes, epsilon):
        self.db = crypte.Crypte(attr=attributes)
        self.pubkey, self.prikey = paillier.generate_paillier_keypair()
        self.epsilon = epsilon

    def insert(self, df):
        # print(df)
        df_list = df.values.tolist()
        # print(df_list)
        for v in tqdm(df_list):
            # print(v)
            encrypted_v = provision.lab_encrypt_vector(self.pubkey, v)
            # print(enc_v)
            self.db.insert(encrypted_v)

    def count(self, attr_num, start, end):
        encrypted_filter = self.db.filter(attr_num, start, end)
        # print(encrypted_filter)
        encrypted_count = self.db.count(encrypted_filter)
        # print(encrypted_count)
        answer = provision.lab_decrypt(self.prikey, encrypted_count)
        # print(answer)
        # true_answer = dummy.iloc[:, 2].sum()
        # print(true_answer)
        # assert answer == true_answer

        sensitivity = 1
        laplace_noise = np.random.laplace(loc=0, scale=2 * sensitivity / self.epsilon)
        # print(laplace_noise)
        encrypted_noise = provision.lab_encrypt(self.pubkey, laplace_noise)
        # print(encrypted_noise)
        encrypted_count = provision.lab_add(encrypted_count, encrypted_noise)
        # print(encnt)

        # CSP checks whether the privacy budget is exceeded

        result = provision.lab_decrypt(self.prikey, encrypted_count)
        # print(m)

        result = result + laplace_noise
        return result


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode], prefix=feature_to_encode, prefix_sep='_')
    # print(dummies.head(10))
    # print(dummies.shape[1])
    num_cols = dummies.shape[1]
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    # print(res.head(10))
    return res, num_cols


if __name__ == '__main__':

    epsilon = 1.0

    df = pd.read_csv("./PUMS.csv")
    # df.drop(["Unnamed: 0", "state", "puma", "income", "latino", "black", "asian"], axis=1, inplace=True)
    # df.drop(["income", "pid"], axis=1, inplace=True)

    df = df[["age"]]
    print(df.head(10))

    features_to_encode = ['age']#, 'sex', 'educ', 'married']  # , 'race']
    df_one_hot = df.copy()
    crypte_attrs = []
    for feature in features_to_encode:
        df_one_hot, num_cols = encode_and_bind(df_one_hot, feature)
        crypte_attrs.append(num_cols)
    # print(df_one_hot.head(10))
    df_one_hot_sum = df_one_hot.sum(axis=0).to_frame().T
    print(df_one_hot_sum)

    crypte = CrypteTest(crypte_attrs, epsilon)
    start = time.time()
    crypte.insert(df_one_hot_sum)
    elapsed = time.time() - start
    print(f"time inserting vector to crypte: {elapsed} s")

    start = time.time()
    res = crypte.count(attr_num=1, start=1, end=crypte_attrs[0])
    elapsed = time.time() - start
    print(f"query time: {elapsed} s")

    print(res)