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


class CrypteTest:
    def __init__(self, attributes, epsilon):
        self.db = crypte.Crypte(attr=attributes)
        self.pubkey, self.prikey = paillier.generate_paillier_keypair()
        self.epsilon = epsilon

    def insert(self, df):
        # print(df)
        df_list = df.values.tolist()
        # print(df_list)
        for v in df_list:
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