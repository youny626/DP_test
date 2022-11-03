import multiprocessing

from crypte_src import crypte
import unittest
import numpy as np
import random
import resource
import time
import phe.paillier as paillier
import json
import pandas as pd
import time
from tqdm import tqdm
import crypte_new.src.crypte.provision as pro
import crypte_new.src.crypte.core as cte
from crypte_new.src.crypte.core import Cdata
from crypte_new.src.crypte.core import CSP, AS
from multiprocessing import Process, Manager


class CrypteTest:
    def __init__(self, attributes : list, epsilon):
        self.attributes = attributes.copy()
        self.epsilon = epsilon

        # set up CSP and derive keys
        self.cs = CSP()
        self.cs.reg_eps(epsilon)
        self.pk, self.sk = self.cs.derive_key()

        # set up AS, pk and init cdata
        self.cdata = Cdata(attr=attributes)
        self.cdata.set_pk(self.pk)

        self.a = AS()
        self.a.set_key(self.pk)
        self.a.load_data(self.cdata)

        # self.db = crypte.Crypte(attr=attributes)
        # self.pubkey, self.prikey = paillier.generate_paillier_keypair()


    def insert(self, data):
        # print(df)

        # print(df_list)
        for v in tqdm(data):
            # print(v)
            encrypted_v = pro.lab_encrypt_vector(self.pk, v)
            # print(enc_v)
            self.cdata.insert(encrypted_v)

    def insert_parallel(self, L: list, data):
        # print(df)

        # print(df_list)
        for v in tqdm(data):
            # print(v)
            encrypted_v = pro.lab_encrypt_vector(self.pk, v)
            # print(enc_v)
            # self.cdata.insert(encrypted_v)
            L.append(encrypted_v)


    def count(self, attr_num, start, end, epsilon):

        def test_count(obj, attr_pred, val_pred_1, val_pred_2):
            encfilter = cte.filter(obj, attr_pred, val_pred_1, val_pred_2)
            encnt = encfilter.count()
            return encnt

        c = self.a.execute(test_count, attr_num, start, end)
        print("\n")
        print("True counting output is", self.cs.reveal_clear(c))
        sens = 1
        c = self.a.laplace_distort(sens, epsilon, 1, c)
        noisy_res = self.cs.reveal_noisy(c, sens, epsilon)
        noisy_res = max(0, noisy_res)
        print("DP counting output is", noisy_res)

        return noisy_res

    def group_by(self, attr_num, epsilon):

        def test_group_by(obj, attr_pred):
            encnt = obj.group_by(attr_pred)
            return encnt

        c = self.a.execute(test_group_by, attr_num)
        print("\n")
        print("True group by output is", self.cs.reveal_clear_vector(c))
        sens = 1
        # print(self.attributes, attr_num-1)
        # print(c)
        c = self.a.laplace_distort_vector(sens, epsilon, self.attributes[attr_num-1], c)
        noisy_res = self.cs.reveal_noisy_vector(c, sens, epsilon, self.attributes[attr_num-1])
        noisy_res = [max(0, x) for x in noisy_res]
        print("DP group by output is", noisy_res)

        return noisy_res

    def serialize(self, file_name):

        enc_data = self.cdata.get_data()
        # print(enc_data)
        values = []
        labels = []
        for enc_v in enc_data:
            values.append([(str(elem[1].ciphertext()), elem[1].exponent) for elem in enc_v])
            labels.append([str(elem[0]) for elem in enc_v])

        enc_with_one_pub_key = {}
        enc_with_one_pub_key['public_key'] = {'n': self.pk.n} # 'g': self.pk.g
        enc_with_one_pub_key['private_key'] = {'p': self.sk.p, 'q': self.sk.q}
        enc_with_one_pub_key['values'] = values
        enc_with_one_pub_key['labels'] = labels
        serialized = json.dumps(enc_with_one_pub_key)
        # Writing to sample.json
        with open(file_name, "w") as outfile:
            outfile.write(serialized)

    def deserialize(self, file_name):

        with open(file_name, 'r') as openfile:
            data_in = json.load(openfile)

        serialized = json.dumps(data_in)
        received_dict = json.loads(serialized)

        pk = received_dict['public_key']
        pubkey = paillier.PaillierPublicKey(n=int(pk['n']))

        sk = received_dict['private_key']
        privkey = paillier.PaillierPrivateKey(public_key=pubkey, p=int(sk['p']), q=int(sk['q']))

        values = received_dict['values']
        labels = received_dict['labels']
        # enc_nums = []
        data = []
        for env_lst, lab_lst in zip(values, labels):
            enc_nums = [paillier.EncryptedNumber(pubkey, int(x[0]), int(x[1])) for x in env_lst]
            labs = [int(x) for x in lab_lst]
            lab_env = [[labs[i], enc_nums[i]] for i in range(len(labs))]
            data.append(lab_env)

        self.pk = pubkey
        self.sk = privkey
        self.cs.pk = self.pk
        self.cs.sk = self.sk
        self.cdata.set_data(data)
        self.cdata.set_pk(self.pk)
        self.a.load_data(self.cdata)
        self.a.set_key(self.pk)

        return data


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

    df = pd.read_csv("adult.csv")
    # df.drop(["Unnamed: 0", "state", "puma", "income", "latino", "black", "asian"], axis=1, inplace=True)
    # df.drop(["income", "pid"], axis=1, inplace=True)
    # df = df.sample(4)

    # df = df[["age"]]
    # print(df.head(10))

    features_to_encode = ["age", "race", "sex", "income"]
    df_one_hot = df[features_to_encode].copy()
    crypte_attrs = []
    for feature in features_to_encode:
        df_one_hot, num_cols = encode_and_bind(df_one_hot, feature)
        crypte_attrs.append(num_cols)
    # print(crypte_attrs)
    # pd.options.display.max_columns = None
    # pd.options.display.max_rows = None
    # print(df_one_hot)
    # squash to one row
    # df_one_hot_sum = df_one_hot.sum(axis=0).to_frame().T
    # print(df_one_hot_sum)
    print(df_one_hot.shape)
    data = df_one_hot.to_numpy()
    # print(cdata)

    f = open("log_parallel.txt", "w")

    crypte = CrypteTest(crypte_attrs, epsilon)
    start = time.time()
    # crypte.insert(data)
    with Manager() as manager:

        L = manager.list()
        processes = []
        num_processes = multiprocessing.cpu_count() - 2

        if num_processes > len(data):
            num_processes = len(data)

        # print(cdata)
        split = np.array_split(data, num_processes)
        # print(split)

        for i in range(num_processes):
            # print(split[i])
            p = Process(target=crypte.insert_parallel, args=(L, split[i]))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        crypte.cdata.set_data(list(L))

    # print(crypte.a.data.db)

    elapsed = time.time() - start
    print(f"num_processes: {num_processes}")
    f.write(f"num_processes: {num_processes}\n")
    print(f"time inserting vector to crypte: {elapsed} s")
    f.write(f"time inserting vector to crypte: {elapsed} s\n")

    start = time.time()
    crypte.serialize("crypte_data_parallel.json")
    elapsed = time.time() - start
    print(f"time to serialize: {elapsed} s")
    f.write(f"time to serialize: {elapsed} s\n")

    f.close()


    # enc_data = crypte.deserialize("crypte_data_parallel.json")

    # mvec = [pro.lab_decrypt_vector(crypte.sk, v) for v in enc_data]
    # print(mvec)
    # print(data)
    # assert sorted(mvec) == sorted(data.tolist())

    # start = time.time()
    # res = crypte.count(attr_num=2, start=1, end=1, epsilon=0.1)
    # elapsed = time.time() - start
    # print(f"query time: {elapsed} s")
    #
    # start = time.time()
    # res = crypte.group_by(attr_num=3, epsilon=0.1)
    # elapsed = time.time() - start
    # print(f"query time: {elapsed} s")

    # print(res)