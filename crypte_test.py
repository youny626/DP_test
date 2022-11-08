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

        self.analytics_server = AS()
        self.analytics_server.set_key(self.pk)
        self.analytics_server.load_data(self.cdata)

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

    def count_filter(self, attr_num, start, end, epsilon):

        def _count_filter(obj, attr_pred, val_pred_1, val_pred_2):
            encfilter = cte.filter(obj, attr_pred, val_pred_1, val_pred_2)
            encnt = encfilter.count()
            return encnt

        c = self.analytics_server.execute(_count_filter, attr_num, start, end)
        print("\n")
        print("True counting output is", self.cs.reveal_clear(c))

        sens = 1
        c = self.analytics_server.laplace_distort(sens, epsilon, 1, c)
        noisy_res = self.cs.reveal_noisy(c, sens, epsilon)
        noisy_res = max(0, noisy_res)
        print("DP counting output is", noisy_res)

        return noisy_res

    def group_by(self, attr_num, epsilon):

        def _group_by(obj, attr_pred):
            encnt = obj.group_by(attr_pred)
            return encnt

        db_copy = self.cdata.get_data().copy()
        attr_copy = self.cdata.get_attr().copy()

        c = self.analytics_server.execute(_group_by, attr_num)
        print("\n")
        print("True group by output is", self.cs.reveal_clear_vector(c))
        sens = 1
        # print(self.attributes, attr_num-1)
        # print(c)
        c = self.analytics_server.laplace_distort_vector(sens, epsilon, self.attributes[attr_num - 1], c)
        noisy_res = self.cs.reveal_noisy_vector(c, sens, epsilon, self.attributes[attr_num-1])
        noisy_res = [max(0, x) for x in noisy_res]
        print("DP group by output is", noisy_res)

        self.cdata.set_data(db_copy)
        self.cdata.set_attr(attr_copy)

        return noisy_res

    def cross_product_filter(self, attr1, attr2, start, end, epsilon):

        def _cross_product(obj, attr_1, attr_2, pk):
            cross_product = cte.cosprod(obj, attr_1, attr_2, pk)
            return cross_product

        def _count_filter(obj, attr_pred, val_pred_1, val_pred_2):
            encfilter = cte.filter(obj, attr_pred, val_pred_1, val_pred_2)
            encnt = encfilter.count()
            return encnt

        cross_product = self.analytics_server.execute(_cross_product, attr1, attr2, self.pk)

        # need to re-encrypt the cross product
        cross_product.set_data(self.cs.re_encrypt_mult(cross_product.get_data()))

        a = AS()
        a.set_key(self.pk)
        a.load_data(cross_product)  # load_data will change the current cdata to the cdata to load

        encrypted_count = a.execute(_count_filter, 1, start, end)
        print("True counting output is", self.cs.reveal_clear(encrypted_count))

        sens = 1
        noisy_res = a.laplace_distort(sens, epsilon, 1, encrypted_count)
        print("DP counting output is", self.cs.reveal_noisy(noisy_res, sens, epsilon))

    def _group_by_filter(self, group_by_attr_num, filter_attr_num, start, end, epsilon):
        # FIXME: does not work

        db_copy = self.cdata.get_data().copy()
        attr_copy = self.cdata.get_attr().copy()

        def test_group_by_filter(obj, group_by_attr_num, filter_attr_num, start, end):
            encfilter = cte.filter(obj, filter_attr_num, start, end)
            # print(encfilter.get_attr())
            # print(self.cs.reveal_clear_vector(encfilter.get_data()[0]))
            # print(self.cs.reveal_clear_vector(encfilter.get_data()[1]))
            # print(self.cs.reveal_clear_vector(encfilter.get_data()[2]))
            encnt = encfilter.group_by(group_by_attr_num)
            return encnt

        c = self.analytics_server.execute(test_group_by_filter, group_by_attr_num, filter_attr_num, start, end)
        print("\n")
        print("True group by output is", self.cs.reveal_clear_vector(c))
        sens = 1
        # print(self.attributes, attr_num-1)
        # print(c)
        c = self.analytics_server.laplace_distort_vector(sens, epsilon, self.attributes[group_by_attr_num - 1], c)
        noisy_res = self.cs.reveal_noisy_vector(c, sens, epsilon, self.attributes[group_by_attr_num-1])
        noisy_res = [max(0, x) for x in noisy_res]
        print("DP group by output is", noisy_res)

        self.cdata.set_data(db_copy)
        self.cdata.set_attr(attr_copy)

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
        self.analytics_server.load_data(self.cdata)
        self.analytics_server.set_key(self.pk)

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
    # df = df.sample(3)

    # df = df[["age"]]
    # print(df.head(10))

    features_to_encode = ["age", "race", "sex", "income"]
    df_one_hot = df[features_to_encode].copy()
    crypte_attrs = []
    for feature in features_to_encode:
        df_one_hot, num_cols = encode_and_bind(df_one_hot, feature)
        crypte_attrs.append(num_cols)
    # print(crypte_attrs)
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    # print(df_one_hot)
    # squash to one row
    # df_one_hot_sum = df_one_hot.sum(axis=0).to_frame().T
    # print(df_one_hot_sum)
    print(df_one_hot.shape)
    # print(df_one_hot.columns)

    # exit()
    # print(df_one_hot)
    data = df_one_hot.to_numpy(dtype=int).tolist()
    # print(cdata)

    # f = open("log_parallel.txt", "w")

    crypte = CrypteTest(crypte_attrs, epsilon)
    # start = time.time()
    # crypte.insert(data)
    # with Manager() as manager:
    #
    #     L = manager.list()
    #     processes = []
    #     num_processes = multiprocessing.cpu_count() - 2
    #
    #     if num_processes > len(data):
    #         num_processes = len(data)
    #
    #     # print(cdata)
    #     split = np.array_split(data, num_processes)
    #     # print(split)
    #
    #     for i in range(num_processes):
    #         # print(split[i])
    #         p = Process(target=crypte.insert_parallel, args=(L, split[i]))
    #         p.start()
    #         processes.append(p)
    #
    #     for p in processes:
    #         p.join()
    #
    #     crypte.cdata.set_data(list(L))
    #
    # # print(crypte.analytics_server.data.db)
    #
    # elapsed = time.time() - start
    # print(f"num_processes: {num_processes}")
    # f.write(f"num_processes: {num_processes}\n")
    # print(f"time inserting vector to crypte: {elapsed} s")
    # f.write(f"time inserting vector to crypte: {elapsed} s\n")
    #
    # start = time.time()
    # crypte.serialize("crypte_data_parallel.json")
    # elapsed = time.time() - start
    # print(f"time to serialize: {elapsed} s")
    # f.write(f"time to serialize: {elapsed} s\n")
    #
    # f.close()

    start = time.time()
    enc_data = crypte.deserialize("crypte_data_parallel.json")
    elapsed = time.time() - start
    print(f"time to deserialize: {elapsed} s")

    # mvec = [pro.lab_decrypt_vector(crypte.sk, v) for v in enc_data]
    # print(mvec)
    # print(data)
    # assert sorted(mvec) == sorted(data.tolist())

    # print(crypte.analytics_server.data.get_attr())

    start = time.time()
    res = crypte.count_filter(attr_num=1, start=4, end=14, epsilon=0.1)
    elapsed = time.time() - start
    print(f"query1 time: {elapsed} s")

    # print(crypte.analytics_server.data.get_attr())

    start = time.time()
    res = crypte.group_by(attr_num=2, epsilon=0.1)
    elapsed = time.time() - start
    print(f"query2 time: {elapsed} s")

    start = time.time()
    res = crypte.cross_product_filter(attr1=3, attr2=4, start=2, end=2, epsilon=0.1)
    elapsed = time.time() - start
    print(f"query3 time: {elapsed} s")

    start = time.time()
    res = crypte.cross_product_filter(attr1=1, attr2=4, start=77, end=87, epsilon=0.1)
    elapsed = time.time() - start
    print(f"query4 time: {elapsed} s")

    # print(crypte.analytics_server.data.get_attr())

    # start = time.time()
    # res = crypte.group_by_filter(group_by_attr_num=3, filter_attr_num=4, start=1, end=1, epsilon=0.1)
    # elapsed = time.time() - start
    # print(f"query time: {elapsed} s")

