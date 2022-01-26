import unittest
import numpy as np
import random
import resource
import time
import phe.paillier as paillier
from dataclasses import dataclass
import util
import json
import provision as pro
from crypte import Crypte
import pandas as pd
import time

if __name__ == '__main__':
    random_list = list(np.random.randint(low=1,high=6,size=10))
    # print(random_list[:10])
    dummy = pd.get_dummies(random_list)
    # print(dummy.head(10))
    # print(dummy.head(10).values.tolist())
    dummy_list = dummy.values.tolist()
    # print(random_list)
    # print(dummy_list)

    start = time.time()
    db = Crypte(attr=[dummy.shape[1]])
    pubkey, prikey = paillier.generate_paillier_keypair()

    for v in dummy_list:
        enc_v = pro.lab_encrypt_vector(pubkey, v)
        # print(enc_v)
        db.insert(enc_v)

    elapsed = (time.time() - start)
    print("time inserting vector: " + str(elapsed))
    start = time.time()

    encfilter = db.filter(1, 2, 3)
    # print(encfilter)
    encnt = db.count(encfilter)
    # print(encnt)
    answer = pro.lab_decrypt(prikey, encnt)
    print(answer)
    true_answer = dummy.iloc[:, 2].sum() + dummy.iloc[:, 1].sum()
    print(true_answer)
    assert answer == true_answer

    sensitivity = 1
    epsilon = 1.0
    laplace_noise = np.random.laplace(loc=0, scale=2 * sensitivity / epsilon)
    # print(laplace_noise)
    encrypted_noise = pro.lab_encrypt(pubkey, laplace_noise)
    # print(encrypted_noise)
    encnt = pro.lab_add(encnt, encrypted_noise)
    # print(encnt)

    # CSP checks whether the privacy budget is exceeded

    m = pro.lab_decrypt(prikey, encnt)
    # print(m)

    result = m + laplace_noise
    print("result = " + str(result))

    elapsed = (time.time() - start)
    print("time computing: " + str(elapsed))


