import pandas as pd
import CDP
import LDP
from crypte_test import CrypteTest
import time

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
    df = pd.read_csv("./PUMS.csv")
    df.drop(["income", "pid"], axis=1, inplace=True)
    df = df.head(10)
    print(df.head(10))
    features_to_encode = ['age', 'sex', 'educ', 'race', 'married']
    df_one_hot = df.copy()
    crypte_attrs = []
    for feature in features_to_encode:
        df_one_hot, num_cols = encode_and_bind(df_one_hot, feature)
        crypte_attrs.append(num_cols)
    print(df_one_hot.head(10))

    epsilon = 1.0

    crypte = CrypteTest(crypte_attrs, epsilon)
    start = time.time()
    crypte.insert(df_one_hot)
    elapsed = time.time() - start
    print("time inserting vector to crypte: " + str(elapsed))

    print("identity: COUNT(age)")

    print("original: ", len(df["age"]))
    print("CDP: ", CDP.count(df["age"], epsilon))
    print("LDP: ", LDP.count(df["age"], epsilon))
    print("Crypte: ", crypte.count(attr_num=1, start=1, end=crypte_attrs[0]))

    print("range(over a single attribute): COUNT(20 <= age <= 30)")

    print("original: ", len(df["age"][df["age"].between(20, 30)]))
    print("CDP: ", CDP.count(df["age"][df["age"].between(20, 30)], epsilon))
    print("LDP: ", LDP.count_in_range(df["age"], 20, 30, epsilon))



