import pandas as pd
import CDP
import LDP
from crypte_test import CrypteTest
import time
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode], prefix=feature_to_encode, prefix_sep='_')
    # print(dummies.head(10))
    # print(dummies.shape[1])
    num_cols = dummies.shape[1]
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    # print(res.head(10))
    return res, num_cols


def print_result(num_iters, true_answer, cdp_res, cdp_time, olh_res, olh_time, rr_res, rr_time, crypte_res,
                 crypte_time):
    print("Method: MAE, STD, Avg time(s)")
    print("CDP: ", mean_absolute_error(true_answer, cdp_res), np.std(cdp_res), cdp_time / num_iters)
    print("LDP(OLH): ", mean_absolute_error(true_answer, olh_res), np.std(olh_res), olh_time / num_iters)
    print("LDP(RR): ", mean_absolute_error(true_answer, rr_res), np.std(rr_res), rr_time / num_iters)
    print("Crypte: ", mean_absolute_error(true_answer, crypte_res), np.std(crypte_res), crypte_time / num_iters)


def plot_result(num_iters, true_answer, cdp_res, cdp_time, olh_res, olh_time, rr_res, rr_time, crypte_res,
                crypte_time, fig_name):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    x = ['CDP', 'LDP(OLH)', 'LDP(RR)', 'Crypte']
    x_pos = np.arange(len(x))

    mae = [mean_absolute_error(true_answer, cdp_res), mean_absolute_error(true_answer, olh_res),
           mean_absolute_error(true_answer, rr_res), mean_absolute_error(true_answer, crypte_res)]
    std = [np.std(cdp_res), np.std(olh_res), np.std(rr_res), np.std(crypte_res)]

    ax1.bar(x_pos, mae, yerr=std)
    # ax1.xlabel("Method")
    ax1.set_ylabel("MAE")
    # ax1.title("MAE")
    # ax1.xticks(x_pos, x)

    time = [cdp_time / num_iters, olh_time / num_iters, rr_time / num_iters, crypte_time / num_iters]
    ax2.bar(x_pos, time)
    ax2.set_xlabel("Method")
    ax2.set_ylabel("average time(s)")
    # ax2.title("MAE")
    plt.xticks(x_pos, x)

    fig.tight_layout()
    plt.savefig("./figures/" + fig_name)
    plt.show()
    plt.close()


if __name__ == '__main__':
    epsilon = 1.0
    num_iters = 10

    df = pd.read_csv("./PUMS_large.csv")
    df.drop(["Unnamed: 0", "state", "puma", "income", "latino", "black", "asian"], axis=1, inplace=True)
    # df.drop(["income", "pid"], axis=1, inplace=True)

    print(df.head(10))

    features_to_encode = ['age', 'sex', 'educ', 'married']#, 'race']
    df_one_hot = df.copy()
    crypte_attrs = []
    for feature in features_to_encode:
        df_one_hot, num_cols = encode_and_bind(df_one_hot, feature)
        crypte_attrs.append(num_cols)
    print(df_one_hot.head(10))
    df_one_hot_sum = df_one_hot.sum(axis=0).to_frame().T
    print(df_one_hot_sum)

    crypte = CrypteTest(crypte_attrs, epsilon)
    start = time.time()
    crypte.insert(df_one_hot_sum)
    elapsed = time.time() - start
    print("time inserting vector to crypte: " + str(elapsed))

    cdp_res, cdp_time = np.zeros(num_iters), 0
    olh_res, olh_time = np.zeros(num_iters), 0
    rr_res, rr_time = np.zeros(num_iters), 0
    crypte_res, crypte_time = np.zeros(num_iters), 0

    print("\nidentity: COUNT(age)")
    answer = len(df["age"])
    print("original: ", answer)
    true_answer = np.full(num_iters, answer)

    for i in range(num_iters):
        start = time.time()
        cdp_res[i] = CDP.count(df["age"], epsilon)
        cdp_time += (time.time() - start)

        start = time.time()
        olh = LDP.OLH(array=df["age"], epsilon=epsilon)
        olh_res[i] = olh.count()
        olh_time += (time.time() - start)

        start = time.time()
        rr = LDP.RR(array=df["age"], epsilon=epsilon)
        rr_res[i] = olh.count()
        rr_time += (time.time() - start)

        start = time.time()
        crypte_res[i] = crypte.count(attr_num=1, start=1, end=crypte_attrs[0])
        crypte_time += (time.time() - start)

        # print("CDP: ", CDP.count(df["age"], epsilon))
        # print("LDP: ", LDP.count(df["age"], epsilon))
        # print("Crypte: ", crypte.count(attr_num=1, start=1, end=crypte_attrs[0]))\

    print_result(num_iters, true_answer, cdp_res, cdp_time, olh_res, olh_time, rr_res, rr_time, crypte_res, crypte_time)
    plot_result(num_iters, true_answer, cdp_res, cdp_time, olh_res, olh_time, rr_res, rr_time, crypte_res,
                crypte_time, "identity")

    cdp_res, cdp_time = np.zeros(num_iters), 0
    olh_res, olh_time = np.zeros(num_iters), 0
    rr_res, rr_time = np.zeros(num_iters), 0
    crypte_res, crypte_time = np.zeros(num_iters), 0

    print("\nrange(over a single attribute): COUNT(20 <= age <= 30)")
    answer = len(df["age"][df["age"].between(20, 30)])
    print("original: ", answer)
    true_answer = np.full(num_iters, answer)

    for i in range(num_iters):
        start = time.time()
        cdp_res[i] = CDP.count(df["age"][df["age"].between(20, 30)], epsilon)
        cdp_time += (time.time() - start)

        start = time.time()
        olh = LDP.OLH(array=df["age"], epsilon=epsilon)
        olh_res[i] = olh.count_in_range(20, 30)
        olh_time += (time.time() - start)

        start = time.time()
        rr = LDP.RR(array=df["age"], epsilon=epsilon)
        rr_res[i] = olh.count_in_range(20, 30)
        rr_time += (time.time() - start)

        start = time.time()
        col_list = df_one_hot_sum.columns.tolist()
        start_pos = col_list.index("age_20") + 1
        end_pos = col_list.index("age_30") + 1
        crypte_res[i] = crypte.count(attr_num=1, start=start_pos, end=end_pos)
        crypte_time += (time.time() - start)

        # print("CDP: ", CDP.count(df["age"][df["age"].between(20, 30)], epsilon))
        # print("LDP: ", LDP.count_in_range(df["age"], 20, 30, epsilon))
        # print("crypte: ", crypte.count(attr_num=1, start=start_pos, end=end_pos)))

    print_result(num_iters, true_answer, cdp_res, cdp_time, olh_res, olh_time, rr_res, rr_time, crypte_res, crypte_time)
    plot_result(num_iters, true_answer, cdp_res, cdp_time, olh_res, olh_time, rr_res, rr_time, crypte_res,
                crypte_time, "range")