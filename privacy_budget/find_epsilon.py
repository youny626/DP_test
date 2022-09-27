import numpy as np
import pandas as pd
from scipy import stats
import tqdm
import time
import warnings
import snsql
from snsql import Privacy


def get_metadata(df: pd.DataFrame, name: str):
    metadata = {}
    metadata[name] = {}  # collection
    metadata[name][name] = {}  # schema
    metadata[name][name][name] = {}  # table

    t = metadata[name][name][name]
    t["row_privacy"] = True
    t["rows"] = len(df)

    df_inferred = df.convert_dtypes()
    # print(df_inferred.dtypes)

    for col in df_inferred:
        t[str(col)] = {}
        c = t[str(col)]
        if pd.api.types.is_string_dtype(df_inferred[col]):
            c["type"] = "string"
        elif pd.api.types.is_bool_dtype(df_inferred[col]):
            c["type"] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(df_inferred[col]):
            c["type"] = "datetime"
        # elif pd.api.types.is_numeric_dtype(df_inferred[col]):
        #     if (df[col].fillna(-9999) % 1  == 0).all():
        #         c["type"] = "int"
        #     else:
        #         c["type"] = "float"
        elif pd.api.types.is_integer_dtype(df_inferred[col]) or pd.api.types.is_timedelta64_dtype(df_inferred[col]):
            c["type"] = "int"
            c["upper"] = str(df_inferred[col].max())
            c["lower"] = str(df_inferred[col].min())
        elif pd.api.types.is_float_dtype(df_inferred[col]):
            c["type"] = "float"
            c["upper"] = str(df_inferred[col].max())
            c["lower"] = str(df_inferred[col].min())
        else:
            raise ValueError("Unknown column type for column {0}".format(col))

    return metadata


def fdr(p_values, q):
    # Based on https://matthew-brett.github.io/teaching/fdr.html

    # q = 0.1 # proportion of false positives we will accept
    N = len(p_values)
    sorted_p_values = np.sort(p_values)  # sort p-values ascended
    i = np.arange(1, N + 1)  # the 1-based i index of the p values, as in p(i)
    below = (sorted_p_values < (q * i / N))  # True where p(i)<q*i/N
    if len(np.where(below)[0]) == 0:
        return False
    max_below = np.max(np.where(below)[0])  # max index where p(i)<q*i/N
    # print('p*:', sorted_p_values[max_below])

    # num_discoveries = 0
    for p in p_values:
        if p < sorted_p_values[max_below]:
            # print("Discovery: " + str(p))
            return True
            # num_discoveries += 1
    # print("num_discoveries =", num_discoveries)
    return False


def find_epsilon(df: pd.DataFrame, data_name: str, query: str,
                 risk_group_percentile: int,
                 eps_to_test: list,
                 num_runs: int,
                 q: float):
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore")

        start_time = time.time()

        privacy = Privacy(epsilon=np.inf, delta=0)  # no privacy
        metadata = get_metadata(df, data_name)

        private_reader = snsql.from_df(df, metadata=metadata, privacy=privacy)
        original_result = private_reader.reader.execute_df(query)

        # Compute the original privacy risks
        original_risks = []
        dfs_one_off = []  # cache
        for i in range(len(df)):
            cur_df = df.copy()
            cur_df = cur_df.drop([i])
            dfs_one_off.append(cur_df)

            private_reader = snsql.from_df(cur_df, metadata=metadata, privacy=privacy)
            cur_result = private_reader.reader.execute_df(query) # execute query without DP

            change = abs(cur_result - original_result).to_numpy().sum()
            original_risks.append(change)

        # print(original_risks)
        elapsed = time.time() - start_time
        print(f"time to compute original risk: {elapsed} s")

        # Select groups we want to equalize risks
        # for now use the default strategy - high risk (upper x% quantile) and low risk (lower x% quantile) group
        sorted_original_risks = list(np.sort(original_risks))
        sorted_original_risks_idx = np.argsort(original_risks)
        idx1 = sorted_original_risks.index(
            np.percentile(sorted_original_risks, risk_group_percentile, interpolation='nearest'))
        idx2 = sorted_original_risks.index(
            np.percentile(sorted_original_risks, 100 - risk_group_percentile, interpolation='nearest'))
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        # print(idx1, idx2)

        # sample1 = sorted_original_risks[:idx1]
        # sample2 = sorted_original_risks[idx2:]
        sample_idx1 = sorted_original_risks_idx[:idx1]
        sample_idx2 = sorted_original_risks_idx[idx2:]

        # Check whether the samples come from the same distribution (null hypothesis)
        # Reject the null if p-value is less than some threshold
        # for now we don't care even if the samples already come from the same distribution

        # Now we test different epsilons
        # For each epsilon, we run n times, with the goal of finding the largest epsilon
        # that can constantly stay in null for all n runs

        best_eps = None

        # We start from the largest epsilon to the smallest
        sorted_eps_to_test = np.sort(eps_to_test)[::-1]
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # print(sorted_eps_to_test)
        # return None

        compute_risk_time = 0.0
        test_equal_distrbution_time = 0.0

        for eps in tqdm.tqdm(sorted_eps_to_test):

            p_values = []

            # For each run, we compute the risks again
            # New risk = DP result - DP result when removing one record

            privacy = Privacy(epsilon=eps)
            private_reader = snsql.from_df(df, metadata=metadata, privacy=privacy)
            dp_result = private_reader.execute_df(query)

            for j in tqdm.tqdm(range(num_runs)):

                start_time = time.time()

                # We get the same samples based on individuals who are present in the original chosen samples
                # we only need to compute risks for those samples
                new_risks1 = []
                new_risks2 = []

                for i in sample_idx1:
                    cur_df = dfs_one_off[i]

                    private_reader = snsql.from_df(cur_df, metadata=metadata, privacy=privacy)
                    cur_result = private_reader.execute_df(query)

                    change = abs(cur_result - dp_result).to_numpy().sum()
                    new_risks1.append(change)

                for i in sample_idx2:
                    cur_df = dfs_one_off[i]

                    private_reader = snsql.from_df(cur_df, metadata=metadata, privacy=privacy)
                    cur_result = private_reader.execute_df(query)

                    change = abs(cur_result - dp_result).to_numpy().sum()
                    new_risks2.append(change)

                elapsed = time.time() - start_time
                print(f"{j}th compute risk time: {elapsed} s")
                compute_risk_time += elapsed

                # We perform the test and record the p-value for each ru
                start_time = time.time()

                cur_res = stats.ks_2samp(new_risks1, new_risks2)
                p_value = cur_res[1]
                p_values.append(p_value)

                elapsed = time.time() - start_time
                test_equal_distrbution_time += elapsed

            # Now we have n p-values for the current epsilon, we use multiple comparisons technique
            # to determine if this epsilon is good enough
            # For now we use false discovery rate
            if not fdr(p_values, q):  # q = proportion of false positives we will accept
                # We want no discovery (fail to reject null) for all n runs
                # If we fail to reject the null, then we break the loop.
                # The current epsilon is the one we choose
                best_eps = eps
                break

        print(f"total time to compute new risk: {compute_risk_time} s")
        print(f"total time to test equal distribution: {test_equal_distrbution_time} s")

        return best_eps


if __name__ == '__main__':
    csv_path = 'PUMS.csv'
    df = pd.read_csv(csv_path).head(100)
    # print(df.head())

    query = "SELECT AVG(age) FROM PUMS.PUMS"

    # design epsilons to test in a way that smaller eps are more frequent and largest eps are less
    eps_list = list(np.arange(0.01, 0.1, 0.01, dtype=float))
    eps_list += list(np.arange(0.1, 1.1, 0.1, dtype=float))
    # eps_list += list(np.arange(1, 11, 1, dtype=float))
    print(eps_list)

    start_time = time.time()
    eps = find_epsilon(df, "PUMS", query, 90, eps_list, 10, 0.05)
    elapsed = time.time() - start_time
    print(f"total time: {elapsed} s")

    print(eps)

# Successfully installed 
# PyYAML==5.4.1 graphviz==0.17 numpy==1.23.3 opendp==0.4.0 pandas==1.5.0 pandasql==0.7.3 pytz==2022.2.1
# sqlalchemy==1.4.41
# PyYAML==5.4.1 antlr4-python3-runtime==4.9.3 graphviz==0.17 numpy==1.23.3 opendp==0.4.0 pandas==1.5.0
# pandasql==0.7.3 python==dateutil==2.8.2 pytz==2022.2.1 six==1.16.0 smartnoise-sql==0.2.4 sqlalchemy==1.4.41
