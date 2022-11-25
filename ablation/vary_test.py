import pandas as pd
import numpy as np
from privacy_budget.find_epsilon import *

if __name__ == '__main__':
    df = pd.read_csv("./scalability/adult_100000.csv")

    num_runs_experiment = 10

    risk_group_percentile = -1
    risk_group_size = 100
    eps_list = list(np.arange(0.001, 0.01, 0.001, dtype=float))
    eps_list += list(np.arange(0.01, 0.1, 0.01, dtype=float))
    eps_list += list(np.arange(0.1, 1, 0.1, dtype=float))
    eps_list += list(np.arange(1, 11, 1, dtype=float))
    num_runs = 1
    q = 0.05

    queries = ["SELECT COUNT(*) FROM adult WHERE income == '>50K' AND education_num == 13 AND age == 25",
               "SELECT marital_status, COUNT(*) FROM adult WHERE race == 'Asian-Pac-Islander' AND age >= 30 AND age "
               "<= 40 GROUP BY marital_status",
               "SELECT COUNT(*) FROM adult WHERE native_country != 'United-States' AND sex == 'Female'",
               "SELECT AVG(hours_per_week) FROM adult WHERE workclass == 'Federal-gov' OR workclass == 'Local-gov' or "
               "workclass == 'State-gov'",
               "SELECT SUM(fnlwgt) FROM adult WHERE capital_gain > 0 AND income == '<=50K' AND occupation == 'Sales'"
               ]

    tests = ["ks", "es"]

    for test in tests:

        print("test:", test)

        res_eps_list = []
        time_list = []

        for i in range(num_runs_experiment):

            print("i:", i)

            res_eps = []
            times = []

            for query_string in queries:

                print(query_string)

                start_time = time.time()
                res = find_epsilon(df, query_string, -1, risk_group_size, eps_list, 1, 0.05, test=test)
                elapsed = time.time() - start_time
                print(f"total time: {elapsed} s")
                times.append(elapsed)

                best_eps = None
                if res is not None:
                    best_eps = res[0]

                print("eps:", best_eps)

                res_eps.append(best_eps)

            res_eps_list.append(res_eps)
            time_list.append(times)

        with open(f"test_{test}_eps.log", "w") as f:
            for eps in res_eps_list:
                f.write(str(eps)[1:-1] + "\n")

        with open(f"test_{test}_time.log", "w") as f:
            for times in time_list:
                f.write(str(times)[1:-1] + "\n")