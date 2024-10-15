import pandas as pd
import numpy as np
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append('/Users/zhiruzhu/Desktop/dp_paper/DP_test')

from privacy_budget.find_epsilon import *
 
if __name__ == '__main__':
    df = pd.read_csv("/Users/zhiruzhu/Desktop/dp_paper/DP_test/scalability/data/adult_100000.csv")
    res_dir = "/Users/zhiruzhu/Desktop/dp_paper/DP_test/ablation/vary_percentile/svt_res"

    num_runs_experiment = 50

    eps_list = list(np.arange(0.001, 0.01, 0.001, dtype=float))
    eps_list += list(np.arange(0.01, 0.1, 0.01, dtype=float))
    eps_list += list(np.arange(0.1, 1, 0.1, dtype=float))
    eps_list += list(np.arange(1, 11, 1, dtype=float))
    num_parallel_processes = 8

    queries = ["SELECT COUNT(*) FROM adult WHERE income == '>50K' AND education_num == 13 AND age == 25",
               "SELECT marital_status, COUNT(*) FROM adult WHERE race == 'Asian-Pac-Islander' AND age >= 30 AND age "
               "<= 40 GROUP BY marital_status",
               "SELECT COUNT(*) FROM adult WHERE native_country != 'United-States' AND sex == 'Female'",
               "SELECT AVG(hours_per_week) FROM adult WHERE workclass == 'Federal-gov' OR workclass == 'Local-gov' or "
               "workclass == 'State-gov'",
               "SELECT SUM(capital_gain) FROM adult"
               ]

    for svt_eps in [0.01, 0.1, 1, 10]:

        print("svt_eps:", svt_eps)

        res_eps_list = []
        time_list = []

        for i in range(num_runs_experiment):

            print("i:", i)

            res_eps = []
            times = []

            for query_string in queries:


                print(query_string)

                start_time = time.time()
                res = find_epsilon(df, query_string, eps_list, 
                                num_parallel_processes=num_parallel_processes, 
                                svt=True,
                                svt_eps=svt_eps)                
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


        with open(f"svt_eps_{svt_eps}_eps.log", "w") as f:
            for eps in res_eps_list:
                f.write(str(eps)[1:-1] + "\n")

        with open(f"svt_eps_{svt_eps}_time.log", "w") as f:
            for times in time_list:
                f.write(str(times)[1:-1] + "\n")