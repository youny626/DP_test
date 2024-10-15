import string

import numpy as np
import pandas as pd

import os
import shutil
import time
import random

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append('/home/cc/DP_test/')

from privacy_budget.find_epsilon import * 

def clear_dir(dir_name):
    for filename in os.listdir(dir_name):
        file_path = os.path.join(dir_name, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


if __name__ == '__main__':

    num_runs_experiment = 10

    eps_list = list(np.arange(0.001, 0.01, 0.001, dtype=float))
    eps_list += list(np.arange(0.01, 0.1, 0.01, dtype=float))
    eps_list += list(np.arange(0.1, 1, 0.1, dtype=float))
    eps_list += list(np.arange(1, 11, 1, dtype=float))
    num_parallel_processes = 8
    percentage = 5
    gaussian = False
    svt_eps = 1
    variance_threshold = 10e-6

    # query_string = "SELECT COUNT(*) FROM adult WHERE income == '>50K' AND education_num == 13 AND age == 25"
    # query_string = "SELECT marital_status, COUNT(*) FROM adult WHERE race == 'Asian-Pac-Islander' AND age >= 30 AND
    # age <= 40 GROUP BY marital_status"
    # query_string = "SELECT COUNT(*) FROM adult WHERE native_country != 'United-States' AND sex == 'Female'"
    # query_string = "SELECT AVG(hours_per_week) FROM adult WHERE workclass == 'Federal-gov' OR workclass == 'Local-gov' or workclass == 'State-gov'"
    # query_string = "SELECT SUM(fnlwgt) FROM adult WHERE capital_gain > 0 AND income == '<=50K' AND occupation ==
    # 'Sales'"

    queries = ["SELECT COUNT(*) FROM adult WHERE income == '>50K' AND education_num == 13 AND age == 25",
               "SELECT marital_status, COUNT(*) FROM adult WHERE race == 'Asian-Pac-Islander' AND age >= 30 AND age <= 40 GROUP BY marital_status",
               "SELECT COUNT(*) FROM adult WHERE native_country != 'United-States' AND sex == 'Female'",
               "SELECT AVG(hours_per_week) FROM adult WHERE workclass == 'Federal-gov' OR workclass == 'Local-gov' or workclass == 'State-gov'",
               "SELECT SUM(capital_gain) FROM adult"
               ]

    root_dir = "/home/cc/DP_test/scalability"
    res_dir = "/home/cc/DP_test/scalability/result"

    data_list = ["adult_1000", "adult_10000", "adult_100000", "adult_1000000"]
    # data_list = ["adult_100000"]

    for svt in [True, False]:

        for data in data_list:

            data_path = f"{root_dir}/data/{data}.csv"        
            df = pd.read_csv(data_path)

            print("data:", data)

            time_list = []
            time_exclude_insert_db_list = []
            res_eps_list = []

            for i in range(num_runs_experiment):

                print("i:", i)

                times = []
                times_exclude_insert_db = []
                res_eps = []

                for sql_query in queries:

                    print("query:", sql_query)
                    
                    start_time = time.time()
                    best_eps, dp_result, insert_db_time = find_epsilon(df, sql_query, eps_list, 
                                    percentage=percentage,
                                    num_parallel_processes=num_parallel_processes, 
                                    gaussian=gaussian,
                                    svt=svt,
                                    svt_eps=svt_eps,
                                    variance_threshold=variance_threshold)
                    elapsed = time.time() - start_time
                    print(f"time: {elapsed} s")
                    times.append(elapsed)
                    times_exclude_insert_db.append(elapsed - insert_db_time)
                    res_eps.append(best_eps)

                    # if best_eps is not None:
                    #     print("best eps:", best_eps)
                    #     res_eps.append(best_eps)
                    # else:
                    #     print("can't find epsilon")
                    #     res_eps.append(None)

                time_list.append(times)
                time_exclude_insert_db_list.append(times_exclude_insert_db)
                res_eps_list.append(res_eps)

                svt_str = ""
                if svt:
                    svt_str = "svt_"
                with open(f"{res_dir}/{svt_str}{data}_time.log", "a") as f:
                    f.write(str(times)[1:-1] + "\n")

                with open(f"{res_dir}/{svt_str}{data}_time_exclude_insert_db.log", "a") as f:
                    f.write(str(times_exclude_insert_db)[1:-1] + "\n")

                with open(f"{res_dir}/{svt_str}{data}_eps.log", "a") as f:
                    f.write(str(res_eps)[1:-1] + "\n")

            # with open(f"{data}_time_full.log", "w") as f:
            #     for times in time_list:
            #         f.write(str(times)[1:-1] + "\n")

            # with open(f"{data}_time_exclude_insert_db_full.log", "w") as f:
            #     for times in time_exclude_insert_db_list:
            #         f.write(str(times)[1:-1] + "\n")

            # with open(f"{data}_eps_full.log", "w") as f:
            #     for res_eps in res_eps_list:
            #         f.write(str(res_eps)[1:-1] + "\n")
