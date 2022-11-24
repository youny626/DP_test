from sqlalchemy import create_engine
from pandas.io.sql import to_sql, read_sql
import pandas as pd
from find_epsilon import extract_table_names, get_metadata
import re
import snsql
from snsql import Privacy
import time

def private_query(df, query_string, eps):

    table_names = extract_table_names(query_string)
    table_name = table_names.pop()

    metadata = get_metadata(df, table_name)

    query_string = re.sub(f" FROM {table_name}", f" FROM {table_name}.{table_name}", query_string, flags=re.IGNORECASE)

    privacy = Privacy(epsilon=eps)
    reader = snsql.from_df(df, privacy=privacy, metadata=metadata)

    private_result = reader.execute_df(query_string)

    return private_result

def query(df, query_string):

    table_names = extract_table_names(query_string)
    table_name = table_names.pop()

    engine = create_engine("sqlite:///:memory:")

    sqlite_connection = engine.connect()

    num_rows = to_sql(df, name=table_name, con=sqlite_connection,
                      index=not any(name is None for name in df.index.names),
                      if_exists="replace")  # load index into db if all levels are named
    if num_rows != len(df):
        print("error when loading to sqlite")
        exit()

    original_result = read_sql(sql=query_string, con=sqlite_connection)

    sqlite_connection.close()

    return original_result

if __name__ == '__main__':

    df = pd.read_csv("/home/cc/DP_test/adult_age_race_sex_income.csv")

    num_runs = 10

    query1_time_list_p = []
    query2_time_list_p = []
    query3_time_list_p = []

    query1_time_list = []
    query2_time_list = []
    query3_time_list = []

    for i in range(num_runs):

        query_string = "SELECT COUNT(*) AS cnt FROM adult WHERE age >= 20 AND age <= 30"

        start_time = time.time()
        res = private_query(df, query_string, 0.1)
        elapsed = time.time() - start_time
        query1_time_list_p.append(elapsed)

        start_time = time.time()
        res = query(df, query_string)
        elapsed = time.time() - start_time
        query1_time_list.append(elapsed)

        query_string = "SELECT COUNT(*) AS cnt FROM adult GROUP BY race"

        start_time = time.time()
        res = private_query(df, query_string, 0.1)
        elapsed = time.time() - start_time
        query2_time_list_p.append(elapsed)

        start_time = time.time()
        res = query(df, query_string)
        elapsed = time.time() - start_time
        query2_time_list.append(elapsed)

        query_string = "SELECT COUNT(*) AS cnt FROM adult WHERE sex == 'Female' AND income == '>50K'"

        start_time = time.time()
        res = private_query(df, query_string, 0.1)
        elapsed = time.time() - start_time
        query3_time_list_p.append(elapsed)

        start_time = time.time()
        res = query(df, query_string)
        elapsed = time.time() - start_time
        query3_time_list.append(elapsed)

    with open("log_no_dp.txt", "w") as f:
        f.write(str(query1_time_list)[1:-1] + "\n")
        f.write(str(query2_time_list)[1:-1] + "\n")
        f.write(str(query3_time_list)[1:-1] + "\n")

    with open("log_dp.txt", "w") as f:
        f.write(str(query1_time_list_p)[1:-1] + "\n")
        f.write(str(query2_time_list_p)[1:-1] + "\n")
        f.write(str(query3_time_list_p)[1:-1] + "\n")

