from sqlalchemy import create_engine
from pandas.io.sql import to_sql, read_sql
import pandas as pd
from find_epsilon import extract_table_names, get_metadata
import re
import snsql
from snsql import Privacy

if __name__ == '__main__':

    query_string = "SELECT AVG(age) FROM adult"
    df = pd.read_csv("../adult.csv")
    private = True
    eps = 0.1

    table_names = extract_table_names(query_string)
    table_name = table_names.pop()

    if not private:

        engine = create_engine("sqlite:///:memory:")

        sqlite_connection = engine.connect()

        num_rows = to_sql(df, name=table_name, con=sqlite_connection,
                          index=not any(name is None for name in df.index.names),
                          if_exists="replace")  # load index into db if all levels are named
        if num_rows != len(df):
            print("error when loading to sqlite")
            exit()

        original_result = read_sql(sql=query_string, con=sqlite_connection)

        print("original_result")
        print(original_result)

    else:

        metadata = get_metadata(df, table_name)

        query_string = re.sub(f" FROM {table_name}", f" FROM {table_name}.{table_name}", query_string, flags=re.IGNORECASE)

        privacy = Privacy(epsilon=eps)
        reader = snsql.from_df(df, privacy=privacy, metadata=metadata)

        private_result = reader.execute_df(query_string)

        print("private_result")
        print(private_result)
