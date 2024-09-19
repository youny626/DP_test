import itertools
import math
import random
import re

import numpy as np
import pandas as pd
from scipy import stats
import tqdm
import time
import warnings
import snsql
from snsql import Privacy
from snsql._ast.expressions.date import parse_datetime
from snsql._ast.expressions import sql as ast
from snsql.sql.reader.base import SortKey
from snsql._ast.ast import Top
from sqlalchemy import create_engine
from pandas.io.sql import to_sql, read_sql
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
from sql_metadata import Parser
from sklearn import preprocessing
# import multiprocessing as mp
import pathos.multiprocessing as mp
from pathos.multiprocessing import ProcessPool
from collections import defaultdict
from scipy.stats import laplace

random_state = 0


def get_metadata(df: pd.DataFrame, name: str):
    metadata = {}
    metadata[name] = {}  # collection
    metadata[name][name] = {}  # db
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
    rejected, pvalue_corrected = fdrcorrection(p_values, alpha=q)
    # print(any(rejected))
    return any(rejected)

    # # Based on https://matthew-brett.github.io/teaching/fdr.html
    #
    # # q = 0.1 # proportion of false positives we will accept
    # N = len(p_values)
    # sorted_p_values = np.sort(p_values)  # sort p-values ascended
    # i = np.arange(1, N + 1)  # the 1-based i index of the p values, as in p(i)
    # below = (sorted_p_values < (q * i / N))  # True where p(i)<q*i/N
    # if len(np.where(below)[0]) == 0:
    #     return False
    # max_below = np.max(np.where(below)[0])  # max index where p(i)<q*i/N
    # print('p*:', sorted_p_values[max_below])
    #
    # # num_discoveries = 0
    # for p in p_values:
    #     if p < sorted_p_values[max_below]:
    #         print("Discovery: " + str(p))
    #         return True
    #         # num_discoveries += 1
    # # print("num_discoveries =", num_discoveries)
    # return False


def compute_exact_aggregates_of_neighboring_data(df, table_name, query_string_in, row_num_col, idx_to_compute,
                                                 private_reader, subquery, query, table_metadata):
    res = []

    reader = private_reader._get_reader(subquery)
    # if isinstance(query, str):
    #     raise ValueError("Please pass ASTs to execute_ast.  To execute strings, use execute.")
    if hasattr(reader, "serializer") and reader.serializer is not None:
        query_string = reader.serializer.serialize(subquery)
    else:
        query_string = str(subquery)

    has_avg = re.search(" AVG\(.*\) ", query_string_in, re.IGNORECASE)
    if has_avg:
        query_string = query_string_in

    query_string = re.sub(f" FROM {table_name}.{table_name}", f" FROM {table_name}", query_string,
                          flags=re.IGNORECASE)

    engine = create_engine("sqlite:///:memory:")

    with engine.connect() as conn:

        # df_with_dummy_row = df.append(pd.Series(), ignore_index=True)
        dummy_row = df.iloc[0].copy()
        for col in df.columns:
            if col in table_metadata.keys():
                if table_metadata[col]["type"] == "int" or table_metadata[col]["type"] == "float":
                    dummy_row[col] = table_metadata[col]["lower"]
                elif table_metadata[col]["type"] == "string":
                    dummy_row[col] = None
            else:
                dummy_row[col] = len(df) + 1

        # print(dummy_row)
        df_with_dummy_row = df.append(dummy_row)

        num_rows = to_sql(df_with_dummy_row, name=table_name, con=conn,
                          # index=not any(name is None for name in df.index.names),
                          if_exists="replace")  # load index into db if all levels are named
        if num_rows != len(df_with_dummy_row):
            print("error when loading to sqlite")
            return None

        for row_num_to_exclude in idx_to_compute:

            pos = query_string.rfind(" WHERE ")  # last pos of WHERE
            if pos == -1:
                pos2 = query_string.rfind(f" FROM {table_name}")
                pos2 += len(f" FROM {table_name}")
                cur_query_string = query_string[:pos2] + f" WHERE {row_num_col} != {row_num_to_exclude} " + query_string[
                                                                                                        pos2:]
            else:
                pos += len(" WHERE ")
                cur_query_string = query_string[:pos] + f"{row_num_col} != {row_num_to_exclude} AND " + query_string[pos:]

            q_result = read_sql(sql=cur_query_string, con=conn)
            # print(q_result)
            exact_aggregates = [val[1:] for val in q_result.itertuples()]
            # print(exact_aggregates)
            res.append(exact_aggregates)

    return res


def compute_noise_prob_using_neighboring_data(neighboring_exact_aggregates, dp_aggregates, exact_aggregates,
                                              private_reader, subquery,
                                              query, query_string_in):

    if len(neighboring_exact_aggregates) != len(exact_aggregates[1:]):
        # corner case: group by results missing for one group
        # print(neighboring_exact_aggregates)
        # print(exact_aggregates)
        missing_group = (set(exact_aggregates[1:]) - set(neighboring_exact_aggregates)).pop()
        missing_group_pos = exact_aggregates[1:].index(missing_group)
        missing_group = list(missing_group)
        kc_pos = exact_aggregates[0].index('keycount')
        missing_group[kc_pos] = 0
        neighboring_exact_aggregates.insert(missing_group_pos, missing_group)
        # print(neighboring_exact_aggregates)

    has_avg = re.search(" AVG\(.*\) ", query_string_in, re.IGNORECASE)
    has_sum = re.search(" SUM\(.*\) ", query_string_in, re.IGNORECASE)
    has_count = re.search(" COUNT\(.*\) ", query_string_in, re.IGNORECASE)

    mechs = private_reader._get_mechanisms(subquery)
    check_sens = [m for m in mechs if m]
    if any([m.sensitivity is np.inf for m in check_sens]):
        raise ValueError(f"Attempting to query an unbounded column")

    kc_pos = private_reader._get_keycount_position(subquery)

    pdfs = []

    if has_avg:

        # prob = 1.0

        # laplace_rv = laplace(scale=avg_scale)

        for row, row_dp in zip(neighboring_exact_aggregates, dp_aggregates):
            row = [v for v in row]
            # set null to 0 before adding noise
            for idx in range(len(row)):
                if row[idx] is None:
                    row[idx] = 0.0

            num_agg_computed = 0
            
            row_pdf = []
            
            for v, v_dp in zip(row, row_dp):

                if not isinstance(v, str):
                    num_agg_computed += 1
                    if num_agg_computed > 1:
                        print("If a query contains AVG, we cannot compute other aggregates")
                        return None

                    # laplace.pdf(x, loc, scale) is identically equivalent to laplace.pdf(y) / scale with y = (x - loc) / scale.

                    laplace_val = v_dp - v
                    # pdf = np.exp(-abs(laplace_val) / avg_scale) / (2. * avg_scale)
                    # print(laplace_val, pdf)
                    cur_pdf = laplace.pdf(laplace_val, scale=avg_scale)
                    print(laplace_val, avg_scale, cur_pdf)
                    row_pdf.append(cur_pdf)
                    
                    # print(laplace.pdf(0, scale=0.5))
                    # print(avg_scale, pdf)
                    
            pdfs.append(row_pdf)

        # print("\n")
        # print(prob)

        # if prob < 0.001:
        #     print(neighboring_exact_aggregates)
        #     print(prob)

        return pdfs


    # prob = 1.0
    for row, row_dp in zip(neighboring_exact_aggregates, dp_aggregates):
        row = [v for v in row]

        # set null to 0 before adding noise
        for idx in range(len(row)):
            if mechs[idx] and row[idx] is None:
                row[idx] = 0.0

        # print(row)
        # print(row_dp)
        idx = 0
        row_pdf = []
        for mech, v, v_dp in zip(mechs, row, row_dp):
            if mech is not None:
                # print(v, v_dp)
                laplace_val = v_dp - v
                # print(mech.scale, mech.sensitivity/mech.epsilon)
                if not (has_sum and not has_count and idx == kc_pos):
                    cur_pdf = laplace.pdf(x=laplace_val, scale=mech.scale)
                    row_pdf.append(cur_pdf)
            idx += 1
    # print(prob)
    
        pdfs.append(row_pdf)

    return pdfs


def execute_rewritten_ast(sqlite_connection, table_name, query_string_in, table_metadata,
                          private_reader, subquery, query, *ignore, accuracy: bool = False, pre_aggregated=None,
                          postprocess=True):
    # if isinstance(query, str):
    #     raise ValueError("Please pass AST to _execute_ast.")
    #
    # subquery, query = self._rewrite_ast(query)

    has_avg = re.search(" AVG\(.*\) ", query_string_in, re.IGNORECASE)
    has_sum = re.search(" SUM\(.*\) ", query_string_in, re.IGNORECASE)
    has_count = re.search(" COUNT\(.*\) ", query_string_in, re.IGNORECASE)

    if pre_aggregated is not None:
        exact_aggregates = private_reader._check_pre_aggregated_columns(pre_aggregated, subquery)
    else:        # has_avg = False

        # exact_aggregates = private_reader._get_reader(subquery)._execute_ast(subquery)
        reader = private_reader._get_reader(subquery)
        # if isinstance(query, str):
        #     raise ValueError("Please pass ASTs to execute_ast.  To execute strings, use execute.")
        if hasattr(reader, "serializer") and reader.serializer is not None:
            query_string = reader.serializer.serialize(subquery)
        else:
            query_string = str(subquery)
        # exact_aggregates = reader.execute(query_string, accuracy=accuracy)

        # print(read_sql(sql=f"SELECT * FROM {table_name}", con=sqlite_connection))

        if has_avg:
            # query_string = str(query)
            query_string = query_string_in

        # print(query_string)
        query_string = re.sub(f" FROM {table_name}.{table_name}", f" FROM {table_name}", query_string,
                              flags=re.IGNORECASE)

        # print(query_string)
        q_result = read_sql(sql=query_string, con=sqlite_connection)
        # print(q_result)
        exact_aggregates = [tuple([col for col in q_result.columns])] + [val[1:] for val in q_result.itertuples()]
        print("exact_aggregates", exact_aggregates)

        # print(subquery)
        # print(query)

    _accuracy = None
    if accuracy:
        raise NotImplementedError(
            "Simple accuracy has been removed.  Please see documentation for information on estimating accuracy.")

    syms = subquery._select_symbols
    source_col_names = [s.name for s in syms]

    pdfs = []

    if has_avg:

        query_string_in_parser = Parser(query_string_in)
        select_cols = query_string_in_parser.columns_dict["select"]
        # print(query_string_in_parser.columns_aliases_names)
        # print(query_string_in_parser.columns_aliases)

        from sqlglot import parse_one, exp


        select_query = parse_one(str(query)).find(exp.Select)
        projections = select_query.expressions
        # for projection in projections:
        #     print(projection)

        # print(select_cols)

        # query_parser = Parser(str(query))
        # select_cols_query = query_parser.columns_dict["select"]
        # print(str(query))

        # assert len(select_cols) == len(select_cols_query)

        global avg_scale
        avg_scale = None
        for i in range(len(select_cols)):
            col_name = select_cols[i]
            # print(projections[i])
            if f"sum_{col_name} / keycount" in str(projections[i]):
                # this is the avg col
                # print(table_metadata[col_name])
                avg_sensitivity = (float(table_metadata[col_name]["upper"]) - float(table_metadata[col_name]["lower"])) / \
                                  table_metadata["rows"]
                # print(col_name)
                # print(avg_sensitivity)
                # print(private_reader.privacy.epsilon)
                avg_scale = avg_sensitivity / private_reader.privacy.epsilon

        if avg_scale is None:
            print("Can't compute sensitivity of the AVG col")
            return None

        # prob = 1.0
        dp_aggregates = []

        for row in exact_aggregates[1:]:
            row = [v for v in row]
            # set null to 0 before adding noise
            for idx in range(len(row)):
                if row[idx] is None:
                    row[idx] = 0.0

            res = []
            num_agg_computed = 0

            row_pdf = []

            for i in range(len(row)):
                if not isinstance(row[i], str):
                    num_agg_computed += 1
                    if num_agg_computed > 1:
                        print("If a query contains AVG, we cannot compute other aggregates")
                        return None

                    laplace_val = laplace.rvs(scale=avg_scale)
                    # print(laplace_val + v)
                    cur_pdf = laplace.pdf(x=laplace_val, scale=avg_scale)
                    row_pdf.append(cur_pdf)
                    # print(laplace_val, avg_scale, cur_pdf)
                    res.append(row[i] + laplace_val)
                else:
                    res.append(row[i])

            pdfs.append(row_pdf)

            dp_aggregates.append(res)

        

        # # get column information for outer query
        # out_syms = query._select_symbols
        # out_types = [s.expression.type() for s in out_syms]
        # out_col_names = [s.name for s in out_syms]

        row0 = [exact_aggregates[0]]
        out_rows = row0 + list(dp_aggregates)
        # print(prob)
        return out_rows, pdfs, dp_aggregates, exact_aggregates

    # tell which are counts, in column order
    is_count = [s.expression.is_count for s in syms]

    # get a list of mechanisms in column order
    mechs = private_reader._get_mechanisms(subquery)
    check_sens = [m for m in mechs if m]
    if any([m.sensitivity is np.inf for m in check_sens]):
        raise ValueError(f"Attempting to query an unbounded column")

    kc_pos = private_reader._get_keycount_position(subquery)
    # print(kc_pos)

    # print(is_count)

    # print(query)
    # print(subquery)

    def randomize_row_values(row_in):
        row = [v for v in row_in]
        # set null to 0 before adding noise
        for idx in range(len(row)):
            if mechs[idx] and row[idx] is None:
                row[idx] = 0.0
        # call all mechanisms to add noise
        # print(row)
        # prob = 1.0
        row_pdf = []
        res = []
        idx = 0
        for mech, v in zip(mechs, row):
            if mech is not None:

                # print(v)
                # print(mech.sensitivity)
                # print(mech.release([v])[0])
                laplace_val = laplace.rvs(scale=mech.scale)
                # print(laplace_val + v)
                if not (has_sum and not has_count and idx == kc_pos):
                    # print(mech.sensitivity)
                    # print(mech.epsilon)
                    cur_pdf = laplace.pdf(x=laplace_val, scale=mech.scale)
                    row_pdf.append(cur_pdf)
                    # print(prob)
                res.append(v + laplace_val)
            else:
                res.append(v)
            idx += 1

        res.append(row_pdf)
        return res
        # return [
        #     mech.release([v])[0] if mech is not None else v
        #     for mech, v in zip(mechs, row)
        # ]

    if hasattr(exact_aggregates, "rdd"):
        # it's a dataframe
        out = exact_aggregates.rdd.map(randomize_row_values)
    elif hasattr(exact_aggregates, "map"):
        # it's an RDD
        out = exact_aggregates.map(randomize_row_values)
    elif isinstance(exact_aggregates, list):
        out = map(randomize_row_values, exact_aggregates[1:])
    elif isinstance(exact_aggregates, np.ndarray):
        out = map(randomize_row_values, exact_aggregates)
    else:
        raise ValueError("Unexpected type for exact_aggregates")

    # print(list(out))
    # prob = 1.0
    dp_aggregates = []
    for lst in list(out):
        dp_aggregates.append(lst[:-1])
        # prob *= lst[-1]
        pdfs.append(lst[-1])
    out = dp_aggregates.copy()
    # print(out)
    # print(prob)

    # censor infrequent dimensions
    # if private_reader._options.censor_dims:
    #     if kc_pos is None:
    #         raise ValueError("Query needs a key count column to censor dimensions")
    #     else:
    #         thresh_mech = mechs[kc_pos]
    #         private_reader.tau = thresh_mech.threshold
    #         # print("xxx")
    #     if hasattr(out, "filter"):
    #         # it's an RDD
    #         tau = private_reader.tau
    #         out = out.filter(lambda row: row[kc_pos] > tau)
    #         # print("yyy")
    #     else:
    #         # print(kc_pos)
    #         # print(private_reader.tau)
    #         out = filter(lambda row: row[kc_pos] > private_reader.tau, out)
    #         # print("zzz")

    # print(list(out))

    if not postprocess:
        return out

    # print(list(out))

    def process_clamp_counts(row_in):
        # clamp counts to be non-negative
        row = [v for v in row_in]
        for idx in range(len(row)):
            if is_count[idx] and row[idx] < 0:
                row[idx] = 0
        return row

    clamp_counts = private_reader._options.clamp_counts
    if clamp_counts:
        if hasattr(out, "rdd"):
            # it's a dataframe
            out = out.rdd.map(process_clamp_counts)
        elif hasattr(out, "map"):
            # it's an RDD
            out = out.map(process_clamp_counts)
        else:
            out = map(process_clamp_counts, out)

    # get column information for outer query
    out_syms = query._select_symbols
    out_types = [s.expression.type() for s in out_syms]
    out_col_names = [s.name for s in out_syms]

    def convert(val, type):
        if val is None:
            return None  # all columns are nullable
        if type == "string" or type == "unknown":
            return str(val)
        elif type == "int":
            return int(float(str(val).replace('"', "").replace("'", "")))
        elif type == "float":
            return float(str(val).replace('"', "").replace("'", ""))
        elif type == "boolean":
            if isinstance(val, int):
                return val != 0
            else:
                return bool(str(val).replace('"', "").replace("'", ""))
        elif type == "datetime":
            v = parse_datetime(val)
            if v is None:
                raise ValueError(f"Could not parse datetime: {val}")
            return v
        else:
            raise ValueError("Can't convert type " + type)

    alphas = [alpha for alpha in private_reader.privacy.alphas]

    def process_out_row(row):
        bindings = dict((name.lower(), val) for name, val in zip(source_col_names, row))
        out_row = [c.expression.evaluate(bindings) for c in query.select.namedExpressions]
        try:
            out_row = [convert(val, type) for val, type in zip(out_row, out_types)]
        except Exception as e:
            raise ValueError(
                f"Error converting output row: {e}\n"
                f"Expecting types {out_types}"
            )

        # compute accuracies
        if accuracy == True and alphas:
            accuracies = [_accuracy.accuracy(row=list(row), alpha=alpha) for alpha in alphas]
            return tuple([out_row, accuracies])
        else:
            return tuple([out_row, []])

    if hasattr(out, "map"):
        # it's an RDD
        out = out.map(process_out_row)
    else:
        out = map(process_out_row, out)

    def filter_aggregate(row, condition):
        bindings = dict((name.lower(), val) for name, val in zip(out_col_names, row[0]))
        keep = condition.evaluate(bindings)
        return keep

    if query.having is not None:
        condition = query.having.condition
        if hasattr(out, "filter"):
            # it's an RDD
            out = out.filter(lambda row: filter_aggregate(row, condition))
        else:
            out = filter(lambda row: filter_aggregate(row, condition), out)

    # sort it if necessary
    if query.order is not None:
        sort_fields = []
        for si in query.order.sortItems:
            if type(si.expression) is not ast.Column:
                raise ValueError("We only know how to sort by column names right now")
            colname = si.expression.name.lower()
            if colname not in out_col_names:
                raise ValueError(
                    "Can't sort by {0}, because it's not in output columns: {1}".format(
                        colname, out_col_names
                    )
                )
            colidx = out_col_names.index(colname)
            desc = False
            if si.order is not None and si.order.lower() == "desc":
                desc = True
            if desc and not (out_types[colidx] in ["int", "float", "boolean", "datetime"]):
                raise ValueError("We don't know how to sort descending by " + out_types[colidx])
            sf = (desc, colidx)
            sort_fields.append(sf)

        def sort_func(row):
            # use index 0, since index 1 is accuracy
            return SortKey(row[0], sort_fields)

        if hasattr(out, "sortBy"):
            out = out.sortBy(sort_func)
        else:
            out = sorted(out, key=sort_func)

    # check for LIMIT or TOP
    limit_rows = None
    if query.limit is not None:
        if query.select.quantifier is not None:
            raise ValueError("Query cannot have both LIMIT and TOP set")
        limit_rows = query.limit.n
    elif query.select.quantifier is not None and isinstance(query.select.quantifier, Top):
        limit_rows = query.select.quantifier.n
    if limit_rows is not None:
        if hasattr(out, "rdd"):
            # it's a dataframe
            out = out.limit(limit_rows)
        elif hasattr(out, "map"):
            # it's an RDD
            out = out.take(limit_rows)
        else:
            out = itertools.islice(out, limit_rows)

    # drop empty accuracy if no accuracy requested
    def drop_accuracy(row):
        return row[0]

    if accuracy == False:
        if hasattr(out, "rdd"):
            # it's a dataframe
            out = out.rdd.map(drop_accuracy)
        elif hasattr(out, "map"):
            # it's an RDD
            out = out.map(drop_accuracy)
        else:
            out = map(drop_accuracy, out)

    # print(list(out))

    # increment odometer
    for mech in mechs:
        if mech:
            private_reader.odometer.spend(Privacy(epsilon=mech.epsilon, delta=mech.delta))

    # output it
    if accuracy == False and hasattr(out, "toDF"):
        # Pipeline RDD
        if not out.isEmpty():
            return out.toDF(out_col_names), pdfs
        else:
            return out, pdfs, dp_aggregates, exact_aggregates
    elif hasattr(out, "map"):
        # Bare RDD
        return out, pdfs, dp_aggregates, exact_aggregates
    else:
        row0 = [out_col_names]
        if accuracy == True:
            row0 = [[out_col_names,
                     [[col_name + '_' + str(1 - alpha).replace('0.', '') for col_name in out_col_names] for alpha in
                      private_reader.privacy.alphas]]]
        out_rows = row0 + list(out)
        return out_rows, pdfs, dp_aggregates, exact_aggregates


def extract_table_names(query):
    """ Extract table names from an SQL query. """
    # a good old fashioned regex. turns out this worked better than actually parsing the code
    tables_blocks = re.findall(r'(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
    tables = [tbl
              for block in tables_blocks
              for tbl in re.findall(r'\w+', block)]
    return set(tables)


def compute_neighboring_results(df, query_string, idx_to_compute, table_name, row_num_col):
    neighboring_results = []
    # original_risks = []
    # risk_score_cache = {}

    query_string = re.sub(f" WHERE ", f" WHERE ", query_string, flags=re.IGNORECASE)
    query_string = re.sub(f" FROM ", f" FROM ", query_string, flags=re.IGNORECASE)

    where_pos = query_string.rfind(" WHERE ")  # last pos of WHERE
    table_name_pos = None
    # no_where_clause = False
    if where_pos == -1:
        # no_where_clause = True
        table_name_pos = query_string.rfind(f" FROM {table_name}")
        table_name_pos += len(f" FROM {table_name}")
    else:
        where_pos += len(" WHERE ")

    engine = create_engine("sqlite:///:memory:")
    # engine = create_engine(f"sqlite:///file:memdb{num}?mode=memory&cache=shared&uri=true")

    with engine.connect() as conn:

        # start_time = time.time()

        num_rows = to_sql(df, name=table_name, con=conn,
                          # index=not any(name is None for name in df.index.names),
                          if_exists="replace")  # load index into db if all levels are named
        if num_rows != len(df):
            print("error when loading to sqlite")
            return None

        # insert_db_time = time.time() - start_time
        # print(f"time to insert to db: {insert_db_time} s")

        for i in tqdm.tqdm(idx_to_compute):

            if i == -1:
                neighboring_results.append(None)
                continue

            # column_values = tuple(df.iloc[i][query_columns].values)
            # print(column_values)
            # if column_values not in risk_score_cache.keys():

            start_time = time.time()

            if where_pos == -1:
                cur_query = query_string[:table_name_pos] + f" WHERE {row_num_col} != {i} " + query_string[
                                                                                              table_name_pos:]
            else:
                cur_query = query_string[:where_pos] + f"{row_num_col} != {i} AND " + query_string[where_pos:]

            # print(cur_query)
            cur_result = read_sql(sql=cur_query, con=conn)
            # print(cur_result.sum(numeric_only=True).sum())

            # risk_score = abs(
            #     cur_result.sum(numeric_only=True).sum() - original_result.sum(numeric_only=True).sum())

            elapsed = time.time() - start_time
            # print(f"time to execute one query: {elapsed} s")
            # break

            # risk_score_cache[column_values] = risk_score
            # else:
            #     risk_score = risk_score_cache[column_values]

            neighboring_results.append(cur_result)

    # engine.dispose()
    # original_risks_dict[num] = original_risks
    # mp_queue.put((num, original_risks))

    return neighboring_results


def get_query_sensitivities(private_reader, subquery, query):
    # get column information for outer query
    out_syms = query._select_symbols
    out_types = [s.expression.type() for s in out_syms]
    out_col_names = [s.name for s in out_syms]

    # print(out_types)
    # print(out_col_names)

    syms = subquery._select_symbols
    source_col_names = [s.name for s in syms]

    # get a list of mechanisms in column order
    mechs = private_reader._get_mechanisms(subquery)
    check_sens = [m for m in mechs if m]
    if any([m.sensitivity is np.inf for m in check_sens]):
        raise ValueError(f"Attempting to query an unbounded column")

    sens = []
    for col, m in zip(source_col_names, mechs):
        if m is None:
            sens.append((col, None))
        else:
            sens.append((col, m.sensitivity))

    return sens


def find_epsilon(df: pd.DataFrame,
                 query_string: str,
                 eps_to_test: list,
                 test_eps_num_runs: int = 1,
                 q: float = 0.05,
                 test: str = "mw",
                 num_parallel_processes: int = 8):
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore")

        table_names = extract_table_names(query_string)
        if len(table_names) != 1:
            print("error: can only query on one table")
            return None
        if query_string.count("SELECT ") > 1:
            # there's subquery
            print("error: can't have subquery")
            return None
        table_name = table_names.pop()
        # print(table_name)

        sql_parser = Parser(query_string)
        query_columns = sql_parser.columns
        # print(query_columns)
        df_copy = df.copy()
        df_copy = df_copy[query_columns]
        # df_copy = df_copy.sample(frac=1, random_state=random_state).reset_index(drop=True)

        metadata = get_metadata(df_copy, table_name)
        table_metadata = metadata[table_name][table_name][table_name]

        # just needed to create a row_num col that doesn't already exist
        row_num_col = f"row_num_{random.randint(0, 10000)}"
        while row_num_col in df_copy.columns:
            row_num_col = f"row_num_{random.randint(0, 10000)}"
        df_copy[row_num_col] = df_copy.reset_index().index
        df_copy.set_index(row_num_col)

        start_time = time.time()

        engine = create_engine("sqlite:///:memory:")
        sqlite_connection = engine.connect()

        num_rows = to_sql(df_copy, name=table_name, con=sqlite_connection,
                          # index=not any(name is None for name in df_copy.index.names),
                          if_exists="replace")  # load index into db if all levels are named
        if num_rows != len(df_copy):
            print("error when loading to sqlite")
            return None

        insert_db_time = time.time() - start_time
        print(f"time to insert to db: {insert_db_time} s")

        start_time = time.time()

        original_result = read_sql(sql=query_string, con=sqlite_connection)

        if len(original_result.select_dtypes(include=np.number).columns) > 1:
            print("error: can only have one numerical column in the query result")
            return None

        # print("original_result")
        # print(original_result)

        neighboring_exact_aggregates = []

        # start_time = time.time()

        cache = defaultdict(list)

        columns_values = list(df[query_columns].itertuples(index=False, name=None))

        for i in range(len(columns_values)):
            cache[columns_values[i]].append(i)

        inv_cache = {}
        indices = np.arange(len(df_copy))
        indices_to_ignore = []

        for k, v in cache.items():

            indices_to_ignore += v[1:]  # only need to compute result for one

            for i in v:
                inv_cache[i] = k

        np.put(indices, indices_to_ignore, [-1] * len(indices_to_ignore))

        # print(len(indices), len(indices_to_ignore), len(indices) - len(indices_to_ignore))

        elapsed = time.time() - start_time
        print(f"time to create cache: {elapsed} s")

        start_time = time.time()

        idx_split = np.array_split(indices, num_parallel_processes)

        with mp.Pool(processes=num_parallel_processes) as mp_pool:

            # args = [(df_copy, query_string, idx_to_compute, table_name, row_num_col)
            #         for idx_to_compute in idx_split]

            query_string = re.sub(f" FROM {table_name}", f" FROM {table_name}.{table_name}", query_string,
                                  flags=re.IGNORECASE)

            privacy = Privacy(epsilon=1.0)  # does not really matter, cuz I am getting the exact aggregates
            private_reader = snsql.from_df(df_copy, metadata=metadata, privacy=privacy)
            private_reader._options.censor_dims = False
            private_reader.rewriter.options.censor_dims = False

            query_ast = private_reader.parse_query_string(query_string)
            subquery, query = private_reader._rewrite_ast(query_ast)

            args = [(df_copy, table_name, query_string, row_num_col, idx_to_compute, private_reader, subquery, query, table_metadata)
                    for idx_to_compute in idx_split]

            for cur_neighboring_results in mp_pool.starmap(compute_exact_aggregates_of_neighboring_data, args):  #
                # *np.array(args).T):
                neighboring_exact_aggregates += cur_neighboring_results

            # compute_exact_aggregates_of_neighboring_data(df_copy, table_name, row_num_col, idx_to_compute,
            #                                              private_reader, subquery, query)

        for i in range(len(neighboring_exact_aggregates)):
            if neighboring_exact_aggregates[i] is None:
                idx_computed = cache[inv_cache[i]][0]
                neighboring_exact_aggregates[i] = neighboring_exact_aggregates[idx_computed]

        elapsed = time.time() - start_time
        print(f"time to compute neighboring_exact_aggregates: {elapsed} s")

        # print(neighboring_exact_aggregates[0])

        # zz: we have now computed all f(x_i').
        #  Next, for each epsilon from largest to smallest:
        #  1. compute the DP result s
        #  2. compute PRI_i = |ln (P(lap(sens(f)/eps) = s - f(x)) / P(lap(sens(f)/eps) = s - f(x_i')))| / eps
        #  3. test whether the PRIs come from the same distribution
        #  4. If yes, return the current epsilon and s

        best_eps = None

        # We start from the largest epsilon to the smallest
        sorted_eps_to_test = np.sort(eps_to_test)[::-1]

        compute_risk_time = 0.0
        test_equal_distribution_time = 0.0

        # query_string = query_string.replace(f" {table_name} ", f" {table_name}.{table_name} ")
        # query_string = re.sub(f" FROM {table_name}", f" FROM {table_name}.{table_name}", query_string,
        # flags=re.IGNORECASE)

        for eps in sorted_eps_to_test:

            print(f"epsilon = {eps}")

            p_values = []

            privacy = Privacy(epsilon=eps)
            private_reader = snsql.from_df(df_copy, metadata=metadata, privacy=privacy)
            private_reader._options.censor_dims = False
            private_reader.rewriter.options.censor_dims = False
            # dp_result = private_reader.execute_df(query)
            # rewrite the query ast once for every epsilon
            query_ast = private_reader.parse_query_string(query_string)
            try:
                subquery, query = private_reader._rewrite_ast(query_ast)
            except ValueError as err:
                print(err)
                return None
            # col_sensitivities = get_query_sensitivities(private_reader, subquery, query)
            # print(col_sensitivities)

            for j in range(test_eps_num_runs):

                start_time = time.time()

                # compute the probability of getting the dp_result from the original dataset
                dp_result, pdfs_original, dp_aggregates, exact_aggregates = execute_rewritten_ast(sqlite_connection, table_name, query_string, table_metadata,
                                                                                private_reader,
                                                                                subquery, query)
                dp_result = private_reader._to_df(dp_result)

                # print(dp_result)
                print("dp_aggregates", dp_aggregates)
                print("pdfs_original", pdfs_original)

                print("neighboring_exact_aggregates", neighboring_exact_aggregates)

                PRIs = []
                for neighbor in range(len(neighboring_exact_aggregates)):
                    pdfs_neighboring = compute_noise_prob_using_neighboring_data(
                        neighboring_exact_aggregates[neighbor],
                        dp_aggregates, exact_aggregates, private_reader,
                        subquery, query, query_string)

                    # print("neighboring_exact_aggregates", neighboring_exact_aggregates[neighbor])
                    # print("pdfs_neighboring", pdfs_neighboring)

                    ratio = 1.0

                    for row_pdf_original, row_pdf_neighboring in zip(pdfs_original, pdfs_neighboring):
                        for cur_pdf_original, cur_pdf_neighboring in zip(row_pdf_original, row_pdf_neighboring):
                            cur_pdf_ratio = cur_pdf_original / cur_pdf_neighboring
                            if math.isclose(cur_pdf_ratio, 1):
                                cur_pdf_ratio = 1
                            ratio *= cur_pdf_ratio

                    # if ratio > math.exp(eps):
                    #     print(ratio)
                    #     print(math.exp(eps))
                    #     exit()

                    # print(ratio)
                    if math.isclose(ratio, 1):
                        PRI = 0.0
                    else:
                        PRI = abs(math.log(ratio)) / eps

                    if PRI > 1:
                        print("neighboring_exact_aggregates", neighboring_exact_aggregates[neighbor])
                        print("pdfs_neighboring", pdfs_neighboring)
                        print("PRI", PRI)

                    PRIs.append(PRI)

                # print(PRIs)

                print(pd.DataFrame(PRIs).describe())

                PRIs.sort()

                PRI1 = PRIs[:100]
                PRI2 = PRIs[-100:]

                elapsed = time.time() - start_time
                # print(f"{j}th compute risk time: {elapsed} s")
                compute_risk_time += elapsed

                # We perform the test and record the p-value for each run
                start_time = time.time()

                if test == "mw":
                    cur_res = stats.mannwhitneyu(PRI1, PRI2)
                elif test == "ks":
                    cur_res = stats.ks_2samp(PRI1, PRI2)  # , method="exact")
                elif test == "es":
                    for i1 in range(len(PRI1)):
                        if PRI1[i1] == 0.0:
                            PRI1[i1] += 1e-100
                    cur_res = stats.epps_singleton_2samp(PRI1, PRI2)

                p_value = cur_res[1]
                # if p_value < 0.01:
                #     reject_null = True # early stopping
                p_values.append(p_value)

                # print(pd.DataFrame(new_risks1).describe())
                # print(pd.DataFrame(new_risks2).describe())
                # print(f"p_value: {p_value}")

                elapsed = time.time() - start_time
                test_equal_distribution_time += elapsed

            # Now we have n p-values for the current epsilon, we use multiple comparisons' technique
            # to determine if this epsilon is good enough
            # For now we use false discovery rate
            # print(p_values)
            if test_eps_num_runs > 1:
                if not fdr(p_values, q):  # q = proportion of false positives we will accept
                    # We want no discovery (fail to reject null) for all n runs
                    # If we fail to reject the null, then we break the loop.
                    # The current epsilon is the one we choose
                    best_eps = eps
                    break
            else:
                if p_values[0] > q:
                    best_eps = eps
                    break

            # TODO: if we find a lot of discoveries (a lot of small p-values),
            #  we can skip epsilons that are close to the current eps (ex. 10 to 9).
            #  If there's only a few discoveries,
            #  we should probe epsilons that are close (10 to 9.9)

        # print(f"total time to compute new risk: {compute_risk_time} s")
        # print(f"total time to test equal distribution: {test_equal_distrbution_time} s")

        sqlite_connection.close()

        if best_eps is None:
            return None

        return best_eps, dp_result  # also return the dp result computed


if __name__ == '__main__':
    csv_path = '../adult.csv'
    df = pd.read_csv(csv_path)#.head(100)
    # print(df.head())
    # df = pd.read_csv("../scalability/adult_1000.csv")
    # df = pd.read_csv("../adult.csv")
    # df = df[["age", "education", "education.num", "race", "income"]]
    df.rename(columns={'education.num': 'education_num'}, inplace=True)
    df = df.sample(1000, random_state=0, ignore_index=True)
    # df = pd.read_csv("adult_100_sample.csv")
    # df.rename(columns={'education.num': 'education_num'}, inplace=True)
    # df["row_num"] = df.reset_index().index
    # print(df)

    # query_string = "SELECT COUNT(*) FROM adult WHERE age < 40 AND income == '>50K'"
    # query_string = "SELECT race, COUNT(*) FROM adult WHERE education_num >= 14 AND income == '<=50K' GROUP BY race"

    # query_string = "SELECT COUNT(*) FROM adult WHERE income == '>50K' AND education_num == 13 AND age == 25"
    # query_string = "SELECT marital_status, COUNT(*) FROM adult WHERE race == 'Asian-Pac-Islander' AND age >= 30 AND age <= 40 GROUP BY marital_status"
    # query_string = "SELECT COUNT(*) FROM adult WHERE native_country != 'United-States' AND sex == 'Female'"
    query_string = "SELECT AVG(hours_per_week) FROM adult WHERE workclass == 'Federal-gov' OR workclass == 'Local-gov' or workclass == 'State-gov'"
    # query_string = "SELECT SUM(fnlwgt) FROM adult WHERE capital_gain > 0 AND income == '<=50K' AND occupation == 'Sales'"

    # query_string = "SELECT sex, AVG(age) FROM adult GROUP BY sex"

    # query_string = "SELECT AVG(age) FROM adult"

    # query_string = "SELECT AVG(capital_loss) FROM adult WHERE hours_per_week > 40 AND workclass == 'Federal-gov' OR
    # workclass == 'Local-gov' or workclass == 'State-gov'"

    # design epsilons to test in a way that smaller eps are more frequent and largest eps are less
    eps_list = list(np.arange(0.001, 0.01, 0.001, dtype=float))
    eps_list += list(np.arange(0.01, 0.1, 0.01, dtype=float))
    eps_list += list(np.arange(0.1, 1, 0.1, dtype=float))
    eps_list += list(np.arange(1, 11, 1, dtype=float))
    eps_list = [0.01]

    start_time = time.time()
    eps = find_epsilon(df, query_string, eps_list, 1, 0.05, test="mw", num_parallel_processes=1)
    elapsed = time.time() - start_time
    print(f"total time: {elapsed} s")

    print(eps)

    # times = []
    #
    # for i in range(1, 9):
    #
    #     start_time = time.time()
    #     eps = find_epsilon(df, query_string, -1, 100, eps_list, 1, 0.05, test="mw", num_parallel_processes=i)
    #     elapsed = time.time() - start_time
    #     print(f"total time: {elapsed} s")
    #
    #     print(eps)
    #
    #     times.append(elapsed)
    #
    # plt.plot(np.arange(1, 9), times)
    # plt.savefig("num_processes")
