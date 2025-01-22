import random
# random_state = 0
# random.seed(random_state)

import itertools
import math
import numbers
import re
from enum import Enum

import numpy as np
import pandas as pd
from pandas.core.dtypes.inference import is_number
from scipy import stats
import tqdm
import time
import warnings
import snsql
from snsql import Privacy
from snsql._ast.expressions.date import parse_datetime
from snsql._ast.expressions import sql as ast
from snsql._ast.ast import Top
from sqlalchemy import create_engine
from pandas.io.sql import to_sql, read_sql
from statsmodels.stats.multitest import fdrcorrection
from sql_metadata import Parser
import multiprocessing as mp
# from multiprocessing.pool import Pool
# import pathos.multiprocessing as mp
# from pathos.multiprocessing import ProcessPool
from collections import defaultdict
from scipy.stats import laplace
from scipy import spatial
from copy import deepcopy
import string
from snsql.sql.reader.base import SortKeyExpressions
from snsql.sql.privacy import Privacy, Stat
from snsql.sql._mechanisms.base import Mechanism

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

def compute_neighboring_results(df, query_string, idx_to_compute, table_name, row_num_col, table_metadata):
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

    engine = create_engine("sqlite:///:memory:", connect_args={'check_same_thread': False})
    # engine = create_engine(f"sqlite:///file:memdb{num}?mode=memory&cache=shared&uri=true")

    with engine.connect() as conn:

        # start_time = time.time()

        # dummy_row = df.iloc[0].copy()
        # for col in df.columns:
        #     if col in table_metadata.keys():
        #         if table_metadata[col]["type"] == "int" or table_metadata[col]["type"] == "float":
        #             dummy_row[col] = table_metadata[col]["lower"]
        #         elif table_metadata[col]["type"] == "string":
        #             dummy_row[col] = None
        #     else:
        #         dummy_row[col] = len(df) + 1
        #
        # # print(dummy_row)
        # df_with_dummy_row = df.append(dummy_row)

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


def execute_rewritten_ast(sqlite_connection, table_name,
                          private_reader, subquery, query, *ignore, accuracy: bool = False, pre_aggregated=None,
                          postprocess=True):
    
    _orig_query = query

    agg_names = []
    for col in _orig_query.select.namedExpressions:
        if isinstance(col.expression, ast.AggFunction):
            agg_names.append(col.expression.name)
        else:
            agg_names.append(None)

    private_reader._options.row_privacy = query.row_privacy
    private_reader._options.censor_dims = False #query.censor_dims
    private_reader._options.reservoir_sample = query.sample_max_ids
    private_reader._options.clamp_counts = True #query.clamp_counts
    private_reader._options.use_dpsu = query.use_dpsu
    private_reader._options.clamp_columns = query.clamp_columns
    private_reader._refresh_options()

    # if isinstance(query, str):
    #     raise ValueError("Please pass AST to _execute_ast.")
    #
    # subquery, query = self._rewrite_ast(query)

    if pre_aggregated is not None:
        exact_aggregates = private_reader._check_pre_aggregated_columns(pre_aggregated, subquery)
    else:
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

        # print(query_string)
        query_string = re.sub(f" FROM {table_name}.{table_name}", f" FROM {table_name}", query_string,
                       flags=re.IGNORECASE)

        # print(query_string)
        q_result = read_sql(sql=query_string, con=sqlite_connection)
        # print(q_result)
        exact_aggregates = [tuple([col for col in q_result.columns])] + [val[1:] for val in q_result.itertuples()]
        # print(exact_aggregates)

    _accuracy = None
    if accuracy:
        raise NotImplementedError("Simple accuracy has been removed.  Please see documentation for information on estimating accuracy.")

    syms = subquery._select_symbols
    source_col_names = [s.name for s in syms]

    # tell which are counts, in column order
    is_count = [s.expression.is_count for s in syms]

    # get a list of mechanisms in column order
    mechs = private_reader._get_mechanisms(subquery)
    check_sens = [m for m in mechs if m]
    if any([m.sensitivity is np.inf for m in check_sens]):
        raise ValueError(f"Attempting to query an unbounded column")
    
    # print("mechs:", mechs)

    kc_pos = private_reader._get_keycount_position(subquery)

    def randomize_row_values(row_in):
        row = [v for v in row_in]
        # set null to 0 before adding noise
        for idx in range(len(row)):
            if mechs[idx] and row[idx] is None:
                row[idx] = 0.0
        # call all mechanisms to add noise
        return [
            mech.release([v])[0] if mech is not None else v
            for mech, v in zip(mechs, row)
        ]

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

    # censor infrequent dimensions
    if private_reader._options.censor_dims:
        if kc_pos is None:
            raise ValueError("Query needs a key count column to censor dimensions")
        else:
            thresh_mech = mechs[kc_pos]
            private_reader.tau = thresh_mech.threshold
        if hasattr(out, "filter"):
            # it's an RDD
            tau = private_reader.tau
            out = out.filter(lambda row: row[kc_pos] > tau)
        else:
            out = filter(lambda row: row[kc_pos] > private_reader.tau, out)

    if not postprocess:
        return out

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
    bind_prefix = ''.join(np.random.choice(list(string.ascii_lowercase), 5))
    binding_col_names = [name if name != "???" else f"col_{bind_prefix}_{i}" for i, name in enumerate(out_col_names)]

    def convert(val, type):
        if val is None:
            return None # all columns are nullable
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
        # fix up case where variance is negative
        out_row_fixed = []
        for val, agg in zip(out_row, agg_names):
            if agg == 'VAR' and val < 0:
                out_row_fixed.append(0.0)
            elif agg == 'STDDEV' and np.isnan(val):
                out_row_fixed.append(0.0)
            else:
                out_row_fixed.append(val)
        out_row = out_row_fixed
        try:
            out_row =[convert(val, type) for val, type in zip(out_row, out_types)]
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
        bindings = dict((name.lower(), val) for name, val in zip(binding_col_names, row[0]))
        keep = condition.evaluate(bindings)
        return keep

    if query.having is not None:
        condition = deepcopy(query.having.condition)
        for i, ne in enumerate(_orig_query.select.namedExpressions):
            source_col = binding_col_names[i]
            condition = condition.replaced(ne.expression, ast.Column(source_col), lock=True)
        if hasattr(out, "filter"):
            # it's an RDD
            out = out.filter(lambda row: filter_aggregate(row, condition))
        else:
            out = filter(lambda row: filter_aggregate(row, condition), out)

    # sort it if necessary
    if query.order is not None:
        sort_expressions = []
        for si in query.order.sortItems:
            desc = False
            if si.order is not None and si.order.lower() == "desc":
                desc = True
            if type(si.expression) is ast.Column and si.expression.name.lower() in out_col_names:
                sort_expressions.append((desc, si.expression))
            else:
                expr = deepcopy(si.expression)
                for i, ne in enumerate(_orig_query.select.namedExpressions):
                    source_col = binding_col_names[i]
                    expr = expr.replaced(ne.expression, ast.Column(source_col), lock=True)
                sort_expressions.append((desc, expr))

        def sort_func(row):
            # use index 0, since index 1 is accuracy
            return SortKeyExpressions(row[0], sort_expressions, binding_col_names)
            
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

    # increment odometer
    for mech in mechs:
        if mech:
            private_reader.odometer.spend(Privacy(epsilon=mech.epsilon, delta=mech.delta))

    # output it
    if accuracy == False and hasattr(out, "toDF"):
        # Pipeline RDD
        if not out.isEmpty():
            return out.toDF(out_col_names)
        else:
            return out
    elif hasattr(out, "map"):
        # Bare RDD
        return out
    else:
        row0 = [out_col_names]
        if accuracy == True:
            row0 = [[out_col_names, [[col_name+'_' + str(1-alpha).replace('0.', '') for col_name in out_col_names] for alpha in private_reader.privacy.alphas ]]]
        out_rows = row0 + list(out)
        return out_rows


def extract_table_names(query):
    """ Extract table names from an SQL query. """
    # a good old fashioned regex. turns out this worked better than actually parsing the code
    tables_blocks = re.findall(r'(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
    tables = [tbl
              for block in tables_blocks
              for tbl in re.findall(r'\w+', block)]
    return set(tables)


class QueryType(Enum):
    COUNT = 1
    SUM = 2
    AVG = 3
    # MIN = 4
    # MAX = 5

def find_epsilon(df: pd.DataFrame,
                 query_string: str,
                 eps_to_test: list,
                 percentage: int = 5,
                 num_parallel_processes: int = 8,
                 gaussian: bool = False,
                 svt: bool = False,
                 svt_eps: float = 1.0,
                 variance_threshold: float = 10e-10):

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore")

        sql_parser = Parser(query_string)

        table_names = sql_parser.tables
        if len(table_names) != 1:
            raise Exception("error: can only query on one table")

        if len(sql_parser.subqueries) > 0:
            # there's subquery
            raise Exception("error: can't have subquery")

        if " COUNT" in query_string:
            query_type = QueryType.COUNT
        elif " SUM" in query_string:
            query_type = QueryType.SUM
        elif " AVG" in query_string:
            query_type = QueryType.AVG
        else:
            raise Exception("error: query type not supported")

        if query_type != QueryType.COUNT:
            select_cols = sql_parser.columns_dict["select"]
            if "group_by" in sql_parser.columns_dict:
                group_by_cols = sql_parser.columns_dict["group_by"]
                select_cols = set(select_cols) - set(group_by_cols)
            # print(select_cols)
            if len(select_cols) != 1:
                raise Exception("error: can only have one select column")
            select_col = select_cols.pop()

        table_name = table_names.pop()
        # print(table_name)
        query_columns = sql_parser.columns
        # print(query_columns)
        df = df[query_columns]
        # df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        metadata = get_metadata(df, table_name)
        table_metadata = metadata[table_name][table_name][table_name]

        # just needed to create a row_num col that doesn't already exist
        row_num_col = "row_num"
        df[row_num_col] = df.reset_index().index
        df.set_index(row_num_col)

        start_time = time.time()

        engine = create_engine("sqlite:///:memory:", connect_args={'check_same_thread': False})
        sqlite_connection = engine.connect()

        num_rows = to_sql(df, name=table_name, con=sqlite_connection,
                          # index=not any(name is None for name in df.index.names),
                          if_exists="replace")  # load index into db if all levels are named
        if num_rows != len(df):
            print("error when loading to sqlite")
            return None

        insert_db_time = time.time() - start_time
        # print(f"time to insert to db: {insert_db_time} s")

        start_time = time.time()

        original_result = read_sql(sql=query_string, con=sqlite_connection)

        sqlite_connection.close()

        original_aggregates = [val[1:] for val in original_result.itertuples()]

        # if len(original_result.select_dtypes(include=np.number).columns) > 1:
        #     print("error: can only have one numerical column in the query result")
        #     return None

        # print("original_result", original_aggregates)
        # print(original_result)

        neighboring_results = []

        # start_time = time.time()

        cache = defaultdict(list)

        columns_values = list(df[query_columns].itertuples(index=False, name=None))

        for i in range(len(columns_values)):
            cache[columns_values[i]].append(i)

        inv_cache = {}
        indices = np.arange(len(df))
        indices_to_ignore = []

        for k, v in cache.items():

            indices_to_ignore += v[1:]  # only need to compute result for one

            for i in v:
                inv_cache[i] = k

        np.put(indices, indices_to_ignore, [-1] * len(indices_to_ignore))

        # print(len(indices), len(indices_to_ignore), len(indices) - len(indices_to_ignore))

        elapsed = time.time() - start_time
        # print(f"time to create cache: {elapsed} s")

        start_time = time.time()

        idx_split = np.array_split(indices, num_parallel_processes)

        with mp.Pool(processes=num_parallel_processes) as mp_pool:

            args = [(df, query_string, idx_to_compute, table_name, row_num_col, table_metadata)
                    for idx_to_compute in idx_split]

            for cur_neighboring_results in mp_pool.starmap(compute_neighboring_results, args):  #
                # *np.array(args).T):
                neighboring_results += cur_neighboring_results

            # compute_exact_aggregates_of_neighboring_data(df, table_name, row_num_col, idx_to_compute,
            #                                              private_reader, subquery, query)

        neighboring_aggregates = []

        for i in range(len(neighboring_results)):
            if neighboring_results[i] is None:
                idx_computed = cache[inv_cache[i]][0]
                neighboring_results[i] = neighboring_results[idx_computed]

            neighboring_result = neighboring_results[i]
            neighboring_aggregate = [val[1:] for val in neighboring_result.itertuples()]
            # print("neighboring_aggregates", neighboring_aggregates)

            if len(neighboring_aggregate) != len(original_aggregates):
                # corner case: group by results missing for one group after removing one record
                missing_group = set(original_aggregates) - set(neighboring_aggregate)
                # assert len(missing_group) == 1
                missing_group = missing_group.pop()
                missing_group_pos = original_aggregates.index(missing_group)
                missing_group = list(missing_group)
                for i in range(len(missing_group)):
                    if isinstance(missing_group[i], numbers.Number):
                        missing_group[i] = 0
                neighboring_aggregate.insert(missing_group_pos, tuple(missing_group))

            # print("neighboring_aggregates", neighboring_aggregates)

            neighboring_aggregates.append(neighboring_aggregate)

        elapsed = time.time() - start_time
        # print(f"time to compute neighboring_results: {elapsed} s")

        best_eps = None

        # We start from the largest epsilon to the smallest
        sorted_eps_to_test = np.sort(eps_to_test)[::-1]

        # query_string = query_string.replace(f" {table_name} ", f" {table_name}.{table_name} ")
        query_string = re.sub(f" FROM {table_name}", f" FROM {table_name}.{table_name}", query_string,
                              flags=re.IGNORECASE)
        
        # SVT
        if svt:
            variance_sens = 1 / num_rows

            eps_svt_1 = svt_eps / (1 + math.pow(2, 2/3))
            eps_svt_2 = math.pow(2, 2/3) * svt_eps / (1 + math.pow(2, 2/3))
            # eps_1 = svt_eps / 3
            # eps_2 = 2 * svt_eps / 3
            threshold_noise = np.random.laplace(loc=0, scale=variance_sens/eps_svt_1)

        if gaussian:
            delta = 1 / pow(num_rows, 2)

        for eps in sorted_eps_to_test:

            print(f"epsilon = {eps}")

            PRIs = []
            k = len(original_aggregates)

            if query_type == QueryType.COUNT:
                sens = 1
            elif query_type == QueryType.SUM:
                sens = int(table_metadata[select_col]["upper"]) - int(table_metadata[select_col]["lower"])
            elif query_type == QueryType.AVG:
                sens = (int(table_metadata[select_col]["upper"]) - int(table_metadata[select_col]["lower"])) / num_rows

            if gaussian:
                scale = 2.0 * (sens**2) * math.log(1.25 / delta) / (eps**2)
            else:
                scale = sens / eps
            # print(scale)

            for neighboring_aggregate in neighboring_aggregates:

                per_instance_sens = 0
                for row1, row2 in zip(original_aggregates, neighboring_aggregate):
                    for val1, val2 in zip(row1, row2):
                        if is_number(val1) and is_number(val2):

                            if gaussian:
                                per_instance_sens += (val1 - val2)**2
                            else:
                                per_instance_sens += abs(val1 - val2)

                if gaussian:
                    # print(per_instance_sens, scale, k * scale)
                    PRI = math.sqrt(per_instance_sens + k * scale)
                else:
                    # print(per_instance_sens)
                    PRI = per_instance_sens + k * scale

                PRIs.append(PRI) 

            # print(pd.DataFrame(PRIs).describe())

            # min_pri = np.min(PRIs)
            max_pri = np.max(PRIs)
            # median = np.median(PRIs)
            # denom = np.percentage(PRIs, percentage)

            # print("max / min", max / min)
            # print("max / denom", max / denom)
            # print("max / med", max / median)

            if svt:
                PRIs = [val/max_pri for val in PRIs]

                variance = np.var(PRIs)
                variance_noise = np.random.laplace(loc=0, scale=2*variance_sens/eps_svt_2)

                noisy_variance = -variance + variance_noise
                noisy_threshold = -variance_threshold + threshold_noise

                # print(PRIs[:10])
                # print(variance, variance_noise)
                # print(variance_threshold, threshold_noise)
                # print(noisy_variance, noisy_threshold)

                # var <= threshold ~ -var >= -threshold
                if noisy_variance >= noisy_threshold:
                    best_eps = float(eps)
                    break                
                # print(ratio, threshold)

                # ratio += np.random.laplace(loc=0, scale=2/eps_svt_2)
                # threshold += threshold_noise
                
                # ratio = np.clip(ratio, 0.0, 1.0)
                # threshold = np.clip(threshold, 0.0, 1.0)

                # print(ratio, threshold)

            else: 
                min_pri = np.min(PRIs)

                # print(min_pri, max_pri)

                ratio = min_pri / max_pri
                threshold = 1.0 - percentage / 100

                if ratio >= threshold:
                    best_eps = float(eps)
                    break

        # print(f"total time to compute new risk: {compute_risk_time} s")
        # print(f"total time to test equal distribution: {test_equal_distrbution_time} s")

        if best_eps is None:
            return None, None, insert_db_time

        if gaussian:
            privacy = Privacy(epsilon=eps, delta=delta)
            privacy.mechanisms.map[Stat.count] = Mechanism.gaussian
            privacy.mechanisms.map[Stat.sum_int] = Mechanism.gaussian
            privacy.mechanisms.map[Stat.sum_large_int] = Mechanism.gaussian
            privacy.mechanisms.map[Stat.sum_float] = Mechanism.gaussian
            privacy.mechanisms.map[Stat.threshold] = Mechanism.gaussian
        else:
            privacy = Privacy(epsilon=eps, delta=0)

        private_reader = snsql.from_df(df, metadata=metadata, privacy=privacy)
        dp_result = private_reader.execute(query_string)
        # dp_aggregates = [val[1:] for val in dp_result.itertuples()]

        return best_eps, dp_result, insert_db_time  # also return the dp result computed


if __name__ == '__main__':

    df = pd.read_csv("../scalability/data/adult_10000.csv")

    # query_string = "SELECT COUNT(*) FROM adult WHERE income == '>50K' AND education_num == 13 AND age == 25"
    # query_string = "SELECT marital_status, COUNT(*) FROM adult WHERE race == 'Asian-Pac-Islander' AND age >= 30 AND age <= 40 GROUP BY marital_status"
    # query_string = "SELECT COUNT(*) FROM adult WHERE native_country != 'United-States' AND sex == 'Female'"
    # query_string = "SELECT AVG(hours_per_week) FROM adult WHERE workclass == 'Federal-gov' OR workclass == 'Local-gov' or workclass == 'State-gov'"
    query_string = "SELECT SUM(capital_gain) FROM adult"

    # query_string = "SELECT sex, AVG(age) FROM adult GROUP BY sex"
    # query_string = "SELECT AVG(age) FROM adult"
    # query_string = "SELECT AVG(capital_loss) FROM adult WHERE hours_per_week > 40 AND workclass == 'Federal-gov' OR
    # workclass == 'Local-gov' or workclass == 'State-gov'"

    # design epsilons to test in a way that smaller eps are more frequent and largest eps are less
    eps_list = list(np.arange(0.001, 0.01, 0.001, dtype=float))
    eps_list += list(np.arange(0.01, 0.1, 0.01, dtype=float))
    eps_list += list(np.arange(0.1, 1, 0.1, dtype=float))
    eps_list += list(np.arange(1, 11, 1, dtype=float))
    # print(len(eps_list))
    # exit()
    # eps_list = [0.01]

    start_time = time.time()
    best_eps, dp_result, insert_db_time = find_epsilon(df, query_string, eps_list, 
                                                       num_parallel_processes=8, 
                                                       percentage=5,
                                                       gaussian=True,
                                                       svt=False,
                                                       svt_eps=1,
                                                       variance_threshold=10e-8)
    elapsed = time.time() - start_time
    print(f"total time: {elapsed} s")

    print(best_eps)
    print(dp_result)

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
