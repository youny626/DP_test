import itertools

import numpy as np
import pandas as pd
from scipy import stats
import tqdm
import time
import warnings
import snsql
from snsql import Privacy
from snsql._ast.expressions.date import parse_datetime
from snsql.sql._mechanisms.accuracy import Accuracy
from snsql._ast.expressions import sql as ast
from snsql.sql.reader.base import SortKey
from snsql._ast.ast import Top


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

def execute_rewritten_ast(private_reader, subquery, query, *ignore, accuracy: bool = False, pre_aggregated=None, postprocess=True):
    # if isinstance(query, str):
    #     raise ValueError("Please pass AST to _execute_ast.")
    # 
    # subquery, query = self._rewrite_ast(query)

    if pre_aggregated is not None:
        exact_aggregates = private_reader._check_pre_aggregated_columns(pre_aggregated, subquery)
    else:
        exact_aggregates = private_reader._get_reader(subquery)._execute_ast(subquery)

    _accuracy = None
    if accuracy:
        _accuracy = Accuracy(query, subquery, private_reader.privacy)

    syms = subquery._select_symbols
    source_col_names = [s.name for s in syms]

    # tell which are counts, in column order
    is_count = [s.expression.is_count for s in syms]

    # get a list of mechanisms in column order
    mechs = private_reader._get_mechanisms(subquery)
    check_sens = [m for m in mechs if m]
    if any([m.sensitivity is np.inf for m in check_sens]):
        raise ValueError(f"Attempting to query an unbounded column")

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
            row0 = [[out_col_names,
                     [[col_name + '_' + str(1 - alpha).replace('0.', '') for col_name in out_col_names] for alpha in
                      private_reader.privacy.alphas]]]
        out_rows = row0 + list(out)
        return out_rows


def find_epsilon(df: pd.DataFrame, data_name: str, query_string: str,
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
        original_result = private_reader.reader.execute_df(query_string)

        # Compute the original privacy risks
        original_risks = []
        dfs_one_off = []  # cache
        for i in range(len(df)):
            cur_df = df.copy()
            cur_df = cur_df.drop([i])
            dfs_one_off.append(cur_df)

            private_reader = snsql.from_df(cur_df, metadata=metadata, privacy=privacy)
            cur_result = private_reader.reader.execute_df(query_string) # execute query without DP
            # print(type(private_reader.reader).__name__)

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

            # reject_null = False
            p_values = []

            # For each run, we compute the risks again
            # New risk = DP result - DP result when removing one record

            privacy = Privacy(epsilon=eps)
            private_reader = snsql.from_df(df, metadata=metadata, privacy=privacy)
            # dp_result = private_reader.execute_df(query)
            # rewrite the query ast once for every epsilon
            # TODO: might be able to do it once if epsilon is not involved in rewriting
            query_ast = private_reader.parse_query_string(query_string)
            subquery, query = private_reader._rewrite_ast(query_ast)

            for j in tqdm.tqdm(range(num_runs)):

                # if reject_null:
                #     continue # this eps does not equalize risks, skip

                start_time = time.time()

                # better to compute a new DP result each run
                # dp_result = private_reader.execute_df(query)
                # query_ast = private_reader.parse_query_string(query_string)
                # subquery, query = private_reader._rewrite_ast(query_ast)
                dp_result = execute_rewritten_ast(private_reader, subquery, query)
                dp_result = private_reader._to_df(dp_result)
                # print(dp_result)

                # We get the same samples based on individuals who are present in the original chosen samples
                # we only need to compute risks for those samples
                new_risks1 = []
                new_risks2 = []

                for i in sample_idx1:
                    cur_df = dfs_one_off[i]

                    private_reader = snsql.from_df(cur_df, metadata=metadata, privacy=privacy)
                    # cur_result = private_reader.execute_df(query)
                    cur_result = execute_rewritten_ast(private_reader, subquery, query)
                    cur_result = private_reader._to_df(cur_result)

                    change = abs(cur_result - dp_result).to_numpy().sum()
                    new_risks1.append(change)

                for i in sample_idx2:
                    cur_df = dfs_one_off[i]

                    private_reader = snsql.from_df(cur_df, metadata=metadata, privacy=privacy)
                    cur_result = execute_rewritten_ast(private_reader, subquery, query)
                    cur_result = private_reader._to_df(cur_result)

                    change = abs(cur_result - dp_result).to_numpy().sum()
                    new_risks2.append(change)

                elapsed = time.time() - start_time
                print(f"{j}th compute risk time: {elapsed} s")
                compute_risk_time += elapsed

                # We perform the test and record the p-value for each run
                start_time = time.time()

                cur_res = stats.ks_2samp(new_risks1, new_risks2)
                p_value = cur_res[1]
                # if p_value < 0.01:
                #     reject_null = True # early stopping
                p_values.append(p_value)

                elapsed = time.time() - start_time
                test_equal_distrbution_time += elapsed

            # Now we have n p-values for the current epsilon, we use multiple comparisons' technique
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
    df = pd.read_csv(csv_path)#.head(100)
    # print(df.head())

    query_string = "SELECT AVG(age) FROM PUMS.PUMS"

    # design epsilons to test in a way that smaller eps are more frequent and largest eps are less
    eps_list = list(np.arange(0.01, 0.1, 0.01, dtype=float))
    eps_list += list(np.arange(0.1, 1.1, 0.1, dtype=float))
    # eps_list += list(np.arange(1, 11, 1, dtype=float))
    print(eps_list)

    start_time = time.time()
    eps = find_epsilon(df, "PUMS", query_string, 90, eps_list, 10, 0.05)
    elapsed = time.time() - start_time
    print(f"total time: {elapsed} s")

    print(eps)

# Successfully installed 
# PyYAML==5.4.1 graphviz==0.17 numpy==1.23.3 opendp==0.4.0 pandas==1.5.0 pandasql==0.7.3 pytz==2022.2.1
# sqlalchemy==1.4.41
# PyYAML==5.4.1 antlr4-python3-runtime==4.9.3 graphviz==0.17 numpy==1.23.3 opendp==0.4.0 pandas==1.5.0
# pandasql==0.7.3 python==dateutil==2.8.2 pytz==2022.2.1 six==1.16.0 smartnoise-sql==0.2.4 sqlalchemy==1.4.41
