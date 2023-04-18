import pickle


from clustering.column_model import Column
from clustering.emd_utils import quantile_emd, intersection_emd
from clustering.quantile_histogram.histogram import QuantileHistogram


def compute_cutoff_threshold(C: list, threshold: float):
 
    C.append({'e': threshold, 'c': 0})
    C = sorted(C, key=lambda k: k['e'])
    cutoff = 0.0
    gap = 0.0
    i = 0
    while i < len(C) - 1 and C[i + 1]['e'] <= threshold:
        if gap < (C[i + 1]['e'] - C[i]['e']):
            gap = C[i + 1]['e'] - C[i]['e']
            cutoff = C[i]['e']
        i += 1
    return cutoff


def column_combinations(columns: list, quantiles: int, intersection: bool = False):
   
    c = len(columns)
    c_i = 0
    while c_i < c:
        name_i = columns[c_i]
        table_i = name_i.split("__")[0]
        c_j = c_i + 1
        while c_j < c:
            name_j = columns[c_j]
            table_j = name_j.split("__")[0]
            if table_i != table_j:
                yield (name_i, name_j), quantiles, intersection
            c_j = c_j + 1
        c_i = c_i + 1


def process_emd(tup: tuple):
    
    name_i, name_j, k, quantile, intersection = unwrap_process_input_tuple(tup)
    with open('cache/'+name_i+'.pkl', 'rb') as pkl_file:
        c1 = pickle.load(pkl_file)
    with open('cache/'+name_j+'.pkl', 'rb') as pkl_file:
        c2 = pickle.load(pkl_file)
    if intersection:
        return k, intersection_emd(c1, c2, quantile)
    else:
        return k, quantile_emd(c1, c2, quantile)


def unwrap_process_input_tuple(tup: tuple):
    names, quantile, intersection = tup
    name_i, name_j = names
    k = str(name_i) + "|" + str(name_j)
    return name_i, name_j, k, quantile, intersection


def insert_to_dict(dc: dict, k: str, v: dict):
   
    if k not in dc:
        dc[k] = list()
    dc[k].append(v)


def transform_dict(dc: dict):

    tmp_dict = dict()
    for k, v in dc.items():
        k1, k2 = k.split("|")
        v1 = {'e': v, 'c': k2}
        v2 = {'e': v, 'c': k1}
        insert_to_dict(tmp_dict, k1, v1)
        insert_to_dict(tmp_dict, k2, v2)
    return tmp_dict


def process_columns(tup: tuple):

    column_name, data, source_name, data_type, quantiles = tup
    column = Column(column_name, data, source_name, data_type, quantiles)
    print("Processing column: ", column.get_long_name())
    column.quantile_histogram = QuantileHistogram(column.get_long_name(), column.ranks, column.size, quantiles)
    with open('cache/' + column.get_long_name() + '.pkl', 'wb') as output:
        pickle.dump(column, output, pickle.HIGHEST_PROTOCOL)


def parallel_cutoff_threshold(tup: tuple):

    A, column, threshold = tup
    name_i = column.get_long_name()
    theta = compute_cutoff_threshold(A[name_i], threshold)
    print("Cutoff threshold for ", name_i, " is ", theta)
    Nc = [(name_i, i['c']) for i in A[name_i] if i['e'] <= theta]
    return Nc


def cuttoff_column_generator(A: dict, columns: list, threshold: float):

    for column_name in columns:
        with open('cache/' + column_name + '.pkl', 'rb') as pkl_file:
            column = pickle.load(pkl_file)
        yield A, column, threshold

