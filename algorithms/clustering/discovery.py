import numpy as np
import networkx as nx
import pulp as plp
from tqdm import tqdm
import re

from clustering.utils import transform_dict, process_emd, column_combinations, \
    parallel_cutoff_threshold, cuttoff_column_generator, compute_cutoff_threshold


def compute_distribution_clusters(columns: list, threshold: float, quantiles: int = 256):
    combinations = list(column_combinations(columns, quantiles, intersection=False))

    total = len(combinations)

    A: dict = transform_dict(dict(tqdm([process_emd(i) for i in combinations], total=total)))
    print(A)

    epc = list([parallel_cutoff_threshold(j) for j in list(cuttoff_column_generator(A, columns, threshold))])
    graph = create_graph(columns, epc)

    nx.draw(graph)
    # plt.show()

    cc = list(nx.connected_components(graph))

    return cc


def compute_attributes(DC: list, threshold: float, quantiles: int = 256):

    combinations = list(column_combinations(DC, quantiles, intersection=True))

    total = len(combinations)

    I = transform_dict(dict(tqdm([process_emd(i) for i in combinations], total=total)))

    GA = dict()
    E = np.zeros((len(DC), len(DC)))

    for i in range(len(DC)):
        name_i = DC[i]

        cutoff_i = compute_cutoff_threshold(I[name_i], threshold)

        Nc = [i['c'] for i in I[name_i] if i['e'] <= cutoff_i]

        for Cj in Nc:
            E[i][DC.index(Cj)] = 1
        GA[DC[i]] = dict()

    M = E + np.dot(E, E)
    for i in range(len(DC)):
        for j in range(len(DC)):
            if M[i][j] == 0:
                GA[DC[i]][DC[j]] = -1
            else:
                GA[DC[i]][DC[j]] = 1

    return GA


def correlation_clustering_pulp(vertexes, edges):

    opt_model = plp.LpProblem(name="MIP_Model", sense=plp.LpMinimize)

    set_u = vertexes
    set_v = vertexes
    # set_w = vertexes

    x_vars = {(i, j): plp.LpVariable(cat=plp.LpInteger, lowBound=0, upBound=1, name="{0}--{1}".format(i, j))
              for i in set_u for j in set_v}

    sum1 = plp.lpSum(x_vars[i, j] for i in set_u for j in set_v if edges[i][j] == 1)
    sum2 = plp.lpSum(1 - x_vars[i, j] for i in set_u for j in set_v if edges[i][j] == -1)

    opt_model.setObjective(sum1 + sum2)

    opt_model.solve()

    result = dict()

    for v in opt_model.variables():
        result[v.name] = v.varValue

    return result


def process_correlation_clustering_result(results, columns):
    clusters = []
    for cluster in results:
        clusters.extend([k for (k, v) in cluster.items() if v == 0])
    edges_per_column = []
    for match in clusters:
        table1, column1, table2, column2 = get_columns_tables_from_match(match)
        edges_per_column.append([(table1+"__"+column1, table2+"__"+column2)])

    graph = create_graph(columns, edges_per_column)

    nx.draw(graph)
    # plt.show()

    connected_components = list(nx.connected_components(graph))

    return connected_components


def create_graph(nodes, edges_per_column):
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node)
    for edges in edges_per_column:
        graph.add_edges_from(edges)
    return graph


def get_columns_tables_from_match(match: str):
    match_obj = re.match(r'(.*)__(.*)__(.*)__(.*)', match)
    return match_obj.group(1), match_obj.group(2), match_obj.group(3), match_obj.group(4)
