import timeit
import json
from pandas import DataFrame
# from ...definitions import ROOT_DIR
# # from ..definitions import ROOT_DIR
import re

import clustering.discovery as discovery
from clustering.utils import process_columns
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class CorrelationClustering:

    def __init__(self, quantiles: int, threshold1: float, threshold2: float):

        self.quantiles = quantiles
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.columns = list()

    def add_data(self, data: DataFrame, source_name: str):
        for column_name in data.columns:
            process_columns((column_name, data[column_name], source_name, data.dtypes[column_name], self.quantiles))

        self.columns = self.columns + list(map(lambda name: source_name + '__' + name, data.columns))

    def find_matches(self):
        start = timeit.default_timer()

        print("Compute distribution clusters ...\n")

        connected_components = discovery.compute_distribution_clusters(self.columns, self.threshold1, self.quantiles)

        stop = timeit.default_timer()

        print('Time: ', stop - start)

        self.write_clusters_to_json(connected_components,
                                    'Distribution_Clusters.json')

        start = timeit.default_timer()

        print("Compute attributes ... \n")
        all_attributes = list()
        for components in connected_components:
            if len(components) > 1:
                edges = discovery.compute_attributes(list(components), self.threshold2, self.quantiles)
                all_attributes.append((list(components), edges))

        print(all_attributes)

        stop = timeit.default_timer()

        print('Time: ', stop - start)

        start = timeit.default_timer()

        print("Solve linear program ... \n")
        results = list()
        for components, edges in all_attributes:
            results.append(discovery.correlation_clustering_pulp(components, edges))

        stop = timeit.default_timer()

        print('Time: ', stop - start)

        start = timeit.default_timer()

        print("Extract clusters ... \n")

        attribute_clusters = discovery.process_correlation_clustering_result(results, self.columns)

        stop = timeit.default_timer()

        print('Time: ', stop - start)

        # self.print_info(attribute_clusters)
        self.write_clusters_to_json(attribute_clusters,
                                    'Attribute_Clusters(Matches).json')

    @staticmethod
    def write_clusters_to_json(clusters: list, file_name: str):

        d = {}
        clusters.sort(key=lambda item: -len(item))
        for (cluster, idx) in zip(clusters, range(len(clusters))):
            d["Cluster " + str(idx + 1)] = list(cluster)
        with open(ROOT_DIR + "/" + file_name, 'w') as fp:
            json.dump(d, fp, indent=2)

    @staticmethod
    def print_info(clusters):
        i = 0
        for cluster in clusters:
            i = i + 1
            entries = []
            for entry in cluster:
                match_obj = re.match(r'(.*)__(.*)_(.*)', entry)
                entries.append(match_obj.group(3))
            unique_num = len(set(entries))
            total = len(entries)
            print("Cluster ", i, " number of unique ", unique_num, " out of ", total)
