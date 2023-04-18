import numpy as np
import pickle


class Column(object):

    def __init__(self, name: str, data: list, source_name: str, data_type: str, quantiles: int):

        self.__long_name = source_name + '__' + name
        self.__name = name
        self.data = list(filter(lambda d: d != '', data))  # remove the empty strings
        self.__data_type = data_type
        self.quantiles = quantiles
        self.ranks = self.get_global_ranks(self.data)
        self.cardinality = len(set(data))
        self.size = len(data)
        self.quantile_histogram = None

    def get_histogram(self):
        return self.quantile_histogram

    def get_original_name(self):
        return self.__name

    def get_original_data(self):
        return self.data

    def get_long_name(self):
        return self.__long_name

    def get_data_type(self):
        return self.__data_type

    @staticmethod
    def get_global_ranks(column: list):
        with open('cache/global_ranks/ranks.pkl', 'rb') as pkl_file:
            global_ranks: dict = pickle.load(pkl_file)
            ranks = np.array(sorted([global_ranks[x] for x in column]))
            return ranks
