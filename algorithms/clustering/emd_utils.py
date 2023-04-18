import math
from pyemd import emd
from clustering.column_model import Column
from clustering.quantile_histogram.histogram import QuantileHistogram


def quantile_emd(column1: Column, column2: Column, quantiles: int = 256):

    histogram1 = column1.get_histogram()
    histogram2 = QuantileHistogram(column2.get_long_name(), column2.ranks, column2.size, quantiles,
                                   reference_hist=histogram1)
    if histogram2.is_empty:
        return math.inf
    return emd(histogram1.get_values, histogram2.get_values, histogram1.dist_matrix)


def intersection_emd(column1: Column, column2: Column, quantiles: int = 256):
 
    common_elements = set(list(column1.get_original_data())).intersection(set(list(column2.get_original_data())))

    # If the two columns do not share any common elements return inf
    if len(common_elements) == 0:
        return math.inf

    intersection = [x for x in list(column1.get_original_data()) + list(column2.get_original_data())
                    if x in common_elements]  # The intersection of the two columns

    intersection_column = Column("Intersection of " + column1.get_long_name() + " " + column2.get_long_name(),
                                 intersection, "", "", quantiles)

    e1 = quantile_emd(column1, intersection_column, quantiles)
    e2 = quantile_emd(column2, intersection_column, quantiles)

    del common_elements, intersection, intersection_column

    return (e1 + e2) / 2
