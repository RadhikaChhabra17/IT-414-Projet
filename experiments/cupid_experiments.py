import os
import re

import numpy as np
import matplotlib.pyplot as plt
from numpy import argmax
from tqdm import tqdm

from os import listdir
from os.path import isfile, join

from cupid.tree_match import tree_match, recompute_wsim, mapping_generation_leaves, mapping_generation_non_leaves

CURRENT_DIR = os.path.dirname(__file__)
RDB_SCHEMA = '/home/radhika/reproducing-schema-matching/data/cupid/paper/rdb_schema.csv'
STAR_SCHEMA = '/home/radhika/reproducing-schema-matching/data/cupid/paper/star_schema.csv'


def write_mappings(mappings, filename):

    f = open(filename, 'w+')
    for m in mappings:
        f.write(str(m))
        f.write("\n")
    f.close()


def run_experiments(source_tree, target_tree, cupid_model, out_dir, leaf_range, th_accept_range):

    factor = 0.01
    for j in tqdm(leaf_range):
        dirname = out_dir + "j-" + str(j)
        os.mkdir(dirname)
        for i in tqdm(th_accept_range):
            sims = tree_match(source_tree, target_tree, cupid_model.get_categories(), th_accept=i, th_low=i - factor,
                              th_high=i + factor,
                              leaf_w_struct=j, w_struct=j + 0.1, th_ns=0.45)
            # new_sims = recompute_wsim(source_tree, target_tree, sims, th_accept=i)
            map1 = mapping_generation_leaves(source_tree, target_tree, sims, th_accept=i)
            # map2 = mapping_generation_non_leaves(source_tree, target_tree, new_sims, th_accept=i)
            print("Leaf matchings:\n {}".format(map1))
            # print("Non-leaf matchings:\n {}".format(map2))

            write_mappings(map1, '{}/test_{}.txt'.format(dirname, i))
    # write_mappings(map2, 'cupid-output/non-leaf_{}.txt'.format(i))


def read_tuple_file(filepath):

    list_tuples = list()
    f = open(filepath, 'r')
    lines = f.readlines()
    for line in lines:
        y = re.search("('*\w+[\(FK\)]*__\w+[\(FK\)]*'*), *('*\w+[\(FK\)]*__\w+[\(FK\)]*'*)", line)
        if y and len(y.groups()) == 2:
            list_tuples.append(
                (re.search("(\w+__\w+)", y.group(1)).group(), re.search("(\w+__\w+)", y.group(2)).group()))
    f.close()
    return list_tuples


def compute_precision(golden_standard, mappings):

    if len(mappings) == 0:
        return 0

    matches = [item for item in golden_standard if item in mappings]
    return len(matches) / len(mappings)


def compute_recall(golden_standard, mappings):

    if len(mappings) == 0:
        return 0

    matches = [item for item in golden_standard if item in mappings]
    return len(matches) / len(golden_standard)


def compute_f1_score(precision, recall):

    if precision == 0 and recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)


def make_plot(x, precision_list, recall_list, f1_list, name):

    plt.figure()
    plt.plot(x, precision_list, color='blue', linewidth=2, label='Precision')
    plt.plot(x, recall_list, color='green', linewidth=2, label='Recall')
    plt.plot(x, f1_list, color='red', linewidth=2, label="F1-score")
    x_tick = x[argmax(f1_list)]
    plt.plot(x_tick, max(f1_list), color='red', linewidth=2, marker="o")
    plt.legend()

    limsx = plt.xlim()
    limsy = plt.ylim()

    xinterval = np.sort(np.append(plt.xticks()[0], x_tick))
    yinterval = np.sort(np.append(plt.yticks()[0], max(f1_list)))

    posx = np.where(xinterval == x_tick)[0][0]
    posy = np.where(yinterval == max(f1_list))[0][0]

    to_remove_x = closer_to(xinterval, posx)
    to_remove_y = closer_to(yinterval, posy)
    to_append = yinterval[posy + to_remove_y] + (to_remove_y / 10)

    xinterval = np.delete(xinterval, posx + to_remove_x)
    yinterval = np.delete(yinterval, posy + to_remove_y)
    yinterval = np.append(yinterval, to_append)

    plt.xticks(xinterval)
    plt.yticks(yinterval)
    plt.xlim(limsx)
    plt.ylim(limsy)

    plt.xlabel('th_accept threshold')
    plt.ylabel('Value')
    plt.title('Precision/Recall/F1-score for w_struct_leaf = {}'.format(name))
    # plt.show()
    plt.savefig('cupid_{}.pdf'.format(name), dpi=300)


def closer_to(interval, pos):

    diff_min = abs(interval[pos] - interval[pos - 1])
    diff_max = abs(interval[pos] - interval[pos + 1])

    if diff_min < diff_max:
        # the point is lower than the maximum f1-score
        return -1
    else:
        # the point is higher than the maximum f1-score
        return 1


def compute_statistics(golden_standard_file, out_path, leaf_range, th_accept_range):

    golden_standard = read_tuple_file(golden_standard_file)

    dirs = [join(out_path, f) for f in listdir(out_path) if not isfile(join(out_path, f))]
    dirs.sort()

    for i, dir in enumerate(dirs):
        files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
        files.sort()

        precision_list = list()
        recall_list = list()
        f1_list = list()
        data_set_size = list()

        for file in files:
            tuples = read_tuple_file(file)

            precision = compute_precision(golden_standard, tuples)
            recall = compute_recall(golden_standard, tuples)
            f1 = compute_f1_score(precision, recall)

            data_set_size.append(len(tuples))
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        make_plot(th_accept_range, precision_list, recall_list, f1_list, leaf_range[i])

