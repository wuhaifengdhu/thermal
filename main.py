#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from os import path
from sklearn import tree
from collections import Counter

ROOT = path.dirname(path.realpath(__file__))
RESOURCE = path.join(ROOT, 'resource')
MODEL_PATH = path.join(ROOT, 'models')


def read_excel(excel_file):
    return pd.read_excel(excel_file).values


def data_set_split(raw_data, eval_set_percentage):
    row_number, column_number = raw_data.shape

    percentage = 1 if abs(eval_set_percentage) > 1 else abs(eval_set_percentage)
    eval_rows = np.random.choice(row_number, int(row_number * percentage), False)
    train_rows = list(set(range(row_number)) - set(eval_rows))
    print "rows used for training: %s" % str(train_rows)
    print "rows used for model evaluation: %s" % str(eval_rows)
    return raw_data[train_rows, :], raw_data[eval_rows, :]


def training(data_set, target_column, exclude_columns=[]):
    row_number, column_number = data_set.shape
    feature_columns = list(set(range(column_number)) - set([target_column]) - set(exclude_columns))
    return tree.DecisionTreeClassifier().fit(data_set[:, feature_columns].tolist(), data_set[:, target_column].tolist())


def predict(model, data_set, exclude_columns=[]):
    row_number, column_number = data_set.shape
    feature_columns = list(set(range(column_number)) - set(exclude_columns))
    return model.predict(data_set[:, feature_columns].tolist())


def single_model(train_set, eval_set, target_column, exclude_columns):
    model = training(train_set, target_column, exclude_columns)
    export_module_dot(model, path.join(MODEL_PATH, 'ourTestData.dot'))
    result = predict(model, eval_set, exclude_columns + [target_column])
    return result


def export_module_dot(model, file_name):
    with open(file_name, "w") as f:
        f = tree.export_graphviz(model, out_file=f)


def most_common_value(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def multi_model(train_set, eval_set, target_column, exclude_columns, model_numbers, cross_train_percentage):
    row_number, column_number = train_set.shape
    train_rows = int(row_number * cross_train_percentage)
    model_data = [train_set[np.random.choice(row_number, train_rows), :] for i in range(model_numbers)]
    results = [single_model(model_data[i], eval_set, target_column, exclude_columns) for i in range(model_numbers)]
    eval_rows = eval_set.shape[0]
    for i in range(model_numbers):
        print "model%i result: %s" % (i, '\t'.join(["%i" % number for number in results[i]]))
    final_result = [round(sum([results[j][i] for j in range(model_numbers)]) / model_numbers) for i in range(eval_rows)]
    # final_result = [most_common_value([results[j][i] for j in range(model_numbers)]) for i in range(eval_rows)]
    return final_result

if __name__ == '__main__':
    # Part 1, Our own data
    local_file1 = path.join(RESOURCE, 'data_Hu_Weizheng.xlsx')
    local_file2 = path.join(RESOURCE, 'data_Wu_Haifeng.xlsx')
    raw_data1 = read_excel(local_file1)
    raw_data2 = read_excel(local_file2)
    raw_data = np.vstack((raw_data1, raw_data2))
    train_data, eval_data = data_set_split(raw_data1, 0.2)
    single_predict = single_model(train_data, eval_data, 19, [0, 1])
    # multi_predict = multi_model(train_data, eval_data, 19, [0, 1], 5, 0.7)

    print "user real tags: %s" % '\t'.join(["%i" % number for number in eval_data[:, 19].tolist()])
    print "single predict: %s" % '\t'.join(["%i" % number for number in single_predict])
    # print "multi predict: %s" % '\t'.join(["%i" % number for number in multi_predict])

    # Part 2, tropic data from others
    # local_file = path.join(RESOURCE, 'tropic_huwz.xlsx')
    # raw_data = read_excel(local_file)
    # ground_truth_column = 13
    # exclude_columns = [0, 2, 4, 5, 11, 12]
    # eval_set_percentage = 0.3
    # train_data, eval_data = data_set_split(raw_data, eval_set_percentage)
    # ground_truth = eval_data[:, ground_truth_column].tolist()
    #
    # single_predict = single_model(train_data, eval_data, ground_truth_column, exclude_columns)
    # # multi_predict = multi_model(train_data, eval_data, ground_truth_column, exclude_columns, 9, 0.7)
    # print "user real tags: %s" % '\t'.join(["%i" % number for number in ground_truth])
    # print "single predict: %s" % '\t'.join(["%i" % number for number in single_predict])
    # # print "multi predict: %s" % '\t'.join(["%i" % number for number in multi_predict])
    # single_model_match_count = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == single_predict[i]])
    # print "match count = %i, total count = %i" % (single_model_match_count, len(ground_truth))
    # # multi_model_match_count = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == multi_predict[i]])
    # # print "match count = %i, total count = %i" % (multi_model_match_count, len(ground_truth))

    # print "single predict: %s" % str(sum([abs(single_predict[i] - ground_truth[i])
    #                                       for i in range(len(single_predict))]) * 1.0 / 785)
    # print "multi predict: %s" % str(
    #     sum([abs(multi_predict[i] - ground_truth[i]) for i in range(len(multi_predict))]) * 1.0 / 785)
