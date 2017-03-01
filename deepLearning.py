# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from os import path

import pandas as pd
import tensorflow as tf

ROOT = path.dirname(path.realpath(__file__))
RESOURCE = path.join(ROOT, 'resource')
MODEL_PATH = path.join(ROOT, 'models')

COLUMNS = ["sub", "sex", "acceptable", "met", "clo", "upholst", "insul", "airTemp", "meanTemp", "airspeed", "humidity",
           "pmv", "ppd", "groundTruth"]
LABEL_COLUMN = "groundTruth"
CATEGORICAL_COLUMNS = ["sex"]
CONTINUOUS_COLUMNS = ["acceptable", "met", "clo", "upholst", "insul", "airTemp", "meanTemp", "airspeed", "humidity",
                      "ppd"]


def build_estimator(model_dir, model_type):
    """Build an estimator."""
    # Sparse base columns.
    gender = tf.contrib.layers.real_valued_column("sex")

    # Continuous base columns.
    acceptable = tf.contrib.layers.real_valued_column("acceptable")
    met = tf.contrib.layers.real_valued_column("met")
    clo = tf.contrib.layers.real_valued_column("clo")
    upholst = tf.contrib.layers.real_valued_column("upholst")
    insul = tf.contrib.layers.real_valued_column("insul")
    airTemp = tf.contrib.layers.real_valued_column("airTemp")
    meanTemp = tf.contrib.layers.real_valued_column("meanTemp")
    airspeed = tf.contrib.layers.real_valued_column("airspeed")
    humidity = tf.contrib.layers.real_valued_column("humidity")
    ppd = tf.contrib.layers.real_valued_column("ppd")

    # Wide columns and deep columns.
    wide_columns = [gender, acceptable, met, clo, upholst, insul, airTemp, meanTemp, airspeed, humidity, ppd]
    deep_columns = [gender, acceptable, met, clo, upholst, insul, airTemp, meanTemp, airspeed, humidity, ppd]

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50])
    return m


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[str(LABEL_COLUMN)].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps):
    """Train and evaluate the model."""
    train_file_name = path.join(RESOURCE, 'train.xlsx')
    test_file_name = path.join(RESOURCE, 'eval.xlsx')
    # df_train = pd.read_csv(
    #     train_file_name,
    #     names=COLUMNS,
    #     skipinitialspace=True,
    #     engine="python")
    # df_test = pd.read_csv(
    #     test_file_name,
    #     names=COLUMNS,
    #     skipinitialspace=True,
    #     skiprows=1,
    #     engine="python")
    strConverter = {k: float for k in COLUMNS}
    df_train = pd.read_excel(train_file_name, converters=strConverter)
    df_test = pd.read_excel(test_file_name, converters=strConverter)

    # remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)

    model_dir = MODEL_PATH if not model_dir else model_dir
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir, model_type)
    m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
    train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Base directory for output models."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="wide_n_deep",
        help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=2000,
        help="Number of training steps."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="",
        help="Path to the training data."
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="",
        help="Path to the test data."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
