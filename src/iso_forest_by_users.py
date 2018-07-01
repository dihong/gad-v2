"""
test naive bayes model for each user.
"""
from __future__ import print_function
from pyspark import SparkContext
from pyspark import SparkConf
from os import path as osp
from functools import partial
from config import config
from operator import itemgetter
import datetime
import numpy as np
import cPickle
import sys
from sklearn.ensemble import IsolationForest
from scipy.optimize import curve_fit
from scipy.misc import factorial
from collections import Counter


def cumulative_recall(rst, budget, increment):
    # rst: n-by-3 matrix, where n is #days, and colums are [day_key, score,
    # red]
    cumulative_recall_score = 0
    total_red = 0.0
    current_red = 0
    log = dict()  # Create dictionary of days
    malform = 0
    for row in rst:
        c_day, anomaly, red = row
        if red > 0.0:
            total_red += 1.0
        pair = (anomaly, red)
        if c_day in log:
            log[c_day].append(pair)
        else:
            log[c_day] = [pair]

    # Sort the log[day] by the anomaly scores
    for key in log.keys():
        log[key].sort(key=lambda x: x[0], reverse=True)

    for index in range(budget):
        for i, key in enumerate(log.keys()):
            day = log[key]
            pair = day[index]
            if pair[1] > 0.0:
                current_red += 1
        if (index % increment) == increment - 1:
            cumulative_recall_score += current_red / total_red
    return cumulative_recall_score


def warning(*objs):
    print(__file__, *objs, file=sys.stderr)


def poisson(k, lamb):
    return (lamb**k / factorial(k)) * np.exp(-lamb)


def run_iso_forest(train_ratio, rdd_user_days):
    # train_ratio: (0,1)
    # rdd_user_days: (user, n-by-(d+1)) for all data
    max_samples = 'auto'
    max_features = 1.0
    n_estimators = 50
    contamination = 1
    bootstrap = False
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                            contamination=contamination, max_features=max_features,
                            bootstrap=bootstrap, n_jobs=-1, verbose=0)
    user, data = rdd_user_days
    data[data == -1] = 0
    reds = data[:, -1]
    data = data[:, :-1]
    ndays, dim = data.shape
    ntrain = int(train_ratio * ndays)
    model.fit(data)
    anomaly_scores = model.decision_function(data)
    anomaly_scores = anomaly_scores[ntrain:]
    reds = reds[ntrain:]
    ret = []
    day = 1
    for red, score in zip(reds, anomaly_scores):
        ret.append((day, -score, red))
        day += 1
    return ret


if __name__ == "__main__":
    conf = (SparkConf()
            .setMaster(config.SPARK_MASTER)
            .set("spark.app.name", __file__)
            .set("spark.executor.memory", "50g")
            .set("spark.driver.maxResultSize", "100g"))
    sc = SparkContext(conf=conf)

    # load all data by users
    with open(config.data.relational_feat, 'r') as fp:
        data_by_user = cPickle.load(fp).items()
    print ("Load data of %d users, each user has %d days." %
           (len(data_by_user), len(data_by_user[0][1])))

    # run parallel jobs: [day, score, red_flag]
    rdd_data = sc.parallelize(data_by_user, len(data_by_user))
    rst = rdd_data.map(partial(run_iso_forest,
                               config.data.train_ratio)).reduce(lambda x, y: x + y)
    print ("Calculated anomaly sores of %d user-days." % len(rst))

    # calculate cr.
    for bucket in config.cr.budgets:
        cr = cumulative_recall(rst, bucket, config.cr.increment)
        print ("CR-%d: %.4f" % (bucket, cr))
