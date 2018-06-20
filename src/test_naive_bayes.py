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
            if index >= len(day):
                continue
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


def test_naive_bayes(train_ratio, params, nb_models, rdd_user_days):
    # train_ratio: (0,1)
    # params: {user: array(k)}
    # rdd_user_days: (user, n-by-(d+1)) for all data
    user = rdd_user_days[0]
    params = params[user]
    ntrain = int(train_ratio * len(rdd_user_days[1]))
    testdata = rdd_user_days[1][ntrain:, :]
    testdata[testdata == -1] = 0
    reds = testdata[:, -1]
    testdata = testdata[:,:-1]
    ntest, dim = testdata.shape
    assert dim == len(params), "%d vs. %d" % (dim, len(params))
    probs = np.ones((ntest, dim), dtype=np.float)
    for k, p in enumerate(params):
        data = testdata[:, k]
        m = nb_models[k]
        if m == 'B':  # Bernoulli
            probs[:, k] = [p if val > 0 else (1 - p) for val in data]
        elif m == 'P':  # Poisson
            probs[:, k] = [poisson(int(val), p) for val in data]
        else:
            # undefined model
            raise ValueError(m)
    probs = np.prod(probs, axis=1)
    ret = []
    for k, p in enumerate(probs):
        ret.append((k, -p, reds[k]))
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

    # load learned model parameters
    infile = osp.join(config.io.model,
                      config.nb.prefix + "_%s.pkl" % config.nb.models)
    with open(infile, 'r') as fp:
        model_params = dict(cPickle.load(fp))

    # run parallel jobs
    rdd_data = sc.parallelize(data_by_user, len(data_by_user))
    rst = rdd_data.map(partial(test_naive_bayes,
                               config.data.train_ratio,
                               model_params,
                               config.nb.models)).reduce(lambda x,y:x+y)
    print ("Calculated anomaly sores of %d user-days." % len(rst))

    # calculate cr.
    for bucket in config.cr.budgets:
        cr = cumulative_recall(rst, bucket, config.cr.increment)
        print ("CR-%d: %.4f" % (bucket, cr))

