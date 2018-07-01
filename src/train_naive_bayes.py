"""
train naive bayes model for each user.
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


def warning(*objs):
    print(__file__, *objs, file=sys.stderr)


def poisson(k, lamb):
    return (lamb**k / factorial(k)) * np.exp(-lamb)


def train_naive_bayes(train_ratio, nb_models, rdd_user_days):
    # train_ratio: (0,1)
    # rdd_user_days: (user, n-by-(d+1)) for all data
    # nb_models: [B, B, B, B, B, P, P, P, B, B]
    ntrain = int(train_ratio * len(rdd_user_days[1]))
    traindata = rdd_user_days[1][:ntrain, :]
    traindata[traindata == -1] = 0
    reds = traindata[:, -1]
    traindata = traindata[reds != 1, :-1]
    ntrain, dim = traindata.shape
    warning("#traindata = %d" % ntrain)
    assert dim == len(nb_models), "%d vs. %d" % (dim, len(nb_models))
    params = []
    for k, m in enumerate(nb_models):
        data = traindata[:, k]
        if m == 'B':  # Bernoulli
            p = sum([1 for val in data if val > 0]) / float(len(data))
        elif m == 'P':  # Poisson
            xy = Counter(data).items()
            xy = sorted(xy)
            th = -1
            x_values = [int(x) for x, y in xy if x>th]
            y_values = [y for x, y in xy if x>th]
            total_cnt = float(sum(y_values))
            y_values = [y/total_cnt for y in y_values]
            if len(x_values) > 1:
                try:
                    p = curve_fit(poisson, x_values, y_values, maxfev=1000)[0]
                except:
                    warning("%s, %s" % (x_values, y_values))
                    raise
            else:
                p = 1e-3
            # warning("Poisson(%d): %s, %s" % (k, x[:5], y[:5]))
        else:
            # undefined model
            raise ValueError(m)
        params.append(p)
    return (rdd_user_days[0], params)


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

    # run parallel jobs
    rdd_data = sc.parallelize(data_by_user, len(data_by_user))
    rst = rdd_data.map(partial(train_naive_bayes,
                               config.data.train_ratio,
                               config.nb.models)).collect()
    outfile = osp.join(config.io.model,
                       config.nb.prefix + "_%s.pkl" % config.nb.models)
    with open(outfile, "w+") as fp:
        cPickle.dump(rst, fp, protocol=2)
    print ("Trained %d models, saved to %s." % (len(rst), outfile))
