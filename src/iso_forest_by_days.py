"""
train iso forest model.
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

def warning(*objs):
    print(__file__, *objs, file=sys.stderr)

def split_train_test(data_by_user, train_ratio):
    # data_by_user: {user:array(n,k+1)} 
    all_users = sorted(data_by_user.keys())
    user_to_index = {user:i for i,user in enumerate(all_users)}
    data_by_user = data_by_user.items()
    ndays, dim = data_by_user[0][1].shape
    nusers = len(data_by_user)
    ret = np.zeros(shape=(ndays, nusers, dim), dtype=np.float)
    for user, all_days_data in data_by_user:
        user_index = user_to_index[user]
        for day_index, row in enumerate(all_days_data):
            ret[day_index, user_index, :] = row
    assert train_ratio > 0 and train_ratio < 1
    ntrain = int(train_ratio*ndays)
    traindata = ret[:ntrain]
    testdata = ret[ntrain:]
    return traindata, testdata

def cumulative_recall(rst, budget, increment):
    # rst: n-by-3 matrix, where n is #days, and colums are [day_key, score, red]
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

def iso_forest(b_data, buckets, increment, rdd_param):
    # b_data: ndays-nusers-nfeat matrix.
    n_estimators, contamination,  bootstrap = rdd_param
    max_samples = 'auto'
    max_features = 1.0
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                            contamination=contamination, max_features=max_features,
                            bootstrap=bootstrap, n_jobs=-1, verbose=0)
    rst = []
    dim = b_data.value.shape[2] - 1
    for day, user_feat in enumerate(b_data.value):
        reds = user_feat[:,-1]
        features = user_feat[:,:-1]
        model.fit(features)
        anomaly_scores = model.decision_function(features)
        for red, score in zip(reds, anomaly_scores):
            rst.append((day, -1*score, red))
    ret = []
    for bucket in buckets:
        cr = cumulative_recall(rst, bucket, increment)
        ret.append((bucket, (cr, rdd_param)))
    warning(ret)
    return ret



if __name__ == "__main__":
    conf = (SparkConf()
            .setMaster(config.SPARK_MASTER)
            .set("spark.app.name", __file__)
            .set("spark.executor.memory", "50g")
            .set("spark.driver.maxResultSize", "100g"))
    sc = SparkContext(conf=conf)

    # split train/test data
    with open(config.data.relational_feat, 'r') as fp:
        data_by_user = cPickle.load(fp)
    traindata, testdata = split_train_test(data_by_user, config.data.train_ratio)
    print ("Train with %d days, and test with %d days." % (len(traindata),
                                                           len(testdata)))
    b_traindata = sc.broadcast(testdata)
    b_testdata = sc.broadcast(testdata)

    # enumerate all parameters
    params = []
    for n_estimators in config.iso_forest.n_estimators:
        for contamination in config.iso_forest.contamination:
            for bootstrap in config.iso_forest.bootstrap:
                params.append((n_estimators, contamination, bootstrap))

    # run parallel jobs: [(bucket, [(cr, param)])]
    rdd_params = sc.parallelize(params, len(params))
    result = rdd_params.flatMap(partial(iso_forest, b_traindata,
                                        config.cr.budgets,
                                        config.cr.increment))\
        .groupByKey()\
        .mapValues(list)\
        .collect()
    best_params = {}
    for bucket, crs in result:
        r = sorted(crs, reverse=True)[0]
        best_params[bucket] = {'cr':r[0],
                                'n_estimators': r[1][0],
                                'contamination': r[1][1],
                                'bootstrap': r[1][2]}
    print("Best iso_forest parameters: %s" % best_params)
    with open(osp.join(config.io.model, "hp_iso_forest.pkl"), "w+") as fp:
        cPickle.dump(best_params, fp, protocol=2)

