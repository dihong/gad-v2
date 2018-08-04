from __future__ import print_function
from pyspark import SparkContext
from pyspark import SparkConf
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
import numpy as np
import os
import cPickle
from config import config
import random
from functools import partial
from util import cumulative_recall
import operator


def create_train_data(train_feat):
    # Deprecated!!
    # train_feat: [user, day, red, target, s1, c1, s2, c2, l1, l2]
    train_feat = np.array(train_feat, dtype=np.int)
    # select all data points with red = 1.
    train_reds = train_feat[train_feat[:, 2] > 0]
    train_feat = train_feat[train_feat[:, 2] == 0]
    # select all data points with l=1
    train_neg_latent = train_feat[
        np.any(train_feat[:, -config.bn.observed_neighbor.num_periods:], axis=1)]
    train_feat = train_feat[
        np.all(train_feat[:, -config.bn.observed_neighbor.num_periods:] == 0, axis=1)]
    random.shuffle(train_neg_latent)
    # randomly select the negative samples of same size with neg_latent.
    num_neg_other = -1
    random.shuffle(train_feat)
    train_neg_other = train_feat[:num_neg_other]
    # merge all training data.
    traindata = np.concatenate((train_reds, train_neg_latent, train_neg_other))
    print('Train with %d positive instances, %d negative (latent>0), total %d.' %
          (len(train_reds), len(train_neg_latent), len(traindata)))
    return traindata

def bn_prediction(header, test_feat):
    test_feat = np.array(test_feat)
    pdata = pd.DataFrame(data={k:v for k,v in zip(header, test_feat[:,2:].T)})
    pdata.drop('R', axis=1, inplace=True)
    for k in range(config.bn.observed_neighbor.num_periods):
        pdata.drop('L%d'%k, axis=1, inplace=True)
    probs = model.predict_probability(pdata)
    R1 = probs[['R_1']] # probability of anonymous
    return R1.values

if __name__ == "__main__":
    # setup spark.
    conf = (SparkConf()
            .setMaster(config.SPARK_MASTER)
            .set("spark.app.name", __file__)
            .set("spark.executor.memory", "50g")
            .set("spark.driver.maxResultSize", "100g"))
    sc = SparkContext(conf=conf)
    # load train data
    train_file = os.path.join('../cache/dnn_nl-5_hs-10_train_bn_feat.pkl')
    with open(train_file, 'rb') as fp:
        train_feat = np.array(cPickle.load(fp), dtype=np.int)
    # train bn model
    header = ['R', 'T']
    for k in range(config.bn.observed_neighbor.num_periods):
        header.append('M%d'%k)
        header.append('C%d'%k)
    for k in range(config.bn.observed_neighbor.num_periods):
        header.append('L%d'%k)
    pdata = pd.DataFrame(data={k:v for k,v in zip(header, train_feat[:,2:].T)})
    edges = [('T','R')]
    for k in range(config.bn.observed_neighbor.num_periods):
        edges.append(('L%d'%k, 'R'))
        edges.append(('M%d'%k, 'L%d'%k))
        edges.append(('C%d'%k, 'L%d'%k))
    model = BayesianModel(edges)
    model.fit(pdata)
    # load test data
    test_file = os.path.join('../cache/dnn_nl-5_hs-10_test_bn_feat.pkl')
    with open(test_file, 'rb') as fp:
        test_feat = np.array(cPickle.load(fp), dtype=np.int)
    for k in range(2, train_feat.shape[1]):
        mx = max(train_feat[:,k])
        test_feat[test_feat[:,k]>mx, k] = mx
    print("Load test data with shape %dx%d" % test_feat.shape)
    # parallelize data.
    rdd_test_feat = sc.parallelize(np.split(test_feat, 50))
    probs = rdd_test_feat.flatMap(partial(bn_prediction, header)).collect()
    # calculate the cr.
    rst = []
    for f,p in zip(test_feat, probs):
        rst.append((f[1], p, f[2]))
    rst = sorted(rst, key=operator.itemgetter(0))
    for b in config.cr.budgets:
        cr_score = cumulative_recall(rst, b, config.cr.increment)
        print("CR(bn)-%d: %.4f" % (b, cr_score))



