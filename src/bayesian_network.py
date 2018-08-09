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
from util import FileName, Bcolors


def bn_prediction(header, test_feat):
    test_feat = np.array(test_feat)
    pdata = pd.DataFrame(
        data={k: v for k, v in zip(header, test_feat[:, 2:].T)})
    pdata.drop('R', axis=1, inplace=True)
    for k in range(config.bn.observed_neighbor.num_periods):
        pdata.drop('L%d' % k, axis=1, inplace=True)
    probs = model.predict_probability(pdata)
    R1 = probs[['R_1']]  # probability of being anomalous
    return R1.values

if __name__ == "__main__":
    # setup spark.
    conf = (SparkConf()
            .setMaster(config.spark.SPARK_MASTER)
            .set("spark.app.name", __file__)
            .set("spark.executor.memory", "50g")
            .set("spark.driver.maxResultSize", "100g")
            .set("spark.ui.showConsoleProgress", "true"))
    sc = SparkContext(conf=conf)
    # all the rst_files to be tested.
    all_rst_files = [FileName.get_dnn_rst_name(),
                     FileName.get_pca_rst_name(),
                     FileName.get_random_rst_name(),
                     FileName.get_iso_forest_rst_name(),
                     FileName.get_svm_rst_name()]
    #fp = open('cpds.txt', 'w+')
    cpds_out = []
    for rst_train_file, rst_test_file in all_rst_files:
        # load bnfeat.
        train_file, test_file = FileName.get_bn_feat_name(rst_train_file,
                                                          rst_test_file)
        with open(train_file, 'rb') as fp:
            train_feat = np.array(cPickle.load(fp), dtype=np.int)
        with open(test_file, 'rb') as fp:
            test_feat = np.array(cPickle.load(fp), dtype=np.int)
        print("Load data from %s and %s" % (train_file, test_file))
        for k in range(2, train_feat.shape[1]):
            mx = max(train_feat[:, k])
            test_feat[test_feat[:, k] > mx, k] = mx
        assert max(train_feat[:,-1]) < 2, max(train_feat[:,-1])
        # prepare train data
        header = ['R', 'T']
        for k in range(config.bn.observed_neighbor.num_periods):
            header.append('M%d' % k)
            header.append('C%d' % k)
        for k in range(config.bn.observed_neighbor.num_periods):
            header.append('L%d' % k)
        pdata = pd.DataFrame(
            data={k: v for k, v in zip(header, train_feat[:, 2:].T)})
        # train bn model: by default fit() will use MLE.
        edges = [('T', 'R')]
        for k in range(config.bn.observed_neighbor.num_periods):
            edges.append(('L%d' % k, 'R'))
            edges.append(('M%d' % k, 'L%d' % k))
            edges.append(('C%d' % k, 'L%d' % k))
        model = BayesianModel(edges)
        model.fit(pdata) 
        # print cpds into file.
        cpds_out.append(rst_train_file.split('/')[-1][:3])
        for cpd in model.get_cpds():
            cpds_out.append(str(cpd))
        continue
        # make prediction
        rdd_test_feat = sc.parallelize(
            np.array_split(test_feat, config.spark.cores), config.spark.cores)
        probs = rdd_test_feat.flatMap(partial(bn_prediction, header)).collect()
        # calculate cr
        rst = []
        for f, p in zip(test_feat, probs):
            rst.append((f[1], p, f[2]))
        rst = sorted(rst, key=operator.itemgetter(0))
        for b in config.cr.budgets:
            cr_score = cumulative_recall(rst, b, config.cr.increment)
            print(Bcolors.WARNING+"CR(%s)-%d: %.4f" %
                  (test_file, b, cr_score)+Bcolors.ENDC)
    outfile = os.path.join(config.io.cache, 'cpds.txt')
    with open(outfile, 'w+') as fp:
        fp.write('\n'.join(cpds_out))
    print("Saved conditional distribution probability to %s." % outfile)
