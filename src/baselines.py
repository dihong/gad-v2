from __future__ import print_function
from pyspark import SparkContext
from pyspark import SparkConf
import sys
sys.path.insert(0, '../safekit-master')
import tensorflow as tf
from safekit.batch import OnlineBatcher, NormalizingReplayOnlineBatcher, split_batch, DayBatcher
from safekit.graph_training_utils import ModelRunner, EarlyStop
from safekit.tf_ops import join_multivariate_inputs, dnn, \
    diag_mvn_loss, multivariate_loss, eyed_mvn_loss, \
    full_mvn_loss, layer_norm, batch_normalize
from safekit.util import get_multivariate_loss_names, make_feature_spec, make_loss_spec, Parser
import random
from operator import itemgetter
import os
import argparse
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.ensemble import IsolationForest
from config import config
import time
from multiprocessing import Pool
from util import cumulative_recall
import operator
from functools import partial
import json
import cPickle
from util import split_train_test
from util import FileName
import numpy as np
import math

# setup spark.
conf = (SparkConf()
        .setMaster(config.spark.SPARK_MASTER)
        .set("spark.app.name", __file__)
        .set("spark.executor.memory", "50g")
        .set("spark.driver.maxResultSize", "100g")
        .set("spark.ui.showConsoleProgress", "true"))
sc = SparkContext(conf=conf)

def save_rst(rst, fname):
    # rst must be [(user, day, score, red)]
    with open(fname, 'w+') as fp:
        fp.write('\n'.join(['%d\t%d\t%.6f\t%d' % (user, day, score, red)
                            for user, day, score, red in rst]))

def eval_cr(rst, name=""):
    rst = [(d,s,r) for u,d,s,r in rst]
    rst = sorted(rst, key=operator.itemgetter(0))
    for b in config.cr.budgets:
        cr_score = cumulative_recall(rst, b, config.cr.increment)
        print("CR(%s)-%d: %.4f" % (name, b, cr_score))

def load_features_txt(datafile):
    with open(datafile) as fp:
        lines = fp.read().strip().split('\n')[1:]
        ret = np.array([l.split(' ') for l in lines])
        return ret.astype(np.float)

def load_features_bin(datafile):
    datafile = datafile[:-4] + '.npy'
    return np.load(datafile)

def group_by_day(all_fixed_feat):
    ret = {}
    for f in all_fixed_feat:
        d = f[0]
        if d not in ret:
            ret[d] = []
        ret[d].append(f)
    return ret.values()

def rdd_svm(nu, kernel, gamma, shrink, rs, mat):
    model = svm.OneClassSVM(
        nu=nu, kernel=kernel, gamma=gamma, shrinking=shrink, random_state=rs)
    mat = np.array(mat)
    if mat.shape[1] == 13:
        # use compact10d
        datadict = {'features': mat[:, 3:],
                    'red': mat[:, 2],
                    'user': mat[:, 1],
                    'day': mat[:, 0]}
    else:
        # use all_fixed
        datadict = {'features': mat[:, 14:],
                    'red': mat[:, 13],
                    'user': mat[:, 1],
                    'day': mat[:, 0]}
    model.fit(datadict['features'])
    anomaly_scores = model.decision_function(datadict['features'])
    rst = []
    for day, user, red, score in zip(datadict['day'],
                                     datadict['user'],
                                     datadict['red'],
                                     anomaly_scores):
        rst.append((user, day, -score, red))
    return rst

def run_svm(data_file, rs, nu, kernel, gamma, shrink, outfile1, outfile2):
    #
    print('running SVM with nu={}, kernel={}, shrink={}'.format(nu, kernel, shrink))
    try:
        feat = load_features_bin(data_file)
    except:
        feat = load_features_txt(data_file)
        npyfile = data_file[:-4] + '.npy'
        np.save(npyfile, feat)
    feat = group_by_day(feat)
    rdd_feat = sc.parallelize(feat, len(feat))
    rst=rdd_feat.flatMap(partial(rdd_svm, nu, kernel, gamma, shrink, rs)).collect()
    train_rst, test_rst = split_train_test(rst)
    save_rst(train_rst, outfile1)
    save_rst(test_rst, outfile2)
    eval_cr(test_rst, 'svm')

def run_pca(data_file, rs, n_components, outfile1, outfile2):
    """Wrapper to run PCA model.

    Parameters
    ----------
    data_file : str
        filepath of data file
    rs : int
        random seed
    n_components: int
        PCA parameter
    outfile : str
        filepath of output file to be generated
    """
    print('running PCA with n_components={}'.format(n_components))
    day_batcher = DayBatcher(data_file, skiprow=1, delimiter=' ')
    mat = day_batcher.next_batch()
    rst = []
    while mat is not None:
        if mat.shape[1] == 13:
            # use compact10d
            datadict = {'features': mat[:, 3:],
                        'red': mat[:, 2],
                        'user': mat[:, 1],
                        'day': mat[:, 0]}
        else:
            # use all_fixed
            datadict = {'features': mat[:, 14:],
                        'red': mat[:, 13],
                        'user': mat[:, 1],
                        'day': mat[:, 0]}
        batch = scale(datadict['features'])
        pca = PCA(n_components=n_components, random_state=rs)
        pca.fit(batch)
        data_reduced = np.dot(batch, pca.components_.T)  # pca transform
        data_original = np.dot(data_reduced, pca.components_)  # inverse_transform
        pointloss = np.mean(np.square(batch - data_original), axis=1)
        loss = np.mean(pointloss)
        for d, u, t, l, in zip(datadict['day'].tolist(),
                               datadict['user'].tolist(),
                               datadict['red'].tolist(),
                               pointloss.flatten().tolist()):
            rst.append((u, d, l, t))
        mat = day_batcher.next_batch()
    train_rst, test_rst = split_train_test(rst)
    save_rst(train_rst, outfile1)
    save_rst(test_rst, outfile2)
    eval_cr(test_rst, 'pca')


def rdd_iso_forest(n_estimators, max_samples, contamination, max_features,
                   bootstrap, mat):
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                            contamination=contamination, max_features=max_features,
                            bootstrap=bootstrap, n_jobs=1, verbose=0)
    mat = np.array(mat)
    if mat.shape[1] == 13:
        # use compact10d
        datadict = {'features': mat[:, 3:],
                    'red': mat[:, 2],
                    'user': mat[:, 1],
                    'day': mat[:, 0]}
    else:
        # use all_fixed
        datadict = {'features': mat[:, 14:],
                    'red': mat[:, 13],
                    'user': mat[:, 1],
                    'day': mat[:, 0]}
    model.fit(datadict['features'])
    anomaly_scores = model.decision_function(datadict['features'])
    rst = []
    for day, user, red, score in zip(datadict['day'], datadict['user'],
                                     datadict['red'], anomaly_scores):
        if math.isnan(score) and not math.isinf(score):
            print('exiting due divergence')
            exit(1)
        else:
            rst.append((user, day, -score, red))
    return rst


def run_iso_forest(data_file, rs, n_estimators, max_samples, contamination, max_features, bootstrap, outfile1, outfile2):
    #
    print('running Isolation Forest with n_estimators={}, max_samples={}, contamination={}, max_features={}, bootstrap={}'.format(n_estimators, max_samples, contamination, max_features, bootstrap))
    try:
        feat = load_features_bin(data_file)
    except:
        feat = load_features_txt(data_file)
        npyfile = data_file[:-4] + '.npy'
        np.save(npyfile, feat)
    feat = group_by_day(feat)
    rdd_feat = sc.parallelize(feat, len(feat))
    rst=rdd_feat.flatMap(partial(rdd_iso_forest, n_estimators, max_samples,
                                 contamination, max_features, bootstrap)).collect()
    train_rst, test_rst = split_train_test(rst)
    save_rst(train_rst, outfile1)
    save_rst(test_rst, outfile2)
    eval_cr(test_rst, 'iso-forest')

def run_random(data_file, rs, outfile1, outfile2):
    print('running RANDOM model...')
    day_batcher = DayBatcher(data_file, skiprow=1, delimiter=' ')
    mat = day_batcher.next_batch()
    random.seed(rs)
    rst = []
    while mat is not None:
        if mat.shape[1] == 13:
            # use compact10d
            datadict = {'features': mat[:, 3:],
                        'red': mat[:, 2],
                        'user': mat[:, 1],
                        'day': mat[:, 0]}
        else:
            # use all_fixed
            datadict = {'features': mat[:, 14:],
                        'red': mat[:, 13],
                        'user': mat[:, 1],
                        'day': mat[:, 0]}
        anomaly_scores = [random.random() for x in datadict['features']]
        for day, user, red, score in zip(datadict['day'], datadict['user'], datadict['red'], anomaly_scores):
            rst.append((user, day, score, red))
        mat = day_batcher.next_batch()
    train_rst, test_rst = split_train_test(rst)
    save_rst(train_rst, outfile1)
    save_rst(test_rst, outfile2)
    eval_cr(test_rst, 'random')

def run_svm_test(args):
    """Run one parameter combination for SVM model."""
    test_file, outdir = args
    start = time.time()
    nu = config.svm.nu
    kernel = config.svm.kernel
    gamma = config.svm.gamma
    shrink = False
    outfile_path1, outfile_path2 = FileName.get_svm_rst_name()
    run_svm(test_file, rs, nu, kernel, gamma, shrink, outfile_path1, outfile_path2)
    dt = time.time() - start
    print("run_svm_test Done. Elapsed time is %.2f seconds." % dt)


def run_pca_test(args):
    """Run one parameter combination for PCA model."""
    test_file, outdir = args
    start = time.time()
    n_components = config.pca.n_components
    outfile_path1, outfile_path2 = FileName.get_pca_rst_name()
    run_pca(test_file, rs, n_components, outfile_path1, outfile_path2)
    dt = time.time() - start
    print("run_pca_test Done. Elapsed time is %.2f seconds." % dt)


def run_iso_forest_test(args):
    """Run one parameter combination for Isolation Forest model."""
    test_file, outdir = args
    start = time.time()
    n_estimators = config.iso_forest.n_estimators
    max_samples = 'auto'
    contamination = config.iso_forest.contamination
    max_features = 1.0  # default is 1.0 (use all features)
    bootstrap = False
    outfile_path1, outfile_path2 = FileName.get_iso_forest_rst_name()
    run_iso_forest(test_file, rs, n_estimators, max_samples, contamination,
                   max_features, bootstrap, outfile_path1, outfile_path2)
    dt = time.time() - start
    print("run_iso_forest_test Done. Elapsed time is %.2f seconds." % dt)


def run_random_test(args):
    """Run one parameter combination for random model."""
    test_file, outdir = args
    start = time.time()
    outfile_path1, outfile_path2 = FileName.get_random_rst_name()
    run_random(test_file, rs, outfile_path1, outfile_path2)
    dt = time.time() - start
    print("run_random_test Done. Elapsed time is %.2f seconds." % dt)

def run_dnn_test(args):
    data_file, data_spec_file, outdir = args
    """Run DNN model on data file given parameters."""
    nl = config.dnn.num_layers
    hs = config.dnn.hidden_size
    # io and state
    print('running DNN with nl={}, hs={}'.format(nl, hs))
    start = time.time()
    dataspecs = json.load(open(data_spec_file, 'r'))
    feature_spec = make_feature_spec(dataspecs)
    datastart_index = dataspecs['counts']['index'][0]
    normalizers = {'none': None,
                   'layer': layer_norm,
                   'batch': batch_normalize}
    tf.set_random_seed(config.state.random_seed)
    data = OnlineBatcher(data_file, config.dnn.batch_size,
                         skipheader=True, delimiter=' ')
    # activation
    if config.dnn.activation == 'tanh':
        activation = tf.tanh
    elif config.dnn.activation == 'relu':
        activation = tf.nn.relu
    else:
        raise ValueError('Activation must be "relu", or "tanh"')
    # mvn
    if config.dnn.dist == "ident":
        mvn = eyed_mvn_loss
    elif config.dnn.dist == "diag":
        mvn = diag_mvn_loss
    elif config.dnn.dist == "full":
        mvn = full_mvn_loss
        raise ValueError('dnn.dist must be "ident", "diag", or "full"')
    # setup tf model
    x, ph_dict = join_multivariate_inputs(
        feature_spec, dataspecs, 0.75, 1000, 2)
    h = dnn(x, layers=[hs for i in range(nl)],
            act=activation, keep_prob=None,
            norm=normalizers[config.dnn.normalizer],
            scale_range=1.0)
    loss_spec = make_loss_spec(dataspecs, mvn)
    loss_matrix = multivariate_loss(h, loss_spec, ph_dict, variance_floor=0.01)
    loss_vector = tf.reduce_sum(loss_matrix, reduction_indices=1)  # is MB x 1
    loss = tf.reduce_mean(loss_vector)  # is scalar
    loss_names = get_multivariate_loss_names(loss_spec)
    eval_tensors = [loss, loss_vector, loss_matrix]
    model = ModelRunner(loss, ph_dict, learnrate=config.dnn.lr, opt='adam',
                        debug=config.dnn.debug, decay_rate=1.0, decay_steps=20)
    raw_batch = data.next_batch()
    current_loss = sys.float_info.max
    not_early_stop = EarlyStop(20)
    loss_feats = [triple[0] for triple in loss_spec]
    # start training
    start_time = time.time()
    continue_training = not_early_stop(raw_batch, current_loss)
    # mat is not None and self.badcount < self.badlimit and loss != inf, nan:
    rst = []
    while continue_training:
        datadict = split_batch(raw_batch, dataspecs)
        targets = {'target_' + name: datadict[name] for name in loss_feats}
        datadict.update(targets)
        current_loss, pointloss, contrib = model.eval(datadict, eval_tensors)
        model.train_step(datadict)
        for user, day, score, red in zip(datadict['user'].flatten().tolist(),
                                         datadict['time'].flatten().tolist(),
                                         pointloss.flatten().tolist(),
                                         datadict['redteam'].flatten().tolist()):
            rst.append((user, int(day), score, red))
        if data.index % 10000 == 1:
            print('index: %s loss: %.4f' % (data.index, current_loss))
            sys.stdout.flush()
        raw_batch = data.next_batch()
        continue_training = not_early_stop(raw_batch, current_loss)
        if continue_training < 0:
            break
    # save the (user, day, score, red).
    train_rst, test_rst = split_train_test(rst)
    outfile1, outfile2 = FileName.get_dnn_rst_name()
    save_rst(train_rst, outfile1)
    save_rst(test_rst, outfile2)
    print('')
    eval_cr(test_rst, 'dnn')
    dt = time.time() - start
    print("run_dnn_test Done. Elapsed time is %.2f seconds." % dt)

if __name__ == '__main__':
    # init
    rs = config.state.random_seed
    outdir = config.io.outdir
    data_file = config.data.compact_txt
    json_file = config.data.compact_json
    run_iso_forest_test((data_file, os.path.join(outdir, 'iso_forest')))
    run_svm_test((data_file, os.path.join(outdir, 'svm')))
    pool = Pool(processes=3)
    results = []
    results.append(pool.apply_async(run_pca_test,
                                  [(data_file, os.path.join(outdir, 'pca'))]))
    results.append(pool.apply_async(run_random_test,
                                  [(data_file, os.path.join(outdir, 'random'))]))
    results.append(pool.apply_async(run_dnn_test,
                                  [(data_file, json_file,
                                    os.path.join(outdir, 'dnn'))]))
    # wait all parallel process.
    for r in results:
        r.get()
