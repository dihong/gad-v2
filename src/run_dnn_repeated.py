import os
import sys
from util import cumulative_recall
from config import config
sys.path.insert(0, '../safekit-master')
import tensorflow as tf
import numpy as np
import json
from safekit.batch import OnlineBatcher, NormalizingReplayOnlineBatcher, split_batch
from safekit.graph_training_utils import ModelRunner, EarlyStop
from safekit.tf_ops import join_multivariate_inputs, dnn, \
    diag_mvn_loss, multivariate_loss, eyed_mvn_loss, \
    full_mvn_loss, layer_norm, batch_normalize
from safekit.util import get_multivariate_loss_names, make_feature_spec, make_loss_spec, Parser
import time
import random
from operator import itemgetter


def run_dnn(data_file, data_spec_file, nl, hs):
    """Run DNN model on data file given parameters."""
    # io and state
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
    loss_feats = [triple[0] for triple in loss_spec]
    # start training: stage 1.
    print("===stage I===")
    rst = []
    data = OnlineBatcher(data_file, config.dnn.batch_size,
                         skipheader=True, delimiter=' ')
    raw_batch = data.next_batch()
    not_early_stop = EarlyStop(20)
    current_loss = sys.float_info.max
    continue_training = not_early_stop(raw_batch, current_loss)
    while continue_training:
        datadict = split_batch(raw_batch, dataspecs)
        targets = {'target_' + name: datadict[name] for name in loss_feats}
        datadict.update(targets)
        current_loss, pointloss, contrib = model.eval(datadict, eval_tensors)
        model.train_step(datadict)
        for day, score, red in zip(datadict['time'].flatten().tolist(),
                               pointloss.flatten().tolist(),
                               datadict['redteam'].flatten().tolist()):
            rst.append((int(day), score, red))
        if data.index % 10000 == 1:
            print('index: %s loss: %.4f' % (data.index, current_loss))
            sys.stdout.flush()
        raw_batch = data.next_batch()
        continue_training = not_early_stop(raw_batch, current_loss)
        if continue_training < 0:
            break
    # calculate CR for training and testing.
    rst = sorted(rst, key=itemgetter(0), reverse=False)
    ntrain = int(len(rst)*config.data.train_ratio)
    last_train_day = rst[ntrain][0]
    train_rst = [r for r in rst if r[0]<=last_train_day]
    test_rst = [r for r in rst if r[0]>last_train_day]
    print('#train user-days: %d, #test user-days: %d.'%(len(train_rst),
                                                        len(test_rst)))
    rst_all = [('Train', train_rst), ('Test', test_rst)]
    ret = {}
    for name_rst, rst in rst_all:
        for b in config.cr.budgets:
            cr_score = cumulative_recall(rst, b, config.cr.increment)
            print ("%s CR at %d (hs = %d, nl = %d): %.4f" %
                   (name_rst, b, hs, nl, cr_score))
            ret[(name_rst, b)] = cr_score
    # start training: stage 2.
    print("===stage II===")
    rst = []
    data = OnlineBatcher(data_file, config.dnn.batch_size,
                         skipheader=True, delimiter=' ')
    raw_batch = data.next_batch()
    not_early_stop = EarlyStop(20)
    current_loss = sys.float_info.max
    continue_training = not_early_stop(raw_batch, current_loss)
    while continue_training:
        datadict = split_batch(raw_batch, dataspecs)
        targets = {'target_' + name: datadict[name] for name in loss_feats}
        datadict.update(targets)
        current_loss, pointloss, contrib = model.eval(datadict, eval_tensors)
        model.train_step(datadict)
        for day, score, red in zip(datadict['time'].flatten().tolist(),
                               pointloss.flatten().tolist(),
                               datadict['redteam'].flatten().tolist()):
            rst.append((int(day), score, red))
        if data.index % 10000 == 1:
            print('index: %s loss: %.4f' % (data.index, current_loss))
            sys.stdout.flush()
        raw_batch = data.next_batch()
        continue_training = not_early_stop(raw_batch, current_loss)
        if continue_training < 0:
            break
    # calculate CR for training and testing.
    rst = sorted(rst, key=itemgetter(0), reverse=False)
    ntrain = int(len(rst)*config.data.train_ratio)
    last_train_day = rst[ntrain][0]
    train_rst = [r for r in rst if r[0]<=last_train_day]
    test_rst = [r for r in rst if r[0]>last_train_day]
    print('#train user-days: %d, #test user-days: %d.'%(len(train_rst),
                                                        len(test_rst)))
    rst_all = [('Train', train_rst), ('Test', test_rst)]
    ret = {}
    for name_rst, rst in rst_all:
        for b in config.cr.budgets:
            cr_score = cumulative_recall(rst, b, config.cr.increment)
            print ("%s CR at %d (hs = %d, nl = %d): %.4f" %
                   (name_rst, b, hs, nl, cr_score))
            ret[(name_rst, b)] = cr_score
    print("-------------------------------------")
    sys.stdout.flush()
    return ret

if __name__ == "__main__":
    print config.dnn.items()
    data_file = '../r6.2/count/features/all_fixed.txt'
    data_spec = '../r6.2/count/features/all_fixed.json'
    data_file = '../r6.2/count/features/compact10d.txt'
    data_spec = '../r6.2/count/features/compact10d.json'
    for hs in config.dnn.hidden_size:
        for nl in config.dnn.num_layers:
            run_dnn(data_file, data_spec, nl, hs)

