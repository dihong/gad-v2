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

def temporal_knn(rst, w, alpha):
    # w = 0, 1, 2... where w = 0 means not using knn.
    # alpha \in [0,1]
    # s = alpha*s+(1-alpha)*n
    # n is mean of elements within 2w+1 window size (excluding self).
    assert w >= 0
    assert alpha>=0 and alpha<=1
    w += 1  # such that w = 1 means using 1 neighbor.
    user_data = {}
    for user, day, score, red in rst:
        if user not in user_data:
            user_data[user] = []
        user_data[user].append((day, score, red))
    ret = []
    for user, data in user_data.iteritems():
        data = sorted(data)
        head = 0
        asum = 0.0
        ac = []
        max_score = []
        min_score = []
        k = 1
        win_elems = []
        for day, score, red in data:
            asum += score
            win_elems.append(score)
            if k >= w: # reach a valid asum
                ac.append(asum)
                max_score.append(max(win_elems))
                min_score.append(min(win_elems))
                asum -= data[head][1]
                win_elems.pop(0)
                head += 1
            k += 1
        for k, val in enumerate(data):
            L = k-w
            H = k+1
            day, score, red = val
            if L>=0 and H<len(ac):
                # score = alpha*score+(1-alpha)*(ac[L]+ac[H])/(2*w)
                score = alpha*score+(1-alpha)*max(score, max_score[L],max_score[H])
            ret.append((day, score, red))
    return ret

def get_anonymous_score_filename(nl, hs):
    return "dnn_nl-%d_hs-%d" % (nl, hs)


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
    from util import split_train_test
    train_rst, test_rst = split_train_test(rst)
    fn = os.path.join("../results/dnn/train", get_anonymous_score_filename(nl, hs))
    with open(fn+'_train', 'w+') as fp:
        fp.write('\n'.join(['%d\t%d\t%.6f\t%d' % (user, day, score, red)
                            for user, day, score, red in train_rst]))
    fn = os.path.join("../results/dnn/test", get_anonymous_score_filename(nl, hs))
    with open(fn+'_test', 'w+') as fp:
        fp.write('\n'.join(['%d\t%d\t%.6f\t%d' % (user, day, score, red)
                            for user, day, score, red in test_rst]))

    # use daily (mean,std) normalized as score.
    rst_daystdnorm = []
    day_rst = {}
    for user, day, score, red in rst:
        if day not in day_rst:
            day_rst[day] = []
        day_rst[day].append((user, score, red))
    for day, data in day_rst.iteritems():
        values = [score for user, score, red in data]
        meanval = np.mean(values)
        stdval = np.std(values)
        data = sorted(data, key=itemgetter(1), reverse=False)
        for user, score, red in data:
            rst_daystdnorm.append((user, day,
                                  (score-meanval)/stdval, red))
    # use daily [0,1] normalized as score.
    rst_day01norm = []
    day_rst = {}
    for user, day, score, red in rst:
        if day not in day_rst:
            day_rst[day] = []
        day_rst[day].append((user, score, red))
    for day, data in day_rst.iteritems():
        values = [score for user, score, red in data]
        minval = min(values)
        maxval = max(values)
        data = sorted(data, key=itemgetter(1), reverse=False)
        for user, score, red in data:
            rst_day01norm.append((user, day,
                                  (score-minval)/(maxval-minval), red))
    # use rank as score.
    rst_rank = []
    day_rst = {}
    for user, day, score, red in rst:
        if day not in day_rst:
            day_rst[day] = []
        day_rst[day].append((user, score, red))
    for day, data in day_rst.iteritems():
        data = sorted(data, key=itemgetter(1), reverse=False)
        rank = 1
        for user, score, red in data:
            rst_rank.append((user, day, rank, red))
            rank += 1
    # apply temporal relation implemented by knn.
    rst = temporal_knn(rst_daystdnorm, w=7, alpha=0.5)
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
    sys.stdout.flush()
    print("-------------------------------------")
    return ret

if __name__ == "__main__":
    print config.dnn.items()
    data_file = '../r6.2/count/features/all_fixed.txt'
    data_spec = '../r6.2/count/features/all_fixed.json'
    data_file = '../r6.2/count/features/compact10d.txt'
    data_spec = '../r6.2/count/features/compact10d.json'
    t_start = time.time()
    # run_dnn(data_file, data_spec, nl=7, hs=10)
    run_dnn(data_file, data_spec, nl=5, hs=10)
    print("Elapsed time is %.2f seconds." % (time.time()-t_start))
    """ Parameter Exploration
    for hs in config.dnn.hidden_size:
        for nl in config.dnn.num_layers:
            run_dnn(data_file, data_spec, nl, hs)
    """

