"""
This script extracts features from anonymous scores for Bayesian Network.
"""

from __future__ import print_function
from pyspark import SparkContext
from pyspark import SparkConf
from config import config
import os
import numpy as np
import operator
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.lines as plt_lines
from matplotlib import collections as mc
import itertools
from functools import partial
from util import cumulative_recall
import cPickle
import copy


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def plot_1std_scores(x, y, s, y2, s2, outfile, params):
    lines = []
    lines2 = []
    for k, val in enumerate(x):
        vx = x[k]
        vy = y[k]
        vs = s[k]
        l = [(vx, vy - vs), (vx, vy + vs)]
        l2 = [(vx + 0.5, y2[k] - s2[k]), (vx + 0.5, y2[k] + s2[k])]
        lines.append(l)
        lines2.append(l2)
    lc = mc.LineCollection(lines, linewidths=0.5, color=[0, 0.2, 0.8])
    lc2 = mc.LineCollection(lines2, linewidths=0.5, color=[0, 0.8, 0.2])
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.add_collection(lc2)
    if params is not None:
        fity = [func(xval, *params) for xval in x]
        plt.plot(x, fity, '--', linewidth=2, color='red')
    ax.autoscale()
    ax.margins(0.1)
    fig.savefig(outfile)
    plt.close(fig)


def extract_xy_from_rst(rst):
    day_scores = {}
    for u, d, s, r in rst:
        if d not in day_scores:
            day_scores[d] = []
        day_scores[d].append(s)
    for k in day_scores.keys():
        day_scores[k] = (np.mean(day_scores[k]), np.std(day_scores[k]))
    day_scores = day_scores.items()  # [(day, (mean_score, std))]
    day_scores = sorted(day_scores, key=operator.itemgetter(0), reverse=False)
    xdata = [day for day, vals in day_scores]
    ydata = [vals[0] for day, vals in day_scores]
    stds = [vals[1] for day, vals in day_scores]
    return xdata, ydata, stds


def train_rectification(rst):
    # rectify the exponential decreasing trends.
    xdata, ydata, stds = extract_xy_from_rst(rst)
    # fit parameters.
    popt, pcov = curve_fit(func, xdata, ydata)
    return popt


def apply_rectification(rst, params):
    a, b, c = params
    x = []
    y = []
    for u, d, s, r in rst:
        x.append(d)
        y.append(s)
    x = np.array(x)
    y = np.array(y)
    new_y = (y - func(x, *params)) + c
    for k in range(len(rst)):
        rst[k][2] = new_y[k]


def calculate_ticks(rst, intervals):
    scores = [s for u, d, s, r in rst]
    scores = sorted(scores, reverse=True)  # larger number goes first.
    ticks = []
    for s, t in intervals:
        ticks.append(scores[int(t * len(scores)) - 1])
    ticks[0] = scores[0]
    ticks[-1] = scores[-1]
    return ticks


def extract_user_feat(target_ticks, neighbor_ticks, user_data):
    # user_data: [(user, day, score, red)] for one user.
    ndays = len(user_data)
    user_data = sorted(user_data, key=operator.itemgetter(1), reverse=False)
    day_scores = [(d, s) for u, d, s, r in user_data]
    day_scores = sorted(day_scores, key=operator.itemgetter(1), reverse=True)
    target_sid = 0
    neighbor_sid = 0
    target_states = []
    neighbor_states = [] # [(day, observed_state)]
    for d, s in day_scores:
        target_states.append((d, target_sid))
        if s < target_ticks[target_sid]:
            target_sid += 1
            target_sid = min(target_sid, len(target_ticks)-1)
        neighbor_states.append((d, neighbor_sid))
        if s < neighbor_ticks[neighbor_sid]:
            neighbor_sid += 1
            neighbor_sid = min(neighbor_sid, len(neighbor_ticks)-1)
    target_states = sorted(target_states)
    neighbor_states = sorted(neighbor_states)
    # [k] is state of day[k]
    target_states = [state for d, state in target_states]
    neighbor_states = [state for d, state in neighbor_states]
    # windist[k] is the distribution of states for a window.
    windist = np.zeros(len(neighbor_ticks), dtype=np.int)
    winsize = config.bn.observed_neighbor.timespan
    num_periods = config.bn.observed_neighbor.num_periods
    # [(min_state, count)]
    userday_neighbor = np.inf*np.ones(shape=(num_periods, ndays, 2), dtype=np.int)
    userday_numreds = np.zeros(shape=(num_periods, ndays), dtype=np.int) # number of reds.
    win_num_reds = 0
    for pos in range(ndays + winsize - 1):
        lb = pos - winsize
        if pos < ndays:
            windist[neighbor_states[pos]] += 1  # add one value to window.
            if user_data[pos][3] > 0:  # redteam.
                win_num_reds += 1
        if lb >= 0:
            windist[neighbor_states[lb]] -= 1  # remove one value from window.
            assert windist[neighbor_states[lb]] >= 0
            if user_data[lb][3] > 0:  # redteam.
                win_num_reds -= 1
                assert win_num_reds >= 0, win_num_reds
        # calculate min_state, count for current windist
        min_state = None
        for s in range(len(windist)):
            assert windist[s] >= 0
            if windist[s] > 0:
                min_state = s
                min_state_count = windist[s]
                break  # find the first and break.
        assert min_state is not None
        # update state for t = pos+k*winsize+1, k = -num_periods,...,
        # 0,1,...,num_periods-1.
        for k in range(-num_periods, num_periods):
            update_index = pos + k * winsize + 1
            period_id = k
            if k < 0:
                update_index -= 1  # need to move back one day.
                period_id = -1 - k  # need to correct the period_id.
            if update_index >= 0 and update_index < ndays:
                # accumulate number of reds to update_index
                userday_numreds[period_id][update_index] += win_num_reds
                # update the min_state and min_state_count.
                if min_state < userday_neighbor[period_id][update_index][0]:
                    userday_neighbor[period_id][update_index] = [
                        min_state, min_state_count]
                elif min_state == userday_neighbor[period_id][update_index][0]:
                    # both sides have the same min_state, add the count!
                    userday_neighbor[period_id][update_index][1] += min_state_count
    if config.bn.use_overlapping_periods:
        # use overlapping periods.
        for period_id in range(1, num_periods):
            for dayid in range(ndays):
                prev_min_state, prev_cnt = userday_neighbor[period_id-1][dayid]
                cur_min_state, cur_cnt = userday_neighbor[period_id][dayid]
                if prev_min_state==cur_min_state:
                    userday_neighbor[period_id][dayid][1] = prev_cnt + cur_cnt
                elif prev_min_state<cur_min_state:
                    userday_neighbor[period_id][dayid] = [prev_min_state, prev_cnt]
                prev_numreds = userday_numreds[period_id-1][dayid]
                userday_numreds[period_id][dayid] += prev_numreds

    # calculate the hidden_state for all userdays.
    userday_latent = copy.deepcopy(userday_numreds)
    for period_id in range(num_periods):
        for dayid, day_numreds in enumerate(userday_numreds[period_id]):
            if day_numreds == 0:
                userday_latent[period_id][dayid] = 0
            else:
                hidden_state = 0
                for l, h in config.bn.latent.count:
                    if l <= day_numreds and day_numreds <= h:
                        break
                    hidden_state += 1
                assert hidden_state < len(config.bn.latent.count)
                userday_latent[period_id][dayid] = hidden_state
    # construct feature [user, day, red, target, s1, c1, s2, c2, l1, l2]
    ret = []
    dayid = 0
    max_min_state = len(neighbor_ticks)-1
    for u, d, s, r in user_data:
        sc = []
        latent = []
        for k in range(num_periods):
            min_state, min_state_count = userday_neighbor[k][dayid]
            if min_state == np.inf:
                # Missing activities for user[u] in day[d].
                min_state = max_min_state
                min_state_count = 1
            # map min_state_count to its state id.
            count_sid = 0
            for start_c, end_c in config.bn.observed_neighbor.count:
                if start_c <= min_state_count and min_state_count <= end_c:
                    break
                count_sid += 1
            latent.append(userday_latent[k][dayid])
            sc.append(min_state)
            sc.append(count_sid)
        feat = [u, d, r, target_states[dayid]] + sc + latent
        assert np.isnan(np.sum(feat)) == False
        ret.append(feat)
        dayid += 1
    return ret


def group_by_user(rst):
    ret = {}
    for u, d, s, r in rst:
        if u not in ret:
            ret[u] = []
        ret[u].append((u, d, s, r))
    return ret.values()

def get_svm_rst_name():
    rs = config.state.random_seed
    nu = 0.25
    kernel = 'sigmoid'
    gamma = 0.1
    shrink = False
    result_dir = os.path.join(config.io.outdir, 'svm')
    name = 'svm__rs_{}__nu_{:.2f}__kernel_{}__gamma_{:.1f}__shrink_{}.txt'.format(
        rs, nu, kernel, gamma, shrink)
    fname1 = os.path.join(result_dir, 'train', name)
    fname2 = os.path.join(result_dir, 'test', name)
    return fname1, fname2

def get_random_rst_name():
    result_dir = os.path.join(config.io.outdir, 'random')
    rs = config.state.random_seed
    name = 'random__rs_{}.txt'.format(rs)
    fname1 = os.path.join(result_dir, 'train', name)
    fname2 = os.path.join(result_dir, 'test', name)
    return fname1, fname2

def get_iso_forest_rst_name():
    result_dir = os.path.join(config.io.outdir, 'iso_forest')
    rs = config.state.random_seed
    n_estimators = 150
    max_samples = 'auto'
    contamination = 0.1
    max_features = 1.0  # default is 1.0 (use all features)
    bootstrap = False
    name = 'iso_forest__rs_{}__n_{}__maxsamples_{}__contamination_{}__maxfeatures_{}__bootstrap_{}.txt'.format(
        rs, n_estimators, max_samples, contamination, max_features, bootstrap)
    fname1 = os.path.join(result_dir, 'train', name)
    fname2 = os.path.join(result_dir, 'test', name)
    return fname1, fname2


def get_pca_rst_name():
    result_dir = os.path.join(config.io.outdir, 'pca')
    rs = config.state.random_seed
    n_components = 10
    name = 'pca__rs_{}__n_{}.txt'.format(rs, n_components)
    fname1 = os.path.join(result_dir, 'train', name)
    fname2 = os.path.join(result_dir, 'test', name)
    return fname1, fname2

def get_dnn_rst_name():
    nl = 1
    hs = 50
    result_dir = os.path.join(config.io.outdir, 'dnn')
    fname1 = os.path.join(result_dir, 'train', 'dnn__nl_{}__hs_{}.txt'.format(nl,hs))
    fname2 = os.path.join(result_dir, 'test', 'dnn__nl_{}__hs_{}.txt'.format(nl,hs))
    return fname1, fname2

if __name__ == "__main__":
    # setup spark.
    conf = (SparkConf()
            .setMaster(config.SPARK_MASTER)
            .set("spark.app.name", __file__)
            .set("spark.executor.memory", "50g")
            .set("spark.ui.showConsoleProgress", "true"))
    sc = SparkContext(conf=conf)
    # define rst files.
    all_rst_files = [get_svm_rst_name(), get_pca_rst_name(), get_random_rst_name(),
                           get_iso_forest_rst_name(), get_dnn_rst_name()]
    all_rst_files = [get_dnn_rst_name()]
    for rst_train_file, rst_test_file in all_rst_files:
        print('-------------------------------------------')
        # load rst
        with open(rst_train_file, "r") as fp:
            lines = fp.read().strip().split('\n')
            train_rst = []
            all_train_rst = []
            for l in lines:
                user, day, score, red = l.split('\t')
                user = int(user)
                day = int(day)
                score = float(score)
                red = int(red)
                start_day, end_day = config.bn.rect.train_days
                all_train_rst.append([user, day, score, red])
                if day >= start_day and day <= end_day:
                    train_rst.append([user, day, score, red])
        with open(rst_test_file, "r") as fp:
            lines = fp.read().strip().split('\n')
            test_rst = []
            for l in lines:
                user, day, score, red = l.split('\t')
                user = int(user)
                day = int(day)
                score = float(score)
                red = int(red)
                test_rst.append([user, day, score, red])
        print("Load %s with %d lines." % (rst_train_file, len(all_train_rst)))
        print("Load %s with %d lines." % (rst_test_file, len(test_rst)))
        # rectify
        xdata, ydata, stds = extract_xy_from_rst(all_train_rst)
        params = train_rectification(train_rst)
        apply_rectification(all_train_rst, params)
        apply_rectification(test_rst, params)
        xdata2, ydata2, stds2 = extract_xy_from_rst(all_train_rst)
        plot_1std_scores(xdata, ydata, stds, ydata2, stds2,
                         '../cache/rect.png', params)
        # group data by user and parallelize.
        train_rst_by_users = group_by_user(train_rst)
        test_rst_by_users = group_by_user(test_rst)
        rdd_train_rst_by_users = sc.parallelize(train_rst_by_users)
        rdd_test_rst_by_users = sc.parallelize(test_rst_by_users)
        # create feature.
        target_ticks = calculate_ticks(
            train_rst, config.bn.observed_target.ratio)
        neighbor_ticks = calculate_ticks(
            train_rst, config.bn.observed_neighbor.ratio)
        # [[user, day, red, target, s1, c1, s2, c2, l1, l2],]
        train_ret = rdd_train_rst_by_users.flatMap(partial(
            extract_user_feat, target_ticks, neighbor_ticks)).collect()
        test_ret = rdd_test_rst_by_users.flatMap(partial(
            extract_user_feat, target_ticks, neighbor_ticks)).collect()
        print('Extracted train features %d' % (len(train_ret)))
        # calculate cr scores using the features
        rst_target = []
        rst_min_state = []
        rst_min_state2 = []
        rst_latent = []
        npf = np.array(test_ret)
        for feat in test_ret:
            u, d, r, t, s, _, s2 = feat[:7]
            l = feat[-config.bn.observed_neighbor.num_periods]
            rst_target.append((d, -t, r))
            rst_min_state.append((d, -s, r))
            rst_min_state2.append((d, -s2, r))
            rst_latent.append((d, l, r))
        rst_target = sorted(rst_target, key=operator.itemgetter(0))
        rst_min_state = sorted(rst_min_state, key=operator.itemgetter(0))
        rst_min_state2 = sorted(rst_min_state2, key=operator.itemgetter(0))
        rst_latent = sorted(rst_latent, key=operator.itemgetter(0))
        for b in config.cr.budgets:
            cr_score_target = cumulative_recall(rst_target, b, config.cr.increment)
            cr_score_minstate = cumulative_recall(rst_min_state, b, config.cr.increment)
            cr_score_minstate2 = cumulative_recall(rst_min_state2, b, config.cr.increment)
            cr_score_latent = cumulative_recall(rst_latent, b, config.cr.increment)
            print("CR-%d (target): %.4f" % (b, cr_score_target))
            print("CR-%d (minstate): %.4f" % (b, cr_score_minstate))
            print("CR-%d (minstate2): %.4f" % (b, cr_score_minstate2))
            print("CR-%d (latent): %.4f" % (b, cr_score_latent))
        # save features
        out_train_file = rst_train_file.split('/')[-1]+'_bn_feat.pkl'
        out_train_file = os.path.join('../cache', out_train_file)
        with open(out_train_file, 'wb+') as fp:
            cPickle.dump(train_ret, fp, protocol=2)
        out_test_file = rst_test_file.split('/')[-1]+'_bn_feat.pkl'
        out_test_file = os.path.join('../cache', out_test_file)
        with open(out_test_file, 'wb+') as fp:
            cPickle.dump(test_ret, fp, protocol=2)
        print('Done. Saved to ../cache/*._bn_feat.pkl')



