from config import config
import os


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class FileName():

    @staticmethod
    def get_svm_rst_name():
        rs = config.state.random_seed
        nu = config.svm.nu
        kernel = config.svm.kernel
        gamma = config.svm.gamma
        shrink = False
        result_dir = os.path.join(config.io.outdir, 'svm')
        name = 'svm__rs_{}__nu_{:.2f}__kernel_{}__gamma_{:.1f}__shrink_{}.txt'.format(
            rs, nu, kernel, gamma, shrink)
        fname1 = os.path.join(result_dir, 'train', name)
        fname2 = os.path.join(result_dir, 'test', name)
        return fname1, fname2

    @staticmethod
    def get_dnn_rst_name():
        nl = config.dnn.num_layers
        hs = config.dnn.hidden_size
        result_dir = os.path.join(config.io.outdir, 'dnn')
        fname1 = os.path.join(result_dir, 'train',
                              'dnn__nl_{}__hs_{}.txt'.format(nl, hs))
        fname2 = os.path.join(result_dir, 'test',
                              'dnn__nl_{}__hs_{}.txt'.format(nl, hs))
        return fname1, fname2

    @staticmethod
    def get_random_rst_name():
        result_dir = os.path.join(config.io.outdir, 'random')
        rs = config.state.random_seed
        name = 'random__rs_{}.txt'.format(rs)
        fname1 = os.path.join(result_dir, 'train', name)
        fname2 = os.path.join(result_dir, 'test', name)
        return fname1, fname2

    @staticmethod
    def get_iso_forest_rst_name():
        result_dir = os.path.join(config.io.outdir, 'iso_forest')
        rs = config.state.random_seed
        n_estimators = config.iso_forest.n_estimators
        max_samples = 'auto'
        contamination = config.iso_forest.contamination
        max_features = 1.0  # default is 1.0 (use all features)
        bootstrap = False
        name = 'iso_forest__rs_{}__n_{}__maxsamples_{}__contamination_{}__maxfeatures_{}__bootstrap_{}.txt'.format(
            rs, n_estimators, max_samples, contamination, max_features, bootstrap)
        fname1 = os.path.join(result_dir, 'train', name)
        fname2 = os.path.join(result_dir, 'test', name)
        return fname1, fname2

    @staticmethod
    def get_pca_rst_name():
        result_dir = os.path.join(config.io.outdir, 'pca')
        rs = config.state.random_seed
        n_components = config.pca.n_components
        name = 'pca__rs_{}__n_{}.txt'.format(rs, n_components)
        fname1 = os.path.join(result_dir, 'train', name)
        fname2 = os.path.join(result_dir, 'test', name)
        return fname1, fname2

    @staticmethod
    def get_bn_feat_name(rst_train_file, rst_test_file):
        train_file = rst_train_file.split('/')[-1] + '_bn_feat.pkl'
        train_file = os.path.join('../cache', train_file)
        test_file = rst_test_file.split('/')[-1] + '_bn_feat.pkl'
        test_file = os.path.join('../cache', test_file)
        return train_file, test_file


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
            cumulative_recall_score += current_red / (0.0001 + total_red)
    return cumulative_recall_score


def split_train_test(rst, split_key=1):
    from config import config
    import operator
    # rst: [(user, day, score, red)]
    rst_sorted = sorted(rst, key=operator.itemgetter(split_key), reverse=False)
    ntrain = int(len(rst) * config.data.train_ratio)
    last_train_day = rst_sorted[ntrain][split_key]
    train_rst = [r for r in rst if r[split_key] <= last_train_day]
    test_rst = [r for r in rst if r[split_key] > last_train_day]
    return train_rst, test_rst
