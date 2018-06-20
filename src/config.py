from easydict import EasyDict as edict
import numpy

config = edict()

config.SPARK_MASTER = 'spark://node5:7077'

# IO directories
config.io = edict()
config.io.indir = '../r6.2'
config.io.outdir = '../results'
config.io.score = '../cr-scores'
config.io.train_outdir = '../train_results' # anomaly score for train data
config.io.model = '../models' # where trained models saved

# hmm
config.hmm = edict()
config.hmm.nfeats = 3 # 1 for [loss], 3 for [loss, mov_mean, mean]

# state
config.state = edict()
config.state.random_seed = 858

# cumulative recall
config.cr = edict()
config.cr.budgets = [400, 1000]
config.cr.increment = 25

# model parameters.
config.iso_forest = edict()
config.iso_forest.n_estimators = numpy.linspace(10, 300, num=5).astype(numpy.int)
config.iso_forest.contamination = numpy.linspace(0, 1.0, num=5)
config.iso_forest.bootstrap = [False]

# naive bayes.
config.nb = edict()
config.nb.models = 'BBBPPBBBPP'
config.nb.prefix = 'nb'

# data
config.data = edict()
config.data.train_file = '../r6.2/count/train_by_days.txt'
config.data.test_file = '../r6.2/count/test_by_days.txt'
config.data.train_by_user_file = '../r6.2/count/train_by_users.pkl'
config.data.test_by_user_file = '../r6.2/count/test_by_users.pkl'
config.data.relational_feat = '../extra-features/feat.pkl'
# config.data.relational_feat = '../r6.2/count/data_by_users_all.pkl'
config.data.train_ratio = 0.85 # ratio of data used for training.

# checks
from os import path as osp
assert osp.isfile(config.data.train_file), "{} does not exist.".format(
    config.data.train_file)
assert osp.isfile(config.data.test_file), "{} does not exist.".format(
    config.data.test_file)
assert osp.isdir(config.io.indir), "{} does not exist.".format(config.io.indir)
assert osp.isdir(config.io.outdir), "{} does not exist.".format(config.io.outdir)
assert osp.isdir(config.io.score), "{} does not exist.".format(config.io.score)

