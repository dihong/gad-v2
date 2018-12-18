from easydict import EasyDict as edict
import numpy

config = edict()

# Spark
config.spark = edict()
config.spark.SPARK_MASTER = 'spark://m65:7077'
config.spark.cores = 52

# IO directories
config.io = edict()
config.io.data_dir = '/data1/dihonggong/datasets/cert/r6.2'
config.io.outdir = '../results'
config.io.cache = '../cache'

# bayesian network (bn)
ratio_intervals = [(0, 0.005),
                   (0.005, 0.01),
                   (0.01, 0.02),
                   (0.02, 0.05),
                   (0.05, 0.1),
                   (0.1, 0.3),
                   (0.3, 0.6),
                   (0.6, 1.0)]
config.bn = edict()
config.bn.observed_target = edict()
config.bn.observed_target.ratio = ratio_intervals
config.bn.latent = edict()
config.bn.latent.count = [(0, 0), (1, numpy.inf)]  # how many ground-truth reds
config.bn.observed_neighbor = edict()
config.bn.observed_neighbor.ratio = ratio_intervals
config.bn.observed_neighbor.count = [(1, 1), (2, 2), (3,3), (4,4), (5, numpy.inf)]
config.bn.observed_neighbor.timespan = 7 # in days.
config.bn.observed_neighbor.num_periods = 1
config.bn.rect = edict()  # retification of bias data.
config.bn.rect.train_days = [10, numpy.inf]
config.bn.use_overlapping_periods = False

# state
config.state = edict()
config.state.random_seed = 858

# cumulative recall
config.cr = edict()
config.cr.budgets = [400, 1000]
config.cr.increment = 25

# model parameters.
config.iso_forest = edict()
config.pca = edict()
config.dnn = edict()
config.svm = edict()
config.iso_forest.n_estimators = 200
config.iso_forest.contamination = 0.1
config.iso_forest.bootstrap = [False]
config.pca.n_components = 3
config.svm.nu = 0.25
config.svm.kernel = 'rbf'
config.svm.gamma = 0.1
config.dnn.lr = 0.005
config.dnn.num_layers = 5
config.dnn.hidden_size = 10
config.dnn.activation = 'tanh'  # tanh or relu
config.dnn.dist = 'diag'  # ident, diag, full for covariance matrix of mvn
config.dnn.batch_size = 128
config.dnn.normalizer = 'none'  # none, layer, or batch
config.dnn.debug = False

# data
config.data = edict()
config.data.compact_txt = '../cache/feat_compact10d.txt' # compact.
config.data.compact_json = '../feat_compact10d.json'
config.data.train_ratio = 0.85  # ratio of data used for training.


# compact feature (cf)
config.cf = edict()
config.cf.regular_hours = [7, 20] # regular working hours in 24 hours format.
config.cf.org_domain = "dtaa.com" # domain used by an organization.
config.cf.user_pcs = "./user_pcs.txt"
config.cf.test_start_month = [2011,03]
config.cf.job_sites = [
    "linkedin.com",
    "glassdoor.com",
    "indeed.com",
    "careerbuilder.com",
    "simplyhired.com",
    "usajobs.gov",
    "linkup.com",
    "snagajob.com",
    "roberthalf.com",
    "dice.com",
    "idealist.com",
    "monster.com",
    "us.jobs",
    "collegerecruiter.com",
    "coolworks.com",
    "efinancialcareers.com",
    "energyjobline.com",
    "engineering.jobs",
    "ecojobs.com",
    "flexjobs.com",
    "healthcarejobsite.com",
    "internships.com",
    "mediabistro.com",
    "onewire.com",
    "jobs.prsa.org",
    "salesgravy.com",
    "salesjobs.com",
    "snagajob.com",
    "talentzoo.com",
    "youtern.com"]
config.cf.cloudstorage_sites = [
    "dropbox.com",
    "wikileaks.org",
    "drive.google.com",
    "onedrive.live.com",
    "box.com",
    "carbonite.com",
    "mega.nz",
    "icloud.com",
    "spideroak.com",
    "idrive.com",
    "pcloud.com"]
