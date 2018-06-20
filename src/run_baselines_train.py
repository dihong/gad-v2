import os
import argparse
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.ensemble import IsolationForest
from config import config
import time
from multiprocessing import Pool

from batch import *


def run_svm(data_file, rs, nu, kernel, gamma, shrink, outfile):
    """Wrapper to run SVM model.

    Parameters
    ----------
    data_file : str
        filepath of data file
    rs : int
        random seed
    nu: float
        SVM parameter
    kernel: str
        SVM parameter
    gamma: float
        SVM parameter
    shrink: bool
        SVM parameter
    outfile : str
        filepath of output file to be generated
    """
    print 'running SVM with nu={}, kernel={}, shrink={}'.format(nu, kernel, shrink)
    day_batcher = DayBatcher(data_file, skiprow=1)
    model = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, shrinking=shrink, random_state=rs)
    mat = day_batcher.next_batch()
    with open(outfile, 'w') as result:
        result.write('day user red anomaly\n')
        while mat is not None:
            datadict = {'features': mat[:, 14:],
                        'red': mat[:, 13],
                        'user': mat[:, 1],
                        'day': mat[:, 0]}
            model.fit(datadict['features'])
            anomaly_scores = model.decision_function(datadict['features'])
            for day, user, red, score in zip(datadict['day'],
                                             datadict['user'],
                                             datadict['red'],
                                             anomaly_scores):
                    result.write('{} {} {} {}\n'.format(day, user, red, score[0]))
            mat = day_batcher.next_batch()


def run_pca(data_file, rs, n_components, outfile):
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
    print 'running PCA with n_components={}'.format(n_components)
    day_batcher = DayBatcher(data_file, skiprow=1)
    mat = day_batcher.next_batch()
    with open(outfile, 'w') as result:
        result.write('day user red loss\n')
        while mat is not None:
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
                result.write('%s %s %s %s\n' % (d, u, t, l))
            mat = day_batcher.next_batch()


def run_iso_forest(data_file, rs, n_estimators, max_samples, contamination, max_features, bootstrap, outfile):
    """Wrapper to run Isolation Forest model.

    Parameters
    ----------
    data_file : str
        filepath of data file
    rs : int
        random seed
    n_estimators: int
        Isolation Forest parameter
    max_samples: int
        Isolation Forest parameter
    contamination: float
        Isolation Forest parameter
    max_features: int
        Isolation Forest parameter
    bootstrap: bool
        Isolation Forest parameter
    outfile : str
        filepath of output file to be generated
    """
    print 'running Isolation Forest with n_estimators={}, max_samples={}, contamination={}, max_features={}, bootstrap={}'.format(n_estimators, max_samples, contamination, max_features, bootstrap)
    day_batcher = DayBatcher(data_file, skiprow=1)
    mat = day_batcher.next_batch()
    with open(outfile, 'w') as result:
        result.write('day user red loss\n')
        model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                contamination=contamination, max_features=max_features,
                                bootstrap=bootstrap, n_jobs=-1, verbose=0)
        while mat is not None:
            datadict = {'features': mat[:, 14:], 'red': mat[:, 13], 'user': mat[:, 1], 'day': mat[:, 0]}
            model.fit(datadict['features'])
            anomaly_scores = model.decision_function(datadict['features'])
            for day, user, red, score in zip(datadict['day'], datadict['user'], datadict['red'], anomaly_scores):
                if math.isnan(score) and not math.isinf(score):
                    print('exiting due divergence')
                    exit(1)
                else:
                    result.write(str(day) + ' ' + str(user) + ' ' + str(red) + ' ' + str(-1 * score) + '\n')
            mat = day_batcher.next_batch()


def run_random(data_file, rs, outfile):
    """Wrapper to run random model.

    Parameters
    ----------
    data_file : str
        filepath of data file
    rs : int
        random seed
    outfile : str
        filepath of output file to be generated
    """
    print 'running random model...'
    day_batcher = DayBatcher(data_file, skiprow=1)
    mat = day_batcher.next_batch()
    with open(outfile, 'w') as result:
        result.write('day user red anomaly\n')
        random.seed(rs)
        while mat is not None:
            datadict = {'features': mat[:, 14:], 'red': mat[:, 13], 'user': mat[:, 1], 'day': mat[:, 0]}
            anomaly_scores = [random.random() for x in datadict['features']]
            for day, user, red, score in zip(datadict['day'], datadict['user'], datadict['red'], anomaly_scores):
                result.write(str(day) + ' ' + str(user) + ' ' + str(red) + ' ' + str(-1 * score) + '\n')
            mat = day_batcher.next_batch()


def run_svm_train(data_file, outdir):
    """Run SVM model overa a range of parameter combinations.
    Fixed random seed.

       kernel : rbf, linear, poly, sigmoid
           nu : (0,1)
    shrinking : True, False

    Parameters
    ----------
    data_file : data file
       outdir : directory to save file with naming format outdir/svm__<rs>__<nu>__<kernel>__<gamma>__<shrink>.txt
    """
    gamma = 0.1
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']
    nu_vals = [float(x) / 100 for x in range(1, 101, 50)]
    shrink_vals = [True, False]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for kernel in kernels:
        for nu in nu_vals:
            for shrink in shrink_vals:
                outfile_name = 'svm__rs_{}__nu_{:.2f}__kernel_{}__gamma_{:.1f}__shrink_{}.txt'.format(rs, nu, kernel, gamma, shrink)
                outfile_path = os.path.join(outdir, outfile_name)
                if os.path.isfile(outfile_path):
                    print 'already run. file={}'.format(outfile_path)
                    continue
                run_svm(data_file, rs, nu, kernel, gamma, shrink, outfile_path)
    pass


def run_svm_test(args):
    """Run one parameter combination for SVM model."""
    test_file, outdir = args
    start = time.time()
    nu = 0.25
    kernel = 'sigmoid'
    gamma = 0.1
    shrink = False
    outfile_name = 'svm__rs_{}__nu_{:.2f}__kernel_{}__gamma_{:.1f}__shrink_{}.txt'.format(rs, nu, kernel, gamma, shrink)
    outfile_path = os.path.join(outdir, outfile_name)
    run_svm(test_file, rs, nu, kernel, gamma, shrink, outfile_path)
    dt = time.time() - start
    print("Done. Elapsed time is %.2f seconds." % dt)


def run_pca_test(args):
    """Run one parameter combination for PCA model."""
    test_file, outdir = args
    start = time.time()
    n_components = 10
    outfile_name = 'pca__rs_{}__n_{}.txt'.format(rs, n_components)
    outfile_path = os.path.join(outdir, outfile_name)
    run_pca(test_file, rs, n_components, outfile_path)
    dt = time.time() - start
    print("Done. Elapsed time is %.2f seconds." % dt)


def run_iso_forest_test(args):
    """Run one parameter combination for Isolation Forest model."""
    test_file, outdir = args
    start = time.time()
    n_estimators = 150
    max_samples = 'auto'
    contamination = 0.1
    max_features = 1.0  # default is 1.0 (use all features)
    bootstrap = False
    outfile_name = 'iso_forest__rs_{}__n_{}__maxsamples_{}__contamination_{}__maxfeatures_{}__bootstrap_{}.txt'.format(
        rs, n_estimators, max_samples, contamination, max_features, bootstrap)
    outfile_path = os.path.join(outdir, outfile_name)
    run_iso_forest(test_file, rs, n_estimators, max_samples, contamination,
                   max_features, bootstrap, outfile_path)
    dt = time.time() - start
    print("Done. Elapsed time is %.2f seconds." % dt)


def run_random_test(args):
    """Run one parameter combination for random model."""
    test_file, outdir = args
    start = time.time()
    outfile_name = 'random__rs_{}.txt'.format(rs)
    outfile_path = os.path.join(outdir, outfile_name)
    run_random(test_file, rs, outfile_path)
    dt = time.time() - start
    print("Done. Elapsed time is %.2f seconds." % dt)


if __name__ == '__main__':
    # init
    rs = config.state.random_seed
    test_file = config.data.train_file
    outdir = config.io.train_outdir
    # run algorithms
    pool = Pool(processes=5)
    results = []
    results.append(pool.apply_async(run_iso_forest_test,
                         [(test_file, os.path.join(outdir, 'iso_forest'))]))
    results.append(pool.apply_async(run_svm_test,
                                  [(test_file, os.path.join(outdir, 'svm'))]))
    results.append(pool.apply_async(run_pca_test,
                                  [(test_file, os.path.join(outdir, 'pca'))]))
    results.append(pool.apply_async(run_random_test,
                                  [(test_file, os.path.join(outdir, 'random'))]))
    for r in results:
        r.get()
