import os
import re
import subprocess
from config import config

def run_dnn(data_file, lr, nl, hs, outdir, num_features=None):
    """Run DNN model on data file given parameters."""
    feature_fp = (feature_json_file_fmt.format(num_features)
                  if num_features
                  else '/media/mkbc/gad/safekit-master/safekit/features/specs/agg/cert_all_in_all_out_agg.json')

    args = ['python', dnn_script,
            '-learnrate', str(lr),
            '-numlayers', str(nl),
            '-hiddensize', str(hs),
            '-random_seed', str(rs),
            data_file,
            outdir,
            feature_fp,
            '-skipheader']
    subprocess.call(args)

def run_dnn_test(data_file, num_features, train_budget, lr, nl, hs, outdir):
    """Wrapper for run_dnn with cumulative recall calculation.

    Writes summary files of CR scores in `outdir`.

    Parameters
    ----------
    data file : str
        input data filepath
    num_features
    train_budget
    lr : float
        DNN model parameter
    nl : int
        DNN model parameter
    hs : int
        DNN model parameter
    outdir: str
        output directory path
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fp = find_filepath(rs, nl, hs, outdir)
    if fp:
        print 'already run.'
    else:
        run_dnn(data_file, lr, nl, hs, outdir, num_features=num_features)
        fp = find_filepath(rs, nl, hs, outdir)
    budget = train_budget / 5
    score = float(cumulative_recall(fp, budget, cr_params['test']['increment']))
    with open(outdir + 'summary{}.txt'.format(budget), 'w') as f:
        s = 'budget={:<4}: CR={:.2f}, NL={}, HS={}\n'.format(budget, score, nl, hs)
        f.write(s)

if __name__ == '__main__':
    lr = 0.01
    rs = config.state.random_seed
    num_layers = 1
    hidden_size = 170
    num_features = None
    in_train = '/media/mkbc/gad/etc/train.txt'
    in_test = '/media/mkbc/gad/etc/test.txt'
    outdir_train = '/media/mkbc/gad/results/dnn/1/orig_train/'
    outdir_test = '/media/mkbc/gad/results/dnn/1/orig_test/'
    result = run_dnn_train(in_train, num_features, outdir_train)
    for budget in result:
        run_dnn_test(in_test, num_features, budget, lr, result[budget]['nl'],
                     result[budget]['hs'], outdir_test)
