# Do evaluation
from config import config
from cr import cr
import glob
import os

result_dir = config.io.outdir

for method in glob.glob(os.path.join(result_dir, '*')):
    for rst_file in glob.glob(os.path.join(method, '*.txt')):
        out_lines = []
        for budget in config.cr.budgets:
            method_name = rst_file.split('/')[-1].split('__')[0]
            score = float(cr.cumulative_recall(rst_file, budget,
                                               config.cr.increment))
            out_lines.append("CR-%d: %.4f" % (budget, score))
        with open(os.path.join(config.io.score, method_name,
                               rst_file.split('/')[-1]), 'w+') as fp:
            fp.write('\n'.join(out_lines))




