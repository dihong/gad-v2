# train the HMM model: transition T and emission E.
# T[i,j] is transition probability from state i to state j, where state 0 means
#   not insider and state 1 means insider.
# E[i] = N(u,s) is a Gaussian emission probability distribution for state i.


import numpy as np
from config import config
import cPickle
import os
import glob


def train_hmm(result_file):
    # result_file: located in config.io.outdir
    # It is the anormaly score calculated by machine learning models.
    T = np.zeros(shape=(2,2), dtype=float)
    E = [None, None]
    data = {}
    with open(result_file, 'r') as fp:
        lines = fp.read().strip().split('\n')[1:]
        state0 = [] # [[loss,mean,mov_mean]] of state 0 (red=0)
        state1 = []
        for l in lines:
            day, user, red, loss = l.split(' ')
            user = int(float(user))
            day = int(float(day))
            red = int(float(red))
            if red > 1:
                red = 1
            loss = float(loss)
            if user not in data:
                data[user] = {}
                data[user]['day'] = day
                data[user]['red'] = red
                data[user]['loss'] = loss
                data[user]['mean'] = loss
                data[user]['count'] = 1
                data[user]['mov_mean'] = loss
                continue
            else:
                assert day>data[user]['day'], "%d vs %d" % (
                    day, data[user]['day'])
                prev_red = data[user]['red']
                T[prev_red, red] += 1
                data[user]['count'] += 1
                c1 = 1.0/data[user]['count']
                c2 = 1.0-c1
                data[user]['mean'] = c1*loss + c2*data[user]['mean']
                data[user]['mov_mean'] = 0.3*loss + 0.7*data[user]['mov_mean']
                if config.hmm.nfeats == 1:
                    feat = [loss]
                else:
                    feat = [loss, data[user]['mov_mean'], data[user]['mean']]
                    assert config.hmm.nfeats == 3
                if red == 0:
                    state0.append(feat)
                else:
                    state1.append(feat)
                data[user]['day'] = day
                data[user]['red'] = red
                data[user]['loss'] = loss
    T[0,:] = T[0,:]/np.sum(T[0,:])
    T[1,:] = T[1,:]/np.sum(T[1,:])
    state0 = np.array(state0)
    state1 = np.array(state1)
    cov0 = np.cov(state0, rowvar=False)
    cov1 = np.cov(state1, rowvar=False)
    if state0.shape[1] == 1:
        # convert scalar to 1x1 matrix for compatibility.
        cov0 = cov0 * np.ones(shape=(1,1), dtype=np.float)
        cov1 = cov1 * np.ones(shape=(1,1), dtype=np.float)
    mean0 = np.mean(state0, axis=0)
    mean1 = np.mean(state1, axis=0)
    E[0] = {'mean': mean0, 'cov': cov0}
    E[1] = {'mean': mean1, 'cov': cov1}
    obj = {'T': T, 'E': E, 'S': np.array([0.99999, 0.00001])} # startprob []
    outmodel_file = os.path.join(config.io.model,
                                 result_file.split('/')[-1]+'.pkl')
    with open(outmodel_file, 'w+') as fp:
        cPickle.dump(obj, fp, protocol=2)

if __name__=="__main__":
    methods = glob.glob(os.path.join(config.io.train_outdir, '*'))
    for m in methods:
        rst_file = glob.glob(os.path.join(m, '*.txt'))
        for fname in rst_file:
            train_hmm(fname)



                



