# convert streaming test data into a list of pickled data by users, ordered by days.

import cPickle
from config import config
from batch import DayBatcher
import numpy as np
import operator
import os


def partition_by_user(infile_by_days, outfile_by_users):
    day_batcher = DayBatcher(infile_by_days, skiprow=1)
    mat = day_batcher.next_batch()
    data = {}
    while mat is not None:
        for row in mat:
            day = row[0]
            user = row[1]
            red = row[13]
            features = row[14:]
            if user not in data:
                data[user] = np.zeros(shape=(0,2+len(features)),dtype=float)
            f = np.zeros(shape=(1,2+len(features)),dtype=float)
            f[0,0] = day
            f[0,1] = red
            f[0,2:] = features
            data[user] = np.append(data[user], f, axis=0)
            if data[user].shape[0]>1:
                assert data[user][-1,0]>=data[user][-2,0]
        mat = day_batcher.next_batch()
    with open(outfile_by_users, "w+") as fp:
        cPickle.dump(data, fp, protocol=2)

def partition_by_user_into_matrices(infile_by_days):
    with open(infile_by_days, 'r') as fp:
        lines = fp.read().strip().split('\n')[1:]
        all_days = set()
        for row in lines:
            row = row.split(' ')
            day = int(float(row[0]))
            all_days.add(day)
        all_days = sorted(list(all_days))
        day_to_index = {d:i for i,d in enumerate(all_days)}
        ret = {}
        for row in lines:
            row = row.split(' ')
            day = int(float(row[0]))
            user = int(float(row[1]))
            red = float(row[13])
            features = [float(val) for val in row[14:]] + [red]
            if user not in ret:
                ret[user] = np.zeros((len(day_to_index), len(features)))
            ret[user][day_to_index[day]] = features
        return ret

def create_compact_features():
    with open(config.data.relational_feat, 'r') as fp:
        data_by_user = cPickle.load(fp)
    print("Load data from %s with %d users." % (config.data.relational_feat,
                                                len(data_by_user)))
    out = ["day user redteam logon_other_pc logon_after_hours access_file_other_pc to_removable from_removable decoy_file http_jobhunting http_cloudstorage recv_nonorg send_nonorg"]
    tmp = []
    userid = 1
    userdict = {}
    for user, val in data_by_user.iteritems():
        if user not in userdict:
            userdict[user] = userid
            userid += 1
        user_key = userdict[user]
        for day, feat in enumerate(val):
            tmp.append([user_key, day] + list(feat))
    # sort by day
    tmp = sorted(tmp, key=operator.itemgetter(1), reverse=False)
    for val in tmp:
        user = val[0]
        day = val[1]+1
        red = val[-1]
        feat = val[2:-1]
        meta = '%d %d %d ' % (day, user, max(0,red))
        out.append(meta+' '.join(['%d'%max(0,v) for v in feat]))
    out = '\n'.join(out)
    with open(config.data.compact_txt, 'w+') as fp:
        fp.write(out)
    os.system('rm %s.npy' % config.data.compact_txt[:-4])
    print("Save features to %s." % config.data.compact_txt)


if __name__=="__main__":

    # train
    """
    infile_by_days = config.data.train_file
    l = list(infile_by_days.split('/')[:-1])
    l.append('train_by_users.pkl')
    outfile_by_users = '/'.join(l)
    partition_by_user(infile_by_days, outfile_by_users)
    # test
    infile_by_days = config.data.test_file
    l = list(infile_by_days.split('/')[:-1])
    l.append('test_by_users.pkl')
    outfile_by_users = '/'.join(l)
    partition_by_user(infile_by_days, outfile_by_users)
    """
    # all_fixed.txt
    """
    data = partition_by_user_into_matrices('../r6.2/count/all_fixed.txt')
    with open('../r6.2/count/data_by_users_all.pkl', 'w+') as fp:
        cPickle.dump(data, fp, protocol=2)
    """
    create_compact_features()
