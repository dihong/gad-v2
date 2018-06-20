"""
Extract regular emails and PCs used by a user.
"""
from pyspark import SparkContext
from pyspark import SparkConf
from os import path as osp
from functools import partial
from config import config
from operator import itemgetter

def extract_user_logon(year, month, rdd_lines):
    ret = {} # {(user, pc):count}
    for line in rdd_lines:
        line_split = line.split(',')
        if line_split[0] == 'id':
            # skip header
            continue
        try:
            identifier, date, user, pc, act = line_split
        except:
            continue
        date = date.split(' ')[0]
        dd, mm, yy = date.split('/')
        mm = int(mm)
        yy = int(yy)
        if yy>= year and mm >= month:
            # skip data reserved for testing.
            continue
        key = "%s:%s" % (user, pc)
        if key not in ret:
            ret[key] = 1
        else:
            ret[key] += 1
    ret = ret.items()
    return ret


if __name__ == "__main__":
    conf = (SparkConf()
            .setMaster(config.SPARK_MASTER)
            .set("spark.app.name", __file__)
            .set("spark.executor.memory", "50g")
            .set("spark.driver.maxResultSize", "100g"))
    sc = SparkContext(conf=conf)

    # for each user, get their email.
    with open(osp.join(config.data_dir, "events/LDAP/2009-12.csv")) as fp:
        lines = fp.read().strip().split('\n')[1:]
        user_email = {}
        for line in lines:
            elems = line.split(',')
            user = elems[1]
            email = elems[2]
            user_email[user] = email

    # load logon.txt
    logon_file = osp.join(config.data_dir, "events", "logon.csv")
    assert osp.isfile(logon_file), logon_file
    with open(logon_file, "r") as fp:
        lines = fp.read().split('\n')
    print("Load %d lines from %s"%(len(lines), logon_file))

    # parallelize lines and extract (user, pc) pairs.
    rdd_lines = sc.parallelize(lines, len(lines)).coalesce(1000).glom()
    user_pc = rdd_lines.flatMap(
        partial(extract_user_logon,
                config.test_start_month[0],
                config.test_start_month[1])).reduceByKey(lambda x,y:x+y).collect()
    print ("Calculated %d pairs of (user, pc)." % len(user_pc))

    # for each user, calculate [(pc, count)] list.
    by_users = {}
    for k,v in user_pc:
        user, pc = k.split(':')
        if user not in by_users:
            by_users[user] = [(pc, v)]
        else:
            by_users[user].append((pc, v))
    res = []
    for user, counts in by_users.items():
        counts = sorted(counts, key=itemgetter(1), reverse=True)
        pc, c = counts[0]
        all_pcs = [pc]
        total_counts = sum([v for k,v in counts])
        res.append((user, pc, float(c)/total_counts))
    res = sorted(res, key=itemgetter(2))
    with open(config.user_pcs, "w+") as fp:
        fp.write("\n".join(["%s %s %s %.6f" % (user, user_email[user], pc, pt)
                            for user, pc, pt in res]))
    print("Done. Results saved to %s."%config.user_pcs)

    




