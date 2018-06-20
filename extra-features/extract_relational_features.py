"""
Extract all relational features as described in relational-feat.txt
"""
from pyspark import SparkContext
from pyspark import SparkConf
from os import path as osp
from functools import partial
from config import config
from operator import itemgetter
import datetime
import numpy as np
import cPickle

def load_cache(fname):
    cache_file = osp.join(config.cache_dir, fname)
    if osp.isfile(cache_file):
        with open(cache_file, "r") as fp:
            return cPickle.load(fp)
    else:
        return None

def save_cache(fname, obj):
    cache_file = osp.join(config.cache_dir, fname)
    with open(cache_file, "w+") as fp:
        cPickle.dump(obj, fp, protocol=2)

def convert_date(yy, mm, dd):
    return yy*415+mm*32+dd

def is_weekend(yy, mm, dd):
    return datetime.date(yy, mm, dd).isocalendar()[2] > 5

def extract_email(org_domain, rdd_lines):
    ret = {} # {(user, date_key): (recv, send)}
    for line in rdd_lines:
        line_split = line.split(',')
        if line_split[0] == 'id':
            # skip header
            continue
        identifier,date,user,pc,to,cc,bcc,_from,activity,size,attachments=\
            line_split
        date, time = date.split(' ')
        mm, dd, yy = date.split('/')
        dd = int(dd)
        mm = int(mm)
        yy = int(yy)
        if is_weekend(yy, mm, dd):
            # skip weekends.
            continue
        date_key = convert_date(yy, mm, dd)
        key = "%s:%s" % (user, date_key)
        if key not in ret:
            ret[key] = np.zeros(2)
        destinations = set()
        if len(to) and to != _from > 1:
            destinations.add(to)
        if len(cc) > 1 and cc != _from:
            destinations.add(cc)
        if len(bcc) > 1 and bcc != _from:
            destinations.add(bcc)
        if activity == "Send":
            for des in destinations:
                if des.endswith(org_domain) == False:
                    # user send email to outside of organization, and
                    # no other users from the organization can view it.
                    ret[key][1] += 1
                    break
        elif activity == "View":
            if _from.endswith(org_domain) == False:
                recv_has_org_emails = False
                for des in destinations:
                    if des.endswith(org_domain) == True:
                        recv_has_org_emails = True
                        break
                if recv_has_org_emails == False:
                    # user received an email from outsie of organization, and
                    # no other users from the organization can view it.
                    ret[key][0] += 1
        else:
            raise ValueError(activity)
    return ret.items()

def extract_http(jobhunting_sites, cloudstorage_sites, rdd_lines):
    ret = {} # {(user, date_key): (jobhunting, cloudstorage)}
    for line in rdd_lines:
        line_split = line.split(',')
        if line_split[0] == 'date':
            # skip header
            continue
        date, user, url, activity = line_split
        date, time = date.split(' ')
        mm, dd, yy = date.split('/')
        dd = int(dd)
        mm = int(mm)
        yy = int(yy)
        if is_weekend(yy, mm, dd):
            # skip weekends.
            continue
        date_key = convert_date(yy, mm, dd)
        key = "%s:%s" % (user, date_key)
        if key not in ret:
            ret[key] = np.zeros(2)
        if url.startswith('www'):
            url = url[4:]
        if url in jobhunting_sites and activity == "WWW Visit":
            ret[key][0] += 1
        if url in cloudstorage_sites and activity == "WWW Upload":
            ret[key][1] += 1
    return ret.items()

def extract_file(user_pe, dfiles, rdd_lines):
    ret = {} # {(user, date_key): (other-pc, from-removable-media,
    # to-removable-media, decoy)}
    for line in rdd_lines:
        line_split = line.split(',')
        if line_split[0] == 'id':
            # skip header
            continue
        identifier, date, user, pc, fname, activity, to_removable_media,\
                from_removable_media = line_split[:8]
        date, time = date.split(' ')
        mm, dd, yy = date.split('/')
        dd = int(dd)
        mm = int(mm)
        yy = int(yy)
        if is_weekend(yy, mm, dd):
            # skip weekends.
            continue
        date_key = convert_date(yy, mm, dd)
        key = "%s:%s" % (user, date_key)
        if key not in ret:
            ret[key] = np.zeros(4)
        if pc != user_pe[user][0]:
            # access files of other pcs.
            ret[key][0] += 1
        if from_removable_media == "True":
            # access files of removable media.
            ret[key][1] += 1
        if to_removable_media == "True":
            # access files of removable media.
            ret[key][2] += 1
        file_full_name = "%s@%s" % (fname, pc)
        if file_full_name in dfiles:
            # access decoy files.
            ret[key][3] += 1
    return ret.items()


def extract_logon(regular_hours, user_pe, rdd_lines):
    ret = {} # {(user, date_key): (after-hours-logon-count, other-pc-logon-count)}
    begin_hour, end_hour = regular_hours
    for line in rdd_lines:
        line_split = line.split(',')
        if line_split[0] == 'id':
            # skip header
            continue
        identifier, date, user, pc, act = line_split
        date, time = date.split(' ')
        mm, dd, yy = date.split('/')
        dd = int(dd)
        mm = int(mm)
        yy = int(yy)
        if is_weekend(yy, mm, dd):
            # skip weekends.
            continue
        date_key = convert_date(yy, mm, dd)
        key = "%s:%s" % (user, date_key)
        if key not in ret:
            ret[key] = np.zeros(2)
        logon_other_pc = 0
        work_after_hours = 0
        if pc != user_pe[user][0]:
            logon_other_pc = 1
        hour = int(time.split(':')[0])
        if hour<begin_hour or hour>end_hour:
            work_after_hours = 1
        ret[key] += [work_after_hours, logon_other_pc]
    return ret.items()


if __name__ == "__main__":
    conf = (SparkConf()
            .setMaster(config.SPARK_MASTER)
            .set("spark.app.name", __file__)
            .set("spark.executor.memory", "50g")
            .set("spark.driver.maxResultSize", "100g"))
    sc = SparkContext(conf=conf)

    # get the red flag of each day for each user.
    insiders = set()
    insider_files = ['r6.2-1.csv', 'r6.2-2.csv', 'r6.2-3.csv', 'r6.2-4.csv', 'r6.2-5.csv']
    for f in insider_files:
        fname = osp.join(config.data_dir, "answers", f)
        with open(fname, "r") as fp:
            for line in fp.read().strip().split('\n'):
                line = line.split(',')
                _datetime = line[2].strip('"')
                user = line[3].strip('"')
                date, time = _datetime.split(' ')
                mm, dd, yy = date.split('/')
                dd = int(dd)
                mm = int(mm)
                yy = int(yy)
                if is_weekend(yy, mm, dd):
                    # skip weekends.
                    continue
                if f == 'r6.2-3.csv' and user!='PLJ1771':
                    # special case: skip non-insider users mentioned in s3.
                    continue
                date_key = convert_date(yy, mm, dd)
                key = "%s:%s" % (user, date_key)
                insiders.add(key)

    # load decoy files.
    cache = load_cache("decoy_file.pkl")
    if cache is None:
        decoy_file = osp.join(config.data_dir, "events", "decoy_file.csv")
        dfiles = set()
        with open(decoy_file, "r") as fp:
            for line in fp.read().strip().split('\n')[1:]:
                line = line.strip()
                fname, pc = line.split(',')
                fname = fname.strip('"')
                pc = pc.strip('"')
                dfiles.add('%s@%s'%(fname,pc))
        save_cache("decoy_file.pkl", dfiles)
        print("Loaded %d decoy files from %s." % (len(dfiles)), decoy_file)
    else:
        dfiles = cache
        print("Loaded %d decoy files from %s." % (len(dfiles), "decoy_file.pkl"))

    # load user (pc, email) info.
    user_pe = {}
    with open(config.user_pcs, "r") as fp:
        for line in fp.read().strip().split('\n'):
            user, email, pc, freq = line.split(' ')
            user_pe[user] = (pc, email)
    print("Loaded %d user emails and pcs from %s." % (len(user_pe), config.user_pcs))

    # load logon.csv
    cache = load_cache("logon.pkl")
    if cache is None:
        logon_file = osp.join(config.data_dir, "events", "logon.csv")
        assert osp.isfile(logon_file), logon_file
        with open(logon_file, "r") as fp:
            logon_lines = [l.strip() for l in fp.read().split('\n') if len(l)>1]
        save_cache("logon.pkl", logon_lines)
        print("Loaded %d lines from %s"%(len(logon_lines), logon_file))
    else:
        logon_lines = cache
        print("Loaded %d lines from %s"%(len(logon_lines), "logon.pkl"))

    # load file.csv
    cache = load_cache("file.pkl")
    if cache is None:
        file_file = osp.join(config.data_dir, "events", "file.csv")
        assert osp.isfile(file_file), file_file
        with open(file_file, "r") as fp:
            file_lines = [l.strip() for l in fp.read().split('\n') if len(l) > 1]
        save_cache("file.pkl", file_lines)
        print("Loaded %d lines from %s"%(len(file_lines), file_file))
    else:
        file_lines = cache
        print("Loaded %d lines from %s"%(len(file_lines), "file.pkl"))

    # load http.csv
    cache = load_cache("http.pkl")
    if cache is None:
        http_file = osp.join(config.data_dir, "events", "http_without_content.csv")
        assert osp.isfile(http_file), http_file
        with open(http_file, "r") as fp:
            http_lines = []
            for l in fp.read().split('\n'):
                try:
                    identifier,date,user,pc,url,activity = l.split(',')
                except:
                    print("skipping line: '%s'" % l)
                    continue
                url = url.split('//')[-1].split('/')[0]
                new_line = "%s,%s,%s,%s" % (date, user, url, activity)
                http_lines.append(new_line)
        save_cache("http.pkl", http_lines)
        print("Loaded %d lines from %s"%(len(http_lines), http_file))
    else:
        http_lines = cache
        print("Loaded %d lines from %s"%(len(http_lines), "http.pkl"))

    # load email.csv
    cache = load_cache("email.pkl")
    if cache is None:
        email_file = osp.join(config.data_dir, "events", "email_without_content.csv")
        assert osp.isfile(email_file), email_file
        with open(email_file, "r") as fp:
            email_lines = [l.strip() for l in fp.read().split('\n') if len(l)>1]
        save_cache("email.pkl", email_lines)
        print("Loaded %d lines from %s"%(len(email_lines), email_file))
    else:
        email_lines = cache
        print("Loaded %d lines from %s"%(len(email_lines), "email.pkl"))

    # debugging
    # email_lines = email_lines[:100000]
    # http_lines = http_lines[:100000]


    # parallelize lines and extract logon.
    rdd_lines = sc.parallelize(logon_lines, len(logon_lines)).coalesce(1000).glom()
    # [(user, (date, (after-hours-logon-count, other-pc-logon-count)))]
    user_logon = rdd_lines.flatMap(
        partial(extract_logon,
                config.regular_hours,
                user_pe)).reduceByKey(lambda x,y:x+y)\
        .map(lambda x: (x[0].split(':')[0], (int(x[0].split(':')[1]), x[1])))\
        .groupByKey()\
        .mapValues(list)\
        .collect()
    tmp = []
    logon_other_pc_count = []
    logon_after_hour_count = []
    num_days = []
    for i in xrange(len(user_logon)):
        tmp.append((user_logon[i][0], sorted(user_logon[i][1])))
        for date_key, val in user_logon[i][1]:
            logon_after_hour_count.append(val[0])
            logon_other_pc_count.append(val[1])
        num_days.append(len(user_logon[i][1]))
    user_logon = tmp
    print ("Calculated %d users logon. Avg: cnt_afterhour(%.6f), \
cnt_otherpc(%.6f), num_days(%.2f)." 
           % (len(user_logon),
              np.mean(logon_after_hour_count),
              np.mean(logon_other_pc_count),
              np.mean(num_days)))

    # parallelize lines and extract file.
    rdd_lines = sc.parallelize(file_lines, len(file_lines)).coalesce(1000).glom()
    # [(user, (date, (other-pc, removable-media, decoy)))]
    user_file = rdd_lines.flatMap(
        partial(extract_file,
                user_pe,
                dfiles)).reduceByKey(lambda x,y:x+y)\
        .map(lambda x: (x[0].split(':')[0], (int(x[0].split(':')[1]), x[1])))\
        .groupByKey()\
        .mapValues(list)\
        .collect()
    tmp = []
    cnt_other_pc = []
    cnt_from_removable_media = []
    cnt_to_removable_media = []
    cnt_decoy = []
    num_days = []
    for i in xrange(len(user_file)):
        tmp.append((user_file[i][0], sorted(user_file[i][1])))
        for date_key, val in user_file[i][1]:
            cnt_other_pc.append(val[0])
            cnt_from_removable_media.append(val[1])
            cnt_to_removable_media.append(val[2])
            cnt_decoy.append(val[3])
        num_days.append(len(user_file[i][1]))
    user_file = tmp
    print ("Calculated %d users file. Avg: cnt_otherpc(%.6f), \
cnt_from_media(%.6f), cnt_to_media(%.6f), cnt_decoy(%.6f), num_days(%.2f)." 
           % (len(user_file),
              np.mean(cnt_other_pc),
              np.mean(cnt_from_removable_media),
              np.mean(cnt_to_removable_media),
              np.mean(cnt_decoy),
              np.mean(num_days)))

    # parallelize lines and extract email.
    rdd_lines = sc.parallelize(email_lines, len(email_lines)).coalesce(1000).glom()
    # [(user, (date, (other-pc, removable-media, decoy)))]
    user_email = rdd_lines.flatMap(
        partial(extract_email,
                config.org_domain)).reduceByKey(lambda x,y:x+y)\
        .map(lambda x: (x[0].split(':')[0], (int(x[0].split(':')[1]), x[1])))\
        .groupByKey()\
        .mapValues(list)\
        .collect()
    tmp = []
    cnt_send = []
    cnt_recv = []
    num_days = []
    for i in xrange(len(user_email)):
        tmp.append((user_email[i][0], sorted(user_email[i][1])))
        for date_key, val in user_email[i][1]:
            cnt_recv.append(val[0])
            cnt_send.append(val[1])
        num_days.append(len(user_email[i][1]))
    user_email = tmp
    print ("Calculated %d users email. Avg: cnt_send(%.6f), \
cnt_recv(%.6f), num_days(%.2f)." 
           % (len(user_email),
              np.mean(cnt_send),
              np.mean(cnt_recv),
              np.mean(num_days)))

    # parallelize lines and extract http.
    rdd_lines = sc.parallelize(http_lines, len(http_lines)).coalesce(1000).glom()
    # [(user, (date, (other-pc, removable-media, decoy)))]
    user_http = rdd_lines.flatMap(
        partial(extract_http,
                set(config.job_sites),
                set(config.cloudstorage_sites))).reduceByKey(lambda x,y:x+y)\
        .map(lambda x: (x[0].split(':')[0], (int(x[0].split(':')[1]), x[1])))\
        .groupByKey()\
        .mapValues(list)\
        .collect()
    tmp = []
    cnt_jobhunting = []
    cnt_cloudstorage = []
    num_days = []
    for i in xrange(len(user_http)):
        tmp.append((user_http[i][0], sorted(user_http[i][1])))
        for date_key, val in user_http[i][1]:
            cnt_jobhunting.append(val[0])
            cnt_cloudstorage.append(val[1])
        num_days.append(len(user_http[i][1]))
    user_http = tmp
    print ("Calculated %d users http. Avg: cnt_jobhunting(%.6f), \
cnt_cloudstorage(%.6f), num_days(%.2f)."
           % (len(user_http),
              np.mean(cnt_jobhunting),
              np.mean(cnt_cloudstorage),
              np.mean(num_days)))



    # merge features: logon(2), file(3), http(2), email(2).
    all_feats = [('logon', user_logon, [0,1]),
                 ('file', user_file, [2,3,4,5]),
                 ('http', user_http, [6,7]),
                 ('email', user_email, [8,9])]
    red_flag_index = 10
    all_day_keys = set()
    for name, data, index in all_feats:
        for user, feat in data:
            all_day_keys |= set([day_key for day_key, _ in feat])
    all_day_keys = list(all_day_keys)
    all_day_keys = sorted(all_day_keys) # sort day by ascending order.
    day_key_to_index = {all_day_keys[i]:i for i in xrange(len(all_day_keys))}
    merged_features = {}
    cnt_red_user_days = 0
    for name, data, index in all_feats:
        for user, feat in data:
            if user not in merged_features:
                merged_features[user] = -1*np.ones((len(all_day_keys),
                                                    1+red_flag_index),
                                                 dtype=np.int)
            for date_key, arr in feat:
                assert len(index) == len(arr)
                merged_features[user][day_key_to_index[date_key]][index] = arr
                insider_key = "%s:%s"%(user, date_key)
                if insider_key in insiders:
                    # this user-day has insider activity.
                    merged_features[user][day_key_to_index[date_key]]\
                        [red_flag_index] = 1
                    cnt_red_user_days += 1
    assert cnt_red_user_days > len(insiders), cnt_red_user_days
    print("Calculated features of %d days, with %d insider user-days." %
          (len(all_day_keys), len(insiders)))

    with open('feat.pkl', 'w+') as fp:
        cPickle.dump(merged_features, fp, protocol=2)
    # stop spark.
    sc.stop()



