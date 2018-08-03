
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
            cumulative_recall_score += current_red / total_red
    return cumulative_recall_score


def split_train_test(rst):
    from config import config
    import operator
    # rst: [(user, day, score, red)]
    rst_sorted = sorted(rst, key=operator.itemgetter(1), reverse=False)
    ntrain = int(len(rst)*config.data.train_ratio)
    last_train_day = rst_sorted[ntrain][1]
    train_rst = [r for r in rst if r[1]<=last_train_day]
    test_rst = [r for r in rst if r[1]>last_train_day]
    return train_rst, test_rst
