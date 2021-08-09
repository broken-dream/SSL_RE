import sklearn.metrics as metric


def eval(ground_truth, prediction, labels):
    res = dict()
    res["acc"] = metric.accuracy_score(ground_truth, prediction)
    res["micro_p"] = metric.precision_score(ground_truth, prediction, labels=labels, average="micro")
    res["micro_r"] = metric.recall_score(ground_truth, prediction, labels=labels, average="micro")
    res["micro_f"] = metric.f1_score(ground_truth, prediction, labels=labels, average="micro")

    res["macro_p"] = metric.precision_score(ground_truth, prediction, labels=labels, average="macro")
    res["macro_r"] = metric.recall_score(ground_truth, prediction, labels=labels, average="macro")
    res["macro_f"] = metric.f1_score(ground_truth, prediction, labels=labels, average="macro")
    return  res