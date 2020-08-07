import numpy as np


def pairwise_operator(codes, method):
    """
    Pairwsise operator based on a method and a list of predictions (e.g., lists of offsets)
    >>> assert pairwise_operator([[],[],[]], f1) == 1
    :param codes: a list of lists of predicted offsets
    :param method: a method to use to compare all pairs
    :return: the mean score between all possible pairs (excl. duplicates)
    """
    pairs = []
    for i,coderi in enumerate(codes):
        for j,coderj in enumerate(codes):
            if j>i:
                pairs.append(method(coderi, coderj))
    return np.mean(pairs)


def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1 if len(predictions)==0 else 0
    nom = 2*len(set(predictions).intersection(set(gold)))
    denom = len(set(predictions))+len(set(gold))
    return nom/denom

