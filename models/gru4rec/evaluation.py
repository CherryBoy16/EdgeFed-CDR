"""
evaluation.py

Evaluation metrics for GRU4Rec.
"""

import torch


def hit_rate(predictions, target):

    return int(target in predictions)


def recall(predictions, target):

    return int(target in predictions)


def mrr(predictions, target):

    if target in predictions:

        rank = predictions.index(target) + 1

        return 1 / rank

    return 0


def ndcg(predictions, target):

    if target in predictions:

        rank = predictions.index(target) + 1

        import math

        return 1 / math.log2(rank + 1)

    return 0