"""Unit tests for training._prototype_retrieval_metrics -- the query-vs-
train-prototype retrieval metric that checkpoint selection in train_reid is
now based on, replacing plain classification accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from cow_reid.training import _prototype_retrieval_metrics


def test_prototype_retrieval_metrics_hand_computed_case():
    train_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    train_cow_ids = ["cow_a", "cow_b"]
    # query1 (cow_a) and query2 (cow_b) rank their own prototype first;
    # query3 is mislabelled cow_a but closer to cow_b's prototype.
    val_embeddings = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9]], dtype=np.float32)
    val_cow_ids = ["cow_a", "cow_b", "cow_a"]

    result = _prototype_retrieval_metrics(train_embeddings, train_cow_ids, val_embeddings, val_cow_ids)

    assert result["queries"] == 3
    assert result["top1"] == pytest.approx(2 / 3)
    # AP: query1=1.0 (rank1), query2=1.0 (rank1), query3=0.5 (rank2) -> mean 0.8333
    assert result["mAP"] == pytest.approx((1.0 + 1.0 + 0.5) / 3)


def test_prototype_retrieval_metrics_skips_val_cows_with_no_train_prototype():
    train_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    train_cow_ids = ["cow_a"]
    val_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    val_cow_ids = ["cow_a", "cow_never_in_train"]

    result = _prototype_retrieval_metrics(train_embeddings, train_cow_ids, val_embeddings, val_cow_ids)

    assert result["queries"] == 1  # only cow_a has a train-side prototype
    assert result["top1"] == 1.0


def test_prototype_retrieval_metrics_empty_inputs_return_none():
    empty = np.zeros((0, 2), dtype=np.float32)
    result = _prototype_retrieval_metrics(empty, [], empty, [])
    assert result == {"top1": None, "mAP": None, "queries": 0}

    non_empty = np.asarray([[1.0, 0.0]], dtype=np.float32)
    result = _prototype_retrieval_metrics(non_empty, ["cow_a"], empty, [])
    assert result["queries"] == 0
    assert result["top1"] is None
