import pytest
import torch
from benchmark import Oracle, run


def test_oracle_budget_and_duplicates():
    oracle = Oracle(1)
    point = torch.tensor([[.2, .3]])
    assert oracle.query(point).shape == (1, 3)
    with pytest.raises(ValueError, match="once"):
        oracle.query(point)
    with pytest.raises(ValueError, match="budget"):
        oracle.query(torch.tensor([[.1, .3]]))


def test_random_trace_accounts_for_queries():
    trace = run("random", 42, rounds=2, batch=3)
    assert [row["calls"] for row in trace] == [8, 11, 14]
    values = [row["hypervolume"] for row in trace]
    assert values == sorted(values)
