"""Compare search policies over a finite, non-biological design space."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood


class Oracle:
    def __init__(self, budget):
        self.budget = budget
        self.seen = set()

    def query(self, candidates):
        keys = [tuple(row.tolist()) for row in candidates]
        if len(set(keys)) != len(keys) or self.seen.intersection(keys):
            raise ValueError("Each candidate may be queried once")
        if len(keys) + len(self.seen) > self.budget:
            raise ValueError("Query budget exceeded")
        self.seen.update(keys)
        x, y = candidates.T
        return torch.stack((
            1 - (x - .18)**2 - (y - .18)**2,
            1 - (x - .83)**2 - (y - .25)**2,
            1 - (x - .52)**2 - (y - .83)**2 + .04 * torch.sin(15 * x),
        ), dim=1)


def propose(policy, x, y, options, batch, rng):
    if policy == "random":
        return rng.choice(len(options), batch, replace=False), "uniform unseen candidates"
    model = SingleTaskGP(x, y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    if policy == "greedy":
        with torch.no_grad():
            values = model.posterior(options).mean[:, 0]
        return values.topk(batch).indices.tolist(), "highest predicted first objective"
    acquisition = qLogNoisyExpectedHypervolumeImprovement(
        model, ref_point=[-1.] * 3, X_baseline=x,
        sampler=SobolQMCNormalSampler(torch.Size([16]), seed=int(rng.integers(100000))),
        prune_baseline=True, cache_root=False,
    )
    picks = []
    for _ in range(batch):
        with torch.no_grad():
            values = acquisition(options.unsqueeze(1))
        values[picks] = -torch.inf
        picks.append(int(values.argmax()))
        acquisition.set_X_pending(options[picks])
    return picks, "sequential expected hypervolume improvement with pending points"


def run(policy, seed, rounds=8, batch=4):
    rng = np.random.default_rng(seed)
    grid = torch.linspace(0, 1, 11, dtype=torch.double)
    pool = torch.cartesian_prod(grid, grid)
    init = rng.choice(len(pool), 8, replace=False)
    oracle = Oracle(8 + rounds * batch)
    x, remaining = pool[init], np.setdiff1d(np.arange(len(pool)), init)
    y = oracle.query(x)
    measure = Hypervolume(torch.tensor([-1.] * 3, dtype=torch.double))
    trace = []

    def record(reason):
        front = y[is_non_dominated(y)]
        trace.append({"calls": len(oracle.seen), "hypervolume": measure.compute(front),
                      "decision": reason, "front": front.tolist()})

    record("shared seeded initial design")
    for _ in range(rounds):
        choices, reason = propose(policy, x, y, pool[remaining], batch, rng)
        selected = pool[remaining[choices]]
        x = torch.cat((x, selected))
        y = torch.cat((y, oracle.query(selected)))
        remaining = np.delete(remaining, choices)
        record(reason)
    return trace


def main():
    torch.set_num_threads(1)
    report = {policy: {str(seed): run(policy, seed) for seed in (17, 42, 79)}
              for policy in ("random", "greedy", "bayesian")}
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "trace.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    for policy, runs in report.items():
        values = np.array([[row["hypervolume"] for row in trace] for trace in runs.values()])
        calls = [row["calls"] for row in runs["42"]]
        axes[0].plot(calls, values.mean(0), label=policy)
        axes[0].fill_between(calls, values.min(0), values.max(0), alpha=.13)
    axes[0].set(xlabel="Oracle calls", ylabel="Hypervolume", title="Mean and seed range")
    axes[0].legend()
    for ax, index in zip(axes[1:], (0, -1)):
        front = np.array(report["bayesian"]["42"][index]["front"])
        chart = ax.scatter(front[:, 0], front[:, 1], c=front[:, 2], vmin=0, vmax=1, cmap="viridis")
        ax.set(xlabel="Objective 1", ylabel="Objective 2", xlim=(.35, 1.04),
               ylim=(.35, 1.04), title="Initial front" if index == 0 else "Final front")
    fig.colorbar(chart, ax=axes[1:], label="Objective 3")
    fig.savefig(out / "optimization.svg")
    print(json.dumps({policy: {seed: round(trace[-1]["hypervolume"], 6)
                              for seed, trace in runs.items()} for policy, runs in report.items()}))


if __name__ == "__main__":
    main()
