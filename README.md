# AbOpt

A finite black-box optimization benchmark over two numerical design variables with three competing synthetic objectives. Each strategy starts with the same eight observations for a given seed and receives eight batches of four new observations. The hidden evaluator exposes only queried responses, enforces the query budget, and rejects duplicates.

Compare random search, a Gaussian-process greedy policy for objective one, and BoTorch qLogNEHVI. The Bayesian policy selects each batch sequentially, conditioning acquisition scores on pending candidates. The controller uses a fixed schedule and records the policy decision, query count, Pareto front, and hypervolume each round. Hypervolume uses a shared reference point of (-1, -1, -1).

Install with `python -m pip install -r requirements.txt`, then run `python benchmark.py`. The results directory contains traces and a figure showing hypervolume across seeds and the initial and final fronts. This finite numerical simulator isolates acquisition behavior under controlled query budgets.
