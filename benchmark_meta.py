"""v0.4 pre-registered synthetic transfer experiment.

Meta-model sees unlabeled graph descriptors only; training graphs are disjoint
from both unseen in-distribution and shifted-family evaluation samples.
"""
from __future__ import annotations

import argparse
import json
from random import Random

from benchmark_memory import make_dataset
from gfc import Edge, certify
from ghost_memory import POLICIES, policy_readout
from meta_memory import (
    apply_meta_memory, evaluate_meta_memory, topology_features, train_meta_memory
)


def make_dense_graph(seed: int) -> tuple[int, list[Edge]]:
    rng = Random(seed)
    n = 20
    endpoints = [
        (u, v) for u in range(n) for v in range(u + 1, n)
        if rng.random() < 0.42
    ]
    permutation = list(range(n))
    rng.shuffle(permutation)
    endpoints = [(permutation[u], permutation[v]) for u, v in endpoints]
    rng.shuffle(endpoints)
    assignment = [rng.randrange(2) for _ in range(n)]
    return n, [
        Edge(u, v, assignment[u] ^ assignment[v] ^ (rng.random() < 0.12))
        for u, v in endpoints
    ]


def dense_dataset(count: int, seed: int) -> list[tuple[int, list[Edge]]]:
    rng = Random(seed)
    return [make_dense_graph(rng.randrange(1 << 63)) for _ in range(count)]


def make_split(modular_count: int, dense_count: int, seed: int):
    """Balanced mixture; type names are not available to the meta learner."""
    mix = make_dataset("modular", modular_count, seed + 1)
    mix += dense_dataset(dense_count, seed + 2)
    Random(seed + 3).shuffle(mix)
    return mix


def experiment(train_per_family: int = 60, test_per_family: int = 80):
    if train_per_family < 1 or test_per_family < 1:
        raise ValueError("Positive train and test sizes are required")
    train = make_split(train_per_family, train_per_family, 73000)
    memory = train_meta_memory(train, seed=9341, min_leaf=max(3, train_per_family // 6))
    matched_modular = make_dataset("modular", test_per_family, 74100)
    matched_dense = dense_dataset(test_per_family, 74200)
    shifted_grid = make_dataset("grid", test_per_family, 74300)
    shifted_hub = make_dataset("hub", test_per_family, 74400)
    sets = {
        "modular_unseen": matched_modular,
        "dense_unseen": matched_dense,
        "modular_dense_mixture_unseen": matched_modular + matched_dense,
        "grid_shift": shifted_grid,
        "hub_shift": shifted_hub,
    }
    results = {
        name: evaluate_meta_memory(memory, graphs, seed=9500 + j)
        for j, (name, graphs) in enumerate(sets.items())
    }
    # A privileged oracle upper bound for the *available policies only*.
    # This is not a legal deployable baseline; it examines all policy costs.
    oracle_cost = 0
    for idx, (n, edges) in enumerate(sets["modular_dense_mixture_unseen"]):
        oracle_cost += min(policy_readout(n, edges, p, 8800 + idx).cost
                           for p in POLICIES)
    for n, edges in (matched_modular + matched_dense + shifted_grid + shifted_hub)[:20]:
        solution = certify(n, edges)
        for p in POLICIES:
            observed = policy_readout(n, edges, p, 61)
            assert (not any(observed.signature)) == solution.satisfiable
        assert (not any(apply_meta_memory(n, edges, memory, 61).signature)) == solution.satisfiable
    report = {
        "experiment": "GFC-v0.4 synthetic meta-memory / 2026-10-10",
        "hypothesis": "A learned structural split improves on best global policy under a mixture, but may fail under shift.",
        "train_count": len(train),
        "selection": json.loads(memory.to_json()),
        "evaluation": results,
        "posthoc_policy_oracle_cost_on_mixture": oracle_cost,
        "limitations": (
            "Cycle-edge visits are a proxy only, and all families are synthetic. "
            "Posthoc oracle is not a fair runtime baseline. No end-to-end or hardware speedup claim."
        ),
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-family", type=int, default=60)
    parser.add_argument("--test-per-family", type=int, default=80)
    parser.add_argument("--memory-out", default=None)
    args = parser.parse_args()
    report = experiment(args.train_per_family, args.test_per_family)
    if args.memory_out:
        from pathlib import Path
        Path(args.memory_out).write_text(
            json.dumps(report["selection"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
