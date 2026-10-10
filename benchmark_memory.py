"""Reproducible Ghost Memory experiment with matched and shifted distributions."""
from __future__ import annotations

import argparse
import json
from random import Random

from gfc import Edge, certify
from ghost_memory import (
    POLICIES, apply_memory, evaluate_memory, policy_readout, train_memory
)


def make_graph(family: str, seed: int) -> tuple[int, list[Edge]]:
    rng = Random(seed)
    n = 20
    topology = set()

    def connect(u: int, v: int) -> None:
        if u != v:
            topology.add(tuple(sorted((u, v))))

    if family == "hub":
        for v in range(1, n):
            connect(0, v)
        for _ in range(28):
            connect(rng.randrange(1, n), rng.randrange(1, n))
    elif family == "grid":
        for r in range(4):
            for c in range(5):
                u = 5 * r + c
                if r < 3:
                    connect(u, u + 5)
                if c < 4:
                    connect(u, u + 1)
                if r < 3 and c < 4 and rng.random() < 0.6:
                    connect(u, u + 6)
    elif family == "random":
        for u in range(1, n):
            connect(u - 1, u)
        for _ in range(35):
            connect(rng.randrange(n), rng.randrange(n))
    elif family == "modular":
        for start in (0, 5, 10, 15):
            for u in range(start, start + 5):
                for v in range(u + 1, start + 5):
                    if rng.random() < 0.72:
                        connect(u, v)
        for u in (4, 9, 14):
            connect(u, u + 1)
    else:
        raise ValueError(f"Unknown family: {family}")

    # Re-label vertices and shuffle the edge input so there is no persistent ID hint.
    permutation = list(range(n))
    rng.shuffle(permutation)
    topology = [tuple(sorted((permutation[u], permutation[v]))) for u, v in topology]
    rng.shuffle(topology)
    hidden = [rng.randrange(2) for _ in range(n)]
    edges = [
        Edge(u, v, hidden[u] ^ hidden[v] ^ int(rng.random() < 0.12))
        for u, v in topology
    ]
    return n, edges


def make_dataset(family: str, count: int, seed: int) -> list[tuple[int, list[Edge]]]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = Random(seed)
    return [make_graph(family, rng.randrange(1 << 63)) for _ in range(count)]


def run_benchmark(train_size: int = 48, test_size: int = 64) -> dict:
    train = make_dataset("hub", train_size, seed=1001)
    ghost = train_memory(train, seed=3901)

    # None of the test graphs, seeds, or labels are used during training.
    matched = make_dataset("hub", test_size, seed=1002)
    shifted = make_dataset("grid", test_size, seed=1003)
    comparison = {
        "matched_hub": evaluate_memory(ghost, matched, seed=3902),
        "shifted_grid": evaluate_memory(ghost, shifted, seed=3903),
    }
    controls = {}
    for name, samples, seed in [
        ("matched_hub", matched, 3902), ("shifted_grid", shifted, 3903)
    ]:
        rng = Random(seed)
        fixed_hub = 0
        for n, edges in samples:
            fixed_hub += policy_readout(n, edges, "hub", rng.randrange(1 << 63)).cost
        controls[name] = fixed_hub
    # An invariant sanity check: no rearrangement may change whether an answer exists.
    for n, edges in matched[:4] + shifted[:4]:
        solution_exists = certify(n, edges).satisfiable
        for policy in POLICIES:
            result = policy_readout(n, edges, policy, 0)
            assert (not any(result.signature)) == solution_exists
    return {
        "description": "Fixed policy class ERM, topology-only training, disjoint seeds",
        "memory": json.loads(ghost.to_json()),
        "training_instances": train_size,
        "policy": ghost.selected_policy,
        "train_normalized_losses": ghost.normalized_losses,
        "test": comparison,
        "fixed_hub_heuristic_edge_visits": controls,
        "caveat": "This tests a readout-length proxy; it is not hardware or runtime speedup.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--memory-out", default=None, help="Path to save learned JSON memory")
    args = parser.parse_args()
    report = run_benchmark(args.train, args.test)
    if args.memory_out:
        from pathlib import Path
        Path(args.memory_out).write_text(
            json.dumps(report["memory"], indent=2, sort_keys=True) + "\\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
