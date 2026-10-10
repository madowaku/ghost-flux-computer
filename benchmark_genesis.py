"""Preregistered-style GFC v0.5 study.

Training families: modular, dense, hub, grid. Held-out: independently seeded
replicates, plus an unseen random family as a distribution-shift probe.
No test graph is used to evolve the genome or tune its hyperparameters.
"""
from __future__ import annotations

import argparse
import json
from random import Random
from time import perf_counter

from benchmark_memory import make_dataset
from benchmark_meta import dense_dataset
from genesis import (
    INITIAL_WEIGHTS, bfs_readout, evaluate_genesis, genome_readout,
    prepare, train_genesis,
)
from gfc import certify
from ghost_memory import POLICIES, policy_readout
from meta_memory import train_meta_memory


def dataset(family: str, count: int, seed: int):
    if count <= 0:
        raise ValueError("Dataset sizes must be positive")
    return dense_dataset(count, seed) if family == "dense" else make_dataset(
        family, count, seed
    )


def experiment(train_per_family: int = 12, test_per_family: int = 32,
               generations: int = 12, population: int = 20) -> dict:
    families = ("modular", "dense", "hub", "grid")
    train = []
    for j, name in enumerate(families):
        train.extend(dataset(name, train_per_family, 81001 + j * 100))
    Random(94111).shuffle(train)

    # Both models see identical TRAINING graphs; no test-set optimization.
    t0 = perf_counter()
    genesis = train_genesis(
        train, seed=31337, generations=generations, population=population
    )
    t1 = perf_counter()
    meta = train_meta_memory(train, seed=31337, min_leaf=max(2, train_per_family // 4))
    meta_fit_seconds = perf_counter() - t1

    held_out = {
        name: dataset(name, test_per_family, 83001 + i * 100)
        for i, name in enumerate((*families, "random"))
    }
    evaluations = {
        name: evaluate_genesis(genesis, graphs, seed=92100 + i, meta_memory=meta)
        for i, (name, graphs) in enumerate(held_out.items())
    }
    # Evaluate mixture with one independent seed stream (not sum of previous trials).
    test_mix = sum((held_out[f] for f in families), [])
    evaluations["in_family_mixture"] = evaluate_genesis(
        genesis, test_mix, seed=92333, meta_memory=meta
    )

    # Exactness audit on randomly selected unseen graphs.
    rng = Random(817)
    checked = 0
    for name, graphs in held_out.items():
        for n, edges in graphs[:min(len(graphs), 12)]:
            truth = certify(n, edges).satisfiable
            prep = prepare(n, edges)
            trial_seed = rng.randrange(1 << 63)
            for obs in (
                genome_readout(prep, genesis.genome, trial_seed),
                bfs_readout(prep, trial_seed),
                policy_readout(n, edges, meta.choose(n, edges), trial_seed)
            ):
                assert (not any(obs.signature)) == truth
            checked += 1

    return {
        "experiment": "GFC v0.5 Ghost Genesis: fixed seeds and held-out families",
        "training_families": list(families),
        "training_graphs": len(train),
        "genesis": json.loads(genesis.to_json()),
        "is_unseen_weight_program": genesis.genome.weights not in INITIAL_WEIGHTS,
        "v0_4_meta_control": json.loads(meta.to_json()),
        "meta_training_seconds": meta_fit_seconds,
        "measured_genesis_training_seconds": perf_counter() - t0 - meta_fit_seconds,
        "evaluations": evaluations,
        "exact_certificates_checked": checked,
        "caveat": (
            "Bounded linear rule search, not free-form algorithm invention. "
            "Training overhead is significant; proxy reductions are not"
            " a demonstrated wall-clock, physical or complexity advantage."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-family", type=int, default=12)
    parser.add_argument("--test-per-family", type=int, default=32)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--memory-out")
    args = parser.parse_args()
    result = experiment(
        train_per_family=args.train_per_family,
        test_per_family=args.test_per_family,
        generations=args.generations,
        population=args.population,
    )
    if args.memory_out:
        from pathlib import Path
        Path(args.memory_out).write_text(
            json.dumps(result["genesis"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
