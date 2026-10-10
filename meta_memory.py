"""GFC v0.4: interpretable, topology-conditioned meta-memory.

Trains a one-split decision policy on graph structure only, compares it with
the best single policy in training, and keeps exact XOR certification separate.
This is a limited form of algorithm selection, not algorithm invention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from random import Random
from typing import Iterable

from gfc import Edge, Readout, _validate
from ghost_memory import POLICIES, policy_readout

FEATURES = ("density", "max_degree_fraction", "transitivity", "degree_heterogeneity")
META_SCHEMA = 1


def topology_features(n: int, edges: list[Edge]) -> dict[str, float]:
    """Permutation-invariant, relation-bit-blind graph features."""
    _validate(n, edges)
    neighbors: list[set[int]] = [set() for _ in range(n)]
    degree = [0] * n
    unique_pairs = set()
    for e in edges:
        degree[e.u] += 1
        degree[e.v] += 1
        if e.u != e.v:
            neighbors[e.u].add(e.v)
            neighbors[e.v].add(e.u)
            unique_pairs.add((min(e.u, e.v), max(e.u, e.v)))
    triples = sum(len(ns) * (len(ns) - 1) // 2 for ns in neighbors)
    triangle_tripled = sum(len(neighbors[u] & neighbors[v]) for u, v in unique_pairs)
    average_degree = sum(degree) / n if n else 0
    return {
        "density": 2 * len(unique_pairs) / (n * (n - 1)) if n > 1 else 0.0,
        "max_degree_fraction": max(degree, default=0) / max(1, n - 1),
        "transitivity": triangle_tripled / triples if triples else 0.0,
        "degree_heterogeneity": (
            sum(d * d for d in degree) / (n * average_degree**2)
            if n and average_degree else 0.0
        ),
    }


@dataclass(frozen=True)
class MetaMemory:
    global_policy: str
    split_feature: str | None
    threshold: float | None
    low_policy: str
    high_policy: str
    train_instances: int
    train_global_risk: float
    train_meta_risk: float
    training_probe_edge_visits: int
    train_random_probe_edge_visits: int
    seed: int
    schema_version: int = META_SCHEMA

    def choose(self, n: int, edges: list[Edge]) -> str:
        if self.split_feature is None:
            return self.global_policy
        value = topology_features(n, edges)[self.split_feature]
        return self.low_policy if value <= self.threshold else self.high_policy

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "MetaMemory":
        item = json.loads(raw)
        if item.get("schema_version") != META_SCHEMA:
            raise ValueError("Unsupported meta-memory format")
        for key in ("global_policy", "low_policy", "high_policy"):
            if item.get(key) not in POLICIES:
                raise ValueError(f"Invalid policy in {key}")
        feature = item.get("split_feature")
        threshold = item.get("threshold")
        if feature is None:
            if threshold is not None:
                raise ValueError("Threshold with no feature")
        elif feature not in FEATURES or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError("Invalid split")
        if item.get("train_instances", 0) < 1:
            raise ValueError("Empty training memory")
        return cls(**item)


def train_meta_memory(
    graphs: Iterable[tuple[int, list[Edge]]],
    seed: int = 20261010,
    min_leaf: int = 8,
    split_penalty: float = 0.0002,
) -> MetaMemory:
    """ERM over all axis-aligned decision stumps on four graph descriptors.

    The penalty applies to the mean normalized readout-length proxy.
    A split is chosen only if it beats the best global policy after penalty.
    Candidate threshold ties are resolved deterministically.
    """
    if min_leaf < 1 or split_penalty < 0:
        raise ValueError("Invalid training hyperparameters")
    rng = Random(seed)
    rows = []
    probe_visits = random_visits = 0
    for n, edges in graphs:
        features = topology_features(n, edges)
        trial_seed = rng.randrange(1 << 63)
        measures = {
            p: policy_readout(n, edges, p, trial_seed).cost
            for p in POLICIES
        }
        denominator = max(1, len(edges) * max(1, n))
        rows.append((features, {p: measures[p] / denominator for p in POLICIES}))
        probe_visits += sum(measures.values())
        random_visits += measures["random"]

    count = len(rows)
    if count == 0:
        raise ValueError("Training set is empty")
    risks = {
        p: sum(costs[p] for _, costs in rows) / count for p in POLICIES
    }
    global_policy = min(POLICIES, key=lambda p: (risks[p], p))
    best_risk = risks[global_policy]
    best_split = None
    # Enumerating training-derived thresholds is safe because held-out data
    # is never consulted. No parity labels are used to select readout layouts.
    for feature in FEATURES:
        unique = sorted({feat[feature] for feat, _ in rows})
        for i in range(len(unique) - 1):
            threshold = (unique[i] + unique[i + 1]) / 2
            left = [cost for feat, cost in rows if feat[feature] <= threshold]
            right = [cost for feat, cost in rows if feat[feature] > threshold]
            if min(len(left), len(right)) < min_leaf:
                continue
            low = min(POLICIES, key=lambda p: (sum(r[p] for r in left), p))
            high = min(POLICIES, key=lambda p: (sum(r[p] for r in right), p))
            if low == high:
                continue
            risk = (
                sum(r[low] for r in left) + sum(r[high] for r in right)
            ) / count
            if risk + split_penalty < best_risk - 1e-12:
                best_risk = risk
                best_split = (feature, threshold, low, high)
    if best_split is None:
        return MetaMemory(
            global_policy, None, None, global_policy, global_policy,
            count, risks[global_policy], risks[global_policy],
            probe_visits, random_visits, seed
        )
    feature, threshold, low, high = best_split
    return MetaMemory(
        global_policy, feature, threshold, low, high,
        count, risks[global_policy], best_risk,
        probe_visits, random_visits, seed
    )


def apply_meta_memory(
    n: int, edges: list[Edge], memory: MetaMemory, seed: int = 0
) -> Readout:
    return policy_readout(n, edges, memory.choose(n, edges), seed)


def evaluate_meta_memory(
    memory: MetaMemory, held_out: Iterable[tuple[int, list[Edge]]],
    seed: int = 20261011
) -> dict:
    """Compare paired out-of-sample readout proxies, without hidden tuning."""
    rng = Random(seed)
    sums = {k: 0 for k in ("random", "hub", "global", "meta")}
    improved = tied = worsened = count = 0
    choices = {p: 0 for p in POLICIES}
    for n, edges in held_out:
        trial_seed = rng.randrange(1 << 63)
        chosen = memory.choose(n, edges)
        choices[chosen] += 1
        results = {
            "random": policy_readout(n, edges, "random", trial_seed).cost,
            "hub": policy_readout(n, edges, "hub", trial_seed).cost,
            "global": policy_readout(n, edges, memory.global_policy, trial_seed).cost,
            "meta": policy_readout(n, edges, chosen, trial_seed).cost,
        }
        for key, value in results.items():
            sums[key] += value
        count += 1
        if results["meta"] < results["global"]:
            improved += 1
        elif results["meta"] == results["global"]:
            tied += 1
        else:
            worsened += 1
    if count == 0:
        raise ValueError("Need at least one held-out graph")
    delta = sums["global"] - sums["meta"]
    extra_training = memory.training_probe_edge_visits - memory.train_random_probe_edge_visits
    return {
        "held_out_graphs": count,
        "edge_visits": sums,
        "policy_counts": choices,
        "meta_vs_global_improved": improved,
        "meta_vs_global_tied": tied,
        "meta_vs_global_worsened": worsened,
        "meta_vs_global_savings_percent": (
            100 * delta / sums["global"] if sums["global"] else 0.0
        ),
        "meta_vs_hub_savings_percent": (
            100 * (sums["hub"] - sums["meta"]) / sums["hub"] if sums["hub"] else 0.0
        ),
        "training_all_policy_probe_edge_visits": memory.training_probe_edge_visits,
        "training_random_only_probe_edge_visits": memory.train_random_probe_edge_visits,
        "additional_training_probes_vs_random_only": extra_training,
        "additional_probe_visits_vs_global_erm": 0,
        "amortization_note": (
            "Global ERM and meta-ERM share the same training policy probes. "
            "Additional feature extraction and split-search costs are NOT measured. "
            "No valid total amortization break-even can yet be reported."
        ),
        "warning": "Only cycle-edge visits: not wall-clock speed, energy, or full computation cost.",
    }
