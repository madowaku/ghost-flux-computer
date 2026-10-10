"""GFC v0.3: a transferable, topology-only memory of cycle-readout policies.

The memory performs empirical risk minimization over a small fixed policy class.
It does not change constraint labels, solve general SAT, or prove a speedup.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from random import Random
from typing import Iterable

from gfc import Edge, Readout, _validate, signature

POLICIES = ("random", "hub", "anti_hub", "triangle", "hub_triangle", "low_triangle")
SCHEMA_VERSION = 1


def policy_readout(n: int, edges: list[Edge], policy: str, seed: int = 0) -> Readout:
    """Construct a fundamental-cycle readout using a topology-only policy.

    Edge relation bits (e.b) are used ONLY to measure the finished cycles' flux.
    Different policies are readout arrangements, not alternative truth semantics.
    """
    _validate(n, edges)
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    degree = [0] * n
    neighbors = [set() for _ in range(n)]
    for e in edges:
        degree[e.u] += 1
        degree[e.v] += 1
        if e.u != e.v:
            neighbors[e.u].add(e.v)
            neighbors[e.v].add(e.u)

    rng = Random(seed)
    order = list(range(len(edges)))
    rng.shuffle(order)  # seeded random tie-breaking, not edge-label dependent

    def priority(i: int) -> int:
        e = edges[i]
        hubs = degree[e.u] + degree[e.v]
        triangles = len(neighbors[e.u] & neighbors[e.v])
        return {
            "random": 0,
            "hub": hubs,
            "anti_hub": -hubs,
            "triangle": triangles,
            "hub_triangle": hubs + 3 * triangles,
            "low_triangle": -triangles,
        }[policy]

    # Python's stable sort preserves randomized order among equal priorities.
    if policy != "random":
        order.sort(key=priority, reverse=True)

    leader = list(range(n))

    def find(x: int) -> int:
        while leader[x] != x:
            leader[x] = leader[leader[x]]
            x = leader[x]
        return x

    tree = [[] for _ in range(n)]
    chords = []
    for eid in order:
        e = edges[eid]
        a, b = find(e.u), find(e.v)
        if a == b:
            chords.append(eid)
        else:
            leader[a] = b
            tree[e.u].append((e.v, eid))
            tree[e.v].append((e.u, eid))

    cycles = []
    for chord in chords:
        e = edges[chord]
        queue = deque([e.u])
        previous = {e.u: (-1, -1)}
        while queue and e.v not in previous:
            u = queue.popleft()
            for v, eid in tree[u]:
                if v not in previous:
                    previous[v] = (u, eid)
                    queue.append(v)
        cycle = [chord]
        v = e.v
        while v != e.u:
            u, eid = previous[v]
            cycle.append(eid)
            v = u
        cycles.append(tuple(cycle))

    basis = tuple(cycles)
    return Readout(seed, sum(len(c) for c in basis), signature(edges, basis), basis)


@dataclass(frozen=True)
class GhostMemory:
    """Learned policy can be serialized and applied without seeing training graphs."""
    selected_policy: str
    training_instances: int
    seed: int
    normalized_losses: dict[str, float]
    training_probe_edge_visits: int
    baseline_probe_edge_visits: int
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "GhostMemory":
        data = json.loads(raw)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported ghost-memory schema")
        if data.get("selected_policy") not in POLICIES:
            raise ValueError("Unknown saved policy")
        if set(data.get("normalized_losses", {})) != set(POLICIES):
            raise ValueError("Saved memory has an incomplete policy table")
        if data.get("training_instances", 0) <= 0:
            raise ValueError("Invalid training instance count")
        return cls(**data)


def train_memory(
    training: Iterable[tuple[int, list[Edge]]], seed: int = 20261010
) -> GhostMemory:
    """Select one policy via held-IN empirical risk, never by looking at test data.

    Loss is bounded by 1: sum of fundamental cycle lengths / max(1, m*n).
    This is a measurement proxy, not wall-clock runtime or hardware energy.
    """
    rng = Random(seed)
    losses = {policy: 0.0 for policy in POLICIES}
    visits = {policy: 0 for policy in POLICIES}
    count = 0
    for n, edges in training:
        _validate(n, edges)
        trial_seed = rng.randrange(1 << 63)
        denominator = max(1, len(edges) * max(1, n))
        for policy in POLICIES:
            observation = policy_readout(n, edges, policy, trial_seed)
            losses[policy] += observation.cost / denominator
            visits[policy] += observation.cost
        count += 1
    if count == 0:
        raise ValueError("At least one training graph is needed")
    average = {p: losses[p] / count for p in POLICIES}
    chosen = min(POLICIES, key=lambda p: average[p])
    return GhostMemory(
        chosen, count, seed, average,
        sum(visits.values()), visits["random"]
    )


def apply_memory(n: int, edges: list[Edge], memory: GhostMemory, seed: int = 0) -> Readout:
    return policy_readout(n, edges, memory.selected_policy, seed)


def evaluate_memory(
    memory: GhostMemory, held_out: Iterable[tuple[int, list[Edge]]],
    seed: int = 20261011
) -> dict:
    """Paired unseen-instance evaluation; include an amortization proxy."""
    rng = Random(seed)
    baseline = 0
    learned = 0
    count = 0
    improved = tied = worsened = 0
    for n, edges in held_out:
        trial_seed = rng.randrange(1 << 63)
        a = policy_readout(n, edges, "random", trial_seed).cost
        b = apply_memory(n, edges, memory, trial_seed).cost
        baseline += a
        learned += b
        count += 1
        if b < a:
            improved += 1
        elif b == a:
            tied += 1
        else:
            worsened += 1
    if count == 0:
        raise ValueError("Need held-out graphs")
    savings_per_graph = (baseline - learned) / count
    extra_probes = memory.training_probe_edge_visits - memory.baseline_probe_edge_visits
    # An intentionally optimistic surrogate: ignores feature computation and sorting.
    break_even = (extra_probes / savings_per_graph) if savings_per_graph > 0 else None
    return {
        "selected_policy": memory.selected_policy,
        "held_out_instances": count,
        "baseline_edge_visits": baseline,
        "memory_edge_visits": learned,
        "readout_savings_percent": (
            100 * (baseline - learned) / baseline if baseline else 0.0
        ),
        "improved": improved,
        "tied": tied,
        "worsened": worsened,
        "extra_training_probe_visits": extra_probes,
        "proxy_break_even_future_graphs": break_even,
        "metric_warning": (
            "Edge visits are a cycle-length proxy, not latency/energy. "
            "The break-even estimate excludes feature, sort, and verification costs."
        ),
    }
