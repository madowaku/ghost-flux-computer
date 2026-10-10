"""Ghost Flux Computer v0.5: evolved, interpretable topology-only readout genomes.

A genome is a linear ranking rule for Kruskal's spanning-forest construction.
Mutation and crossover search a bounded rule grammar. Every resulting forest
is an exact cycle-basis readout, independent of all XOR edge labels.
This is evolutionary hyperparameter optimization, NOT proven new computation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from random import Random
from time import perf_counter
from typing import Iterable

from gfc import Edge, Readout, _validate, signature
from ghost_memory import POLICIES, policy_readout

GENOME_SCHEMA = 1
WEIGHT_BOUND = 6
FEATURE_NAMES = ("endpoint_degree_sum", "common_neighbors", "degree_gap", "degree_product_scaled")


@dataclass(frozen=True)
class Genome:
    """Four signed integer weights defining a new spanning-forest priority rule."""
    weights: tuple[int, int, int, int]

    def __post_init__(self):
        if len(self.weights) != 4 or any(
            type(w) is not int or not -WEIGHT_BOUND <= w <= WEIGHT_BOUND
            for w in self.weights
        ):
            raise ValueError("Genome requires four integer weights in [-6,6]")

    def mutate(self, rng: Random) -> "Genome":
        values = list(self.weights)
        i = rng.randrange(len(values))
        values[i] = max(-WEIGHT_BOUND, min(WEIGHT_BOUND, values[i] + rng.choice((-2, -1, 1, 2))))
        if rng.random() < 0.2:
            j = rng.randrange(len(values))
            values[j] = rng.randrange(-WEIGHT_BOUND, WEIGHT_BOUND + 1)
        return Genome(tuple(values))

    def cross(self, other: "Genome", rng: Random) -> "Genome":
        return Genome(tuple(
            a if rng.random() < 0.5 else b for a, b in zip(self.weights, other.weights)
        ))


@dataclass(frozen=True)
class Prepared:
    n: int
    edges: tuple[Edge, ...]
    features: tuple[tuple[float, float, float, float], ...]


def prepare(n: int, edges: list[Edge]) -> Prepared:
    """Features depend ONLY on topology and can be reused across all candidates."""
    _validate(n, edges)
    degree = [0] * n
    neighbors = [set() for _ in range(n)]
    for e in edges:
        degree[e.u] += 1
        degree[e.v] += 1
        if e.u != e.v:
            neighbors[e.u].add(e.v)
            neighbors[e.v].add(e.u)
    edge_features = []
    for e in edges:
        du, dv = degree[e.u], degree[e.v]
        edge_features.append((
            float(du + dv),
            float(len(neighbors[e.u] & neighbors[e.v])),
            float(abs(du - dv)),
            float(du * dv) / max(1, n),
        ))
    return Prepared(n, tuple(edges), tuple(edge_features))


def _cycle_readout(
    prep: Prepared, edge_order: list[int], seed: int
) -> Readout:
    """Interpret a chosen edge order as a spanning forest, then extract its cycles."""
    n, edges = prep.n, prep.edges
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = [[] for _ in range(n)]
    chords = []
    for eid in edge_order:
        e = edges[eid]
        a, b = find(e.u), find(e.v)
        if a == b:
            chords.append(eid)
        else:
            parent[a] = b
            tree[e.u].append((e.v, eid))
            tree[e.v].append((e.u, eid))

    cycles = []
    for chord in chords:
        e = edges[chord]
        q = deque([e.u])
        previous = {e.u: (-1, -1)}
        while q and e.v not in previous:
            u = q.popleft()
            for v, eid in tree[u]:
                if v not in previous:
                    previous[v] = (u, eid)
                    q.append(v)
        path = [chord]
        v = e.v
        while v != e.u:
            u, eid = previous[v]
            path.append(eid)
            v = u
        cycles.append(tuple(path))
    basis = tuple(cycles)
    return Readout(seed, sum(map(len, basis)), signature(list(edges), basis), basis)


def genome_readout(prep: Prepared, genome: Genome, seed: int = 0) -> Readout:
    """One deterministic genome-defined readout, seeded random tie resolution."""
    rng = Random(seed)
    order = list(range(len(prep.edges)))
    rng.shuffle(order)
    weights = genome.weights
    order.sort(
        key=lambda eid: sum(w * f for w, f in zip(weights, prep.features[eid])),
        reverse=True,
    )
    return _cycle_readout(prep, order, seed)


def bfs_readout(prep: Prepared, seed: int = 0) -> Readout:
    """Classical BFS spanning forest, independent of edge labels."""
    rng = Random(seed)
    n, edges = prep.n, prep.edges
    neighbors = [[] for _ in range(n)]
    for eid, e in enumerate(edges):
        neighbors[e.u].append((e.v, eid))
        if e.u != e.v:
            neighbors[e.v].append((e.u, eid))
    for row in neighbors:
        rng.shuffle(row)
    seen = [False] * n
    forest_edge_ids = []
    roots = list(range(n))
    rng.shuffle(roots)
    for root in roots:
        if seen[root]:
            continue
        seen[root] = True
        q = deque([root])
        while q:
            u = q.popleft()
            for v, eid in neighbors[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
                    forest_edge_ids.append(eid)
    chosen = set(forest_edge_ids)
    return _cycle_readout(
        prep, forest_edge_ids + [i for i in range(len(edges)) if i not in chosen], seed
    )


@dataclass(frozen=True)
class GenesisMemory:
    genome: Genome
    training_graphs: int
    training_candidates: int
    generated_candidates: int
    best_train_normalized_loss: float
    total_train_cycle_edge_visits: int
    search_seconds: float
    seed: int
    schema_version: int = GENOME_SCHEMA

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "GenesisMemory":
        payload = json.loads(raw)
        if payload.get("schema_version") != GENOME_SCHEMA:
            raise ValueError("Unknown genesis memory schema")
        payload["genome"] = Genome(tuple(payload["genome"]["weights"]))
        if payload.get("training_graphs", 0) <= 0:
            raise ValueError("Invalid training count")
        return cls(**payload)


INITIAL_GENOMES = (
    Genome((0, 0, 0, 0)),   # random order
    Genome((1, 0, 0, 0)),   # hub
    Genome((-1, 0, 0, 0)),  # anti_hub
    Genome((0, 1, 0, 0)),   # triangle
    Genome((1, 3, 0, 0)),   # hub_triangle
    Genome((0, -1, 0, 0)),  # low_triangle
)
INITIAL_WEIGHTS = frozenset(g.weights for g in INITIAL_GENOMES)


def train_genesis(
    training: Iterable[tuple[int, list[Edge]]],
    *,
    generations: int = 12,
    population: int = 20,
    seed: int = 20261010,
) -> GenesisMemory:
    """Evolutionary search over readout construction rules, never over labels.

    Training loss is the instance-averaged normalized fundamental-cycle length.
    Seeded graph-level readout tie choices are shared across genome candidates.
    """
    if generations < 1 or population < len(INITIAL_GENOMES):
        raise ValueError("generations >= 1 and population >= 6 required")
    graphs = [prepare(n, edges) for n, edges in training]
    if not graphs:
        raise ValueError("Need at least one training graph")
    rng = Random(seed)
    tie_seeds = [rng.randrange(1 << 63) for _ in graphs]
    pop = list(INITIAL_GENOMES)
    while len(pop) < population:
        pop.append(Genome(tuple(rng.randrange(-6, 7) for _ in range(4))))
    cache: dict[Genome, float] = {}
    probes = 0
    generated = 0
    start = perf_counter()

    def fitness(genome: Genome) -> float:
        nonlocal probes
        if genome not in cache:
            total = 0.0
            for prep, tie in zip(graphs, tie_seeds):
                readout = genome_readout(prep, genome, tie)
                probes += readout.cost
                total += readout.cost / max(1, len(prep.edges) * max(1, prep.n))
            cache[genome] = total / len(graphs)
        return cache[genome]

    # Evaluate all members even in the final generation, and preserve elites.
    for step in range(generations):
        for g in pop:
            fitness(g)
        ranked = sorted(set(pop), key=lambda g: (fitness(g), g.weights))
        if step == generations - 1:
            break
        elite = ranked[:max(2, population // 4)]
        parents = ranked[:max(4, population // 2)]
        next_pop = list(elite)
        next_pop.extend(INITIAL_GENOMES)  # safe, known classical baselines remain
        while len(next_pop) < population:
            a = rng.choice(parents)
            b = rng.choice(parents)
            child = a.cross(b, rng).mutate(rng) if rng.random() < 0.6 else a.mutate(rng)
            generated += 1
            next_pop.append(child)
        pop = next_pop[:population]

    best = min(cache, key=lambda g: (cache[g], g.weights))
    return GenesisMemory(
        genome=best,
        training_graphs=len(graphs),
        training_candidates=len(cache),
        generated_candidates=generated,
        best_train_normalized_loss=cache[best],
        total_train_cycle_edge_visits=probes,
        search_seconds=perf_counter() - start,
        seed=seed,
    )


def evaluate_genesis(
    memory: GenesisMemory,
    graphs: Iterable[tuple[int, list[Edge]]],
    *,
    seed: int = 20261011,
    meta_memory=None,
) -> dict:
    """Paired, unseen-only policy benchmark; posthoc oracle is non-deployable."""
    rng = Random(seed)
    names = ("genesis", "bfs", "random", "hub", "meta", "oracle6")
    totals = {name: 0 for name in names}
    wins_hub = ties_hub = losses_hub = 0
    novel_layouts = better_than_six = better_than_bfs = 0
    count = 0
    inference_seconds = {key: 0.0 for key in names if key != "oracle6"}
    for n, edges in graphs:
        prep = prepare(n, edges)
        tie = rng.randrange(1 << 63)
        t0 = perf_counter()
        created = genome_readout(prep, memory.genome, tie)
        inference_seconds["genesis"] += perf_counter() - t0
        t0 = perf_counter()
        bfs = bfs_readout(prep, tie)
        inference_seconds["bfs"] += perf_counter() - t0
        t0 = perf_counter()
        baseline = {
            p: policy_readout(n, edges, p, tie)
            for p in POLICIES
        }
        inference_seconds["random"] += perf_counter() - t0
        if meta_memory is not None:
            t0 = perf_counter()
            meta = policy_readout(n, edges, meta_memory.choose(n, edges), tie)
            inference_seconds["meta"] += perf_counter() - t0
        else:
            meta = baseline["hub"]

        costs = {
            "genesis": created.cost,
            "bfs": bfs.cost,
            "random": baseline["random"].cost,
            "hub": baseline["hub"].cost,
            "meta": meta.cost,
            "oracle6": min(r.cost for r in baseline.values()),
        }
        for key, value in costs.items():
            totals[key] += value
        if created.cost < baseline["hub"].cost:
            wins_hub += 1
        elif created.cost == baseline["hub"].cost:
            ties_hub += 1
        else:
            losses_hub += 1
        novel_layouts += int(all(created.cycles != b.cycles for b in baseline.values()))
        better_than_six += int(created.cost < costs["oracle6"])
        better_than_bfs += int(created.cost < bfs.cost)
        count += 1
    if not count:
        raise ValueError("Need held-out graphs")
    return {
        "graphs": count,
        "cycle_edge_visits": totals,
        "genesis_vs_hub_percent": (
            100 * (totals["hub"] - totals["genesis"]) / totals["hub"] if totals["hub"] else 0
        ),
        "genesis_vs_bfs_percent": (
            100 * (totals["bfs"] - totals["genesis"]) / totals["bfs"] if totals["bfs"] else 0
        ),
        "wins_ties_losses_vs_hub": [wins_hub, ties_hub, losses_hub],
        "distinct_basis_vs_six_on_graphs": novel_layouts,
        "better_than_all_six_on_graphs": better_than_six,
        "better_than_bfs_on_graphs": better_than_bfs,
        "timing_seconds_exploratory": inference_seconds,
        "warnings": (
            "Cycle visits are a proxy; timings are single-pass Python estimates. "
            "The oracle6 compares every policy on test data and is NOT deployable. "
            "The 6-policy timing sums all 6, not single random-policy time."
        ),
    }
