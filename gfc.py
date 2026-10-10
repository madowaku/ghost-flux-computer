"""Ghost Flux Computer v0.2: exact XOR-constraint certificates and adaptive cycle readout.

Only Python standard library is required.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    b: int

    def __post_init__(self):
        if self.u < 0 or self.v < 0 or self.b not in (0, 1):
            raise ValueError("Vertices must be nonnegative and b must be 0 or 1")


@dataclass(frozen=True)
class Certificate:
    satisfiable: bool
    assignment: tuple[int, ...] | None
    conflict_cycle: tuple[int, ...] | None
    flux: int
    # conflict_cycle lists edge indices; it is an even-degree edge set with XOR(b)=1.


def _validate(n: int, edges: list[Edge]) -> None:
    if n < 0:
        raise ValueError("n must be nonnegative")
    if any(max(e.u, e.v) >= n for e in edges):
        raise ValueError("Edge endpoint outside graph")


def certify(n: int, edges: list[Edge]) -> Certificate:
    """O(n+m) constructive SAT assignment or independently checkable UNSAT cycle."""
    _validate(n, edges)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        adj[e.u].append((e.v, i))
        adj[e.v].append((e.u, i))
    values: list[int | None] = [None] * n
    parent = [-1] * n
    via = [-1] * n
    depth = [0] * n
    for root in range(n):
        if values[root] is not None:
            continue
        values[root] = 0
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v, eid in adj[u]:
                expected = values[u] ^ edges[eid].b
                if values[v] is None:
                    values[v] = expected
                    parent[v] = u
                    via[v] = eid
                    depth[v] = depth[u] + 1
                    queue.append(v)
                elif values[v] != expected:
                    a, b = u, v
                    path: list[int] = [eid]
                    while depth[a] > depth[b]:
                        path.append(via[a]); a = parent[a]
                    while depth[b] > depth[a]:
                        path.append(via[b]); b = parent[b]
                    while a != b:
                        path.extend((via[a], via[b]))
                        a, b = parent[a], parent[b]
                    proof = tuple(path)
                    assert verify_conflict(n, edges, proof)
                    return Certificate(False, None, proof, 1)
    assignment = tuple(int(x) for x in values)
    assert verify_assignment(n, edges, assignment)
    return Certificate(True, assignment, None, 0)


def verify_assignment(n: int, edges: list[Edge], assignment: tuple[int, ...]) -> bool:
    _validate(n, edges)
    return (len(assignment) == n
            and all(x in (0, 1) for x in assignment)
            and all((assignment[e.u] ^ assignment[e.v]) == e.b for e in edges))


def verify_conflict(n: int, edges: list[Edge], proof: tuple[int, ...]) -> bool:
    """Independent certificate checker: even degree at every vertex and odd parity."""
    _validate(n, edges)
    if not proof or len(set(proof)) != len(proof):
        return False
    parity = [0] * n
    flux = 0
    for i in proof:
        if not 0 <= i < len(edges):
            return False
        e = edges[i]
        parity[e.u] ^= 1
        parity[e.v] ^= 1
        flux ^= e.b
    return flux == 1 and not any(parity)


def cycle_basis(n: int, edges: list[Edge], seed: int = 0) -> tuple[tuple[int, ...], ...]:
    """Seed-dependent fundamental cycle basis, including loops and parallel edges."""
    _validate(n, edges)
    rng = Random(seed)
    order = list(range(len(edges)))
    rng.shuffle(order)
    leader = list(range(n))
    def find(x: int) -> int:
        while leader[x] != x:
            leader[x] = leader[leader[x]]
            x = leader[x]
        return x
    tree: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    chords = []
    for i in order:
        e = edges[i]
        a, b = find(e.u), find(e.v)
        if a == b:
            chords.append(i)
        else:
            leader[a] = b
            tree[e.u].append((e.v, i))
            tree[e.v].append((e.u, i))
    cycles = []
    for chord in chords:
        e = edges[chord]
        q = deque([e.u])
        predecessors = {e.u: (-1, -1)}
        while q and e.v not in predecessors:
            u = q.popleft()
            for v, i in tree[u]:
                if v not in predecessors:
                    predecessors[v] = (u, i)
                    q.append(v)
        path = [chord]
        at = e.v
        while at != e.u:
            at_parent, i = predecessors[at]
            path.append(i)
            at = at_parent
        cycles.append(tuple(path))
    return tuple(cycles)


def signature(edges: list[Edge], cycles: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(sum(edges[i].b for i in cycle) % 2 for cycle in cycles)


@dataclass(frozen=True)
class Readout:
    seed: int
    cost: int
    signature: tuple[int, ...]
    cycles: tuple[tuple[int, ...], ...]


def readout(n: int, edges: list[Edge], seed: int = 0) -> Readout:
    cycles = cycle_basis(n, edges, seed)
    return Readout(seed, sum(map(len, cycles)), signature(edges, cycles), cycles)


def optimize_readout(n: int, edges: list[Edge], trials: int = 100, seed: int = 0) -> Readout:
    """Optimize a proxy (total edge visits), not actual energy/time or full computation cost."""
    if trials < 1:
        raise ValueError("trials must be >= 1")
    rng = Random(seed)
    candidates = [readout(n, edges, rng.randrange(2**63)) for _ in range(trials)]
    best = min(candidates, key=lambda item: (item.cost, item.seed))
    assert any(best.signature) == (not certify(n, edges).satisfiable)
    return best
