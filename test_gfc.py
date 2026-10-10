import random
import unittest

from gfc import Edge, certify, cycle_basis, optimize_readout, readout, verify_assignment, verify_conflict


def brute_force(n, edges):
    for k in range(1 << n):
        a = tuple((k >> i) & 1 for i in range(n))
        if verify_assignment(n, edges, a):
            return True
    return False


class GhostFluxTests(unittest.TestCase):
    def test_triangles(self):
        good = [Edge(0, 1, 0), Edge(1, 2, 0), Edge(2, 0, 0)]
        bad = good[:-1] + [Edge(2, 0, 1)]
        self.assertTrue(certify(3, good).satisfiable)
        answer = certify(3, bad)
        self.assertFalse(answer.satisfiable)
        self.assertTrue(verify_conflict(3, bad, answer.conflict_cycle))

    def test_loops_parallel_disconnected(self):
        for n, edges in [
            (0, []),
            (3, []),
            (1, [Edge(0, 0, 1)]),
            (2, [Edge(0, 1, 0), Edge(0, 1, 1)]),
        ]:
            result = certify(n, edges)
            self.assertEqual(result.satisfiable, brute_force(n, edges))
            if result.satisfiable:
                self.assertTrue(verify_assignment(n, edges, result.assignment))
            else:
                self.assertTrue(verify_conflict(n, edges, result.conflict_cycle))

    def test_randomized_against_brute_force(self):
        rng = random.Random(20261010)
        for _ in range(500):
            n = rng.randrange(0, 9)
            m = rng.randrange(0, 19) if n else 0
            edges = [Edge(rng.randrange(n), rng.randrange(n), rng.randrange(2))
                     for _ in range(m)]
            result = certify(n, edges)
            self.assertEqual(result.satisfiable, brute_force(n, edges))
            if result.satisfiable:
                self.assertTrue(verify_assignment(n, edges, result.assignment))
            else:
                self.assertTrue(verify_conflict(n, edges, result.conflict_cycle))
            baseline = readout(n, edges, seed=0)
            trained = optimize_readout(n, edges, trials=20, seed=77)
            self.assertEqual(any(baseline.signature), any(trained.signature))
            self.assertEqual(any(trained.signature), not result.satisfiable)
            cycles = cycle_basis(n, edges, 11)
            self.assertEqual(len(cycles), len(edges) - n + self._components(n, edges))

    @staticmethod
    def _components(n, edges):
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for e in edges:
            parent[find(e.u)] = find(e.v)
        return len({find(i) for i in range(n)})


if __name__ == "__main__":
    unittest.main()
