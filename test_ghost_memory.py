import json
import random
import unittest

from benchmark_memory import make_dataset, make_graph, run_benchmark
from gfc import Edge, certify
from ghost_memory import (
    GhostMemory, POLICIES, apply_memory, evaluate_memory, policy_readout,
    train_memory,
)


class GhostMemoryTests(unittest.TestCase):
    def test_every_policy_preserves_theorem(self):
        rng = random.Random(301)
        for _ in range(200):
            n = rng.randrange(0, 11)
            m = rng.randrange(0, 26) if n else 0
            edges = [Edge(rng.randrange(n), rng.randrange(n), rng.randrange(2))
                     for _ in range(m)]
            answer = certify(n, edges)
            for policy in POLICIES:
                measurement = policy_readout(n, edges, policy, 31)
                self.assertEqual(not any(measurement.signature), answer.satisfiable)
                self.assertEqual(measurement.cost,
                                 sum(map(len, measurement.cycles)))
                self.assertEqual(len(measurement.cycles), m - n
                                 + self._components(n, edges))

    @staticmethod
    def _components(n, edges):
        leader = list(range(n))
        def find(x):
            while leader[x] != x:
                x = leader[x]
            return x
        for e in edges:
            leader[find(e.u)] = find(e.v)
        return len({find(i) for i in range(n)})

    def test_topology_only_layout_ignores_constraint_bits(self):
        n, original = make_graph("hub", 293)
        flipped = [Edge(e.u, e.v, e.b ^ 1) for e in original]
        for policy in POLICIES:
            a = policy_readout(n, original, policy, 98)
            b = policy_readout(n, flipped, policy, 98)
            self.assertEqual(a.cycles, b.cycles)
            self.assertEqual(a.cost, b.cost)

    def test_serialize_and_transfer(self):
        train = make_dataset("hub", 12, 89)
        memory = train_memory(train, seed=99)
        restored = GhostMemory.from_json(memory.to_json())
        self.assertEqual(restored, memory)
        self.assertIn(memory.selected_policy, POLICIES)
        n, edges = make_graph("grid", 999)  # Never used in training
        self.assertEqual(apply_memory(n, edges, restored, 15),
                         policy_readout(n, edges, memory.selected_policy, 15))
        report = evaluate_memory(memory, make_dataset("grid", 5, 90), seed=94)
        self.assertEqual(report["held_out_instances"], 5)
        self.assertEqual(report["improved"] + report["tied"]
                         + report["worsened"], 5)

    def test_training_determinism(self):
        graphs = make_dataset("hub", 9, 123)
        self.assertEqual(train_memory(graphs, 17), train_memory(graphs, 17))
        self.assertNotEqual(make_graph("hub", 2), make_graph("hub", 3))

    def test_edge_conditions(self):
        with self.assertRaises(ValueError):
            train_memory([])
        with self.assertRaises(ValueError):
            evaluate_memory(train_memory([(0, [])]), [])
        with self.assertRaises(ValueError):
            policy_readout(2, [Edge(0, 1, 0)], "alien")
        with self.assertRaises(ValueError):
            GhostMemory.from_json(json.dumps({"schema_version": 999}))
        with self.assertRaises(ValueError):
            make_dataset("hub", 0, 9)

    def test_small_end_to_end_benchmark(self):
        report = run_benchmark(train_size=6, test_size=7)
        self.assertIn(report["policy"], POLICIES)
        self.assertEqual(report["test"]["matched_hub"]["held_out_instances"], 7)
        self.assertEqual(report["test"]["shifted_grid"]["held_out_instances"], 7)


if __name__ == "__main__":
    unittest.main()
