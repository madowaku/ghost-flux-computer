import json
import random
import unittest

from benchmark_memory import make_dataset
from benchmark_meta import dense_dataset, experiment, make_split
from gfc import Edge, certify
from ghost_memory import POLICIES, policy_readout
from meta_memory import (
    FEATURES, MetaMemory, apply_meta_memory, evaluate_meta_memory,
    topology_features, train_meta_memory
)


class MetaMemoryTests(unittest.TestCase):
    def test_structural_features_permutation_and_bit_invariant(self):
        edges = [Edge(0, 1, 0), Edge(1, 2, 1), Edge(2, 0, 1),
                 Edge(2, 3, 0), Edge(2, 3, 1)]
        perm = [3, 2, 1, 0]
        alternate = [Edge(perm[e.u], perm[e.v], e.b ^ 1) for e in reversed(edges)]
        a = topology_features(4, edges)
        b = topology_features(4, alternate)
        for key in FEATURES:
            self.assertAlmostEqual(a[key], b[key])
        self.assertAlmostEqual(a["transitivity"], 0.6)

    def test_graph_corner_cases(self):
        for n, edges in [
            (0, []),
            (1, []),
            (1, [Edge(0, 0, 1)]),
            (2, [Edge(0, 1, 0), Edge(0, 1, 1)]),
        ]:
            features = topology_features(n, edges)
            self.assertTrue(all(x >= 0 for x in features.values()))
            self.assertTrue(all(k in FEATURES for k in features))

    def test_held_out_certificates_unchanged(self):
        rng = random.Random(20261010)
        train = make_split(8, 8, 95001)
        memory = train_meta_memory(train, seed=55, min_leaf=3)
        for _ in range(130):
            n = rng.randrange(1, 10)
            edges = [Edge(rng.randrange(n), rng.randrange(n), rng.randrange(2))
                     for _ in range(rng.randrange(23))]
            result = certify(n, edges)
            readout = apply_meta_memory(n, edges, memory, 103)
            self.assertEqual(not any(readout.signature), result.satisfiable)
            self.assertEqual(readout.cost, sum(map(len, readout.cycles)))

    def test_learning_not_using_constraint_bits(self):
        train = make_split(10, 10, 879)
        flipped = [(n, [Edge(e.u, e.v, e.b ^ 1) for e in edges])
                   for n, edges in train]
        a = train_meta_memory(train, seed=2, min_leaf=3)
        b = train_meta_memory(flipped, seed=2, min_leaf=3)
        self.assertEqual(a, b)

    def test_transfer_serialization_and_split_guarantee(self):
        graphs = make_split(15, 15, 891)
        memory = train_meta_memory(graphs, seed=999, min_leaf=4)
        self.assertLessEqual(memory.train_meta_risk, memory.train_global_risk)
        self.assertEqual(memory, MetaMemory.from_json(memory.to_json()))
        self.assertIn(memory.global_policy, POLICIES)
        if memory.split_feature:
            self.assertIn(memory.split_feature, FEATURES)
            self.assertNotEqual(memory.low_policy, memory.high_policy)
        target = make_dataset("hub", 11, 109)
        report = evaluate_meta_memory(memory, target, seed=221)
        self.assertEqual(report["held_out_graphs"], 11)
        self.assertEqual(sum(report["policy_counts"].values()), 11)
        self.assertEqual(sum(report[k] for k in (
            "meta_vs_global_improved",
            "meta_vs_global_tied", "meta_vs_global_worsened"
        )), 11)

    def test_repeatable_and_invalid_input(self):
        graphs = make_split(5, 5, 770)
        self.assertEqual(train_meta_memory(graphs, 91),
                         train_meta_memory(graphs, 91))
        with self.assertRaises(ValueError):
            train_meta_memory([])
        with self.assertRaises(ValueError):
            train_meta_memory(graphs, min_leaf=0)
        with self.assertRaises(ValueError):
            evaluate_meta_memory(train_meta_memory(graphs), [])
        with self.assertRaises(ValueError):
            MetaMemory.from_json(json.dumps({"schema_version": 123}))

    def test_end_to_end(self):
        report = experiment(train_per_family=7, test_per_family=9)
        self.assertEqual(report["train_count"], 14)
        self.assertEqual(report["evaluation"]["modular_dense_mixture_unseen"]["held_out_graphs"], 18)
        self.assertEqual(len(report["evaluation"]), 5)


if __name__ == "__main__":
    unittest.main()
