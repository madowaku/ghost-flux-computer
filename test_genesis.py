import json
import random
import unittest

from benchmark_memory import make_dataset
from benchmark_genesis import experiment
from genesis import (
    FEATURE_NAMES, INITIAL_WEIGHTS, Genome, GenesisMemory,
    bfs_readout, evaluate_genesis, genome_readout, prepare, train_genesis,
)
from gfc import Edge, certify
from ghost_memory import POLICIES, policy_readout


class GenesisTests(unittest.TestCase):
    def test_genome_validation_and_mutation(self):
        for weights in ((9, 0, 0, 0), (0, 0, 0), (1.0, 0, 0, 0)):
            with self.assertRaises(ValueError):
                Genome(weights)
        rng = random.Random(43)
        a, b = Genome((3, 0, -3, 2)), Genome((-3, 6, 2, 0))
        for _ in range(120):
            child = a.cross(b, rng).mutate(rng)
            self.assertTrue(all(-6 <= w <= 6 for w in child.weights))
        self.assertEqual(len(FEATURE_NAMES), 4)

    def test_topology_blindness(self):
        n = 5
        edges = [
            Edge(0, 1, 0), Edge(1, 2, 0), Edge(2, 0, 1),
            Edge(2, 3, 1), Edge(3, 4, 0), Edge(4, 2, 0),
        ]
        flipped = [Edge(e.u, e.v, e.b ^ 1) for e in edges]
        p1, p2 = prepare(n, edges), prepare(n, flipped)
        self.assertEqual(p1.features, p2.features)
        for g in (Genome((2, 4, -1, 3)), Genome((-6, 1, 6, -2))):
            self.assertEqual(genome_readout(p1, g, 38).cycles,
                             genome_readout(p2, g, 38).cycles)
        self.assertEqual(bfs_readout(p1, 15).cycles, bfs_readout(p2, 15).cycles)

    def test_cycle_basis_exactness_and_edge_cases(self):
        rng = random.Random(51)
        genome = Genome((5, -3, 2, -1))
        for _ in range(220):
            n = rng.randrange(0, 11)
            m = rng.randrange(0, 26) if n else 0
            edges = [
                Edge(rng.randrange(n), rng.randrange(n), rng.randrange(2))
                for _ in range(m)
            ]
            prep = prepare(n, edges)
            expected = certify(n, edges).satisfiable
            for observation in (
                genome_readout(prep, genome, 72),
                bfs_readout(prep, 72),
            ):
                self.assertEqual(not any(observation.signature), expected)
                self.assertEqual(observation.cost,
                                 sum(map(len, observation.cycles)))

    def test_memory_serialize_and_determinism(self):
        train = make_dataset("modular", 6, 171)
        a = train_genesis(train, generations=3, population=9, seed=11)
        b = train_genesis(train, generations=3, population=9, seed=11)
        self.assertEqual(a.genome, b.genome)
        self.assertAlmostEqual(a.best_train_normalized_loss,
                               b.best_train_normalized_loss)
        self.assertEqual(a.training_candidates, b.training_candidates)
        self.assertEqual(a, GenesisMemory.from_json(a.to_json()))
        self.assertGreaterEqual(a.training_candidates, len(INITIAL_WEIGHTS))
        self.assertGreater(a.total_train_cycle_edge_visits, 0)

    def test_evaluation_matched_controls(self):
        train = make_dataset("modular", 7, 171)
        memory = train_genesis(train, generations=3, population=9, seed=15)
        heldout = make_dataset("modular", 11, 181)
        report = evaluate_genesis(memory, heldout, seed=71)
        self.assertEqual(report["graphs"], 11)
        self.assertEqual(sum(report["wins_ties_losses_vs_hub"]), 11)
        self.assertLessEqual(report["cycle_edge_visits"]["oracle6"],
                             report["cycle_edge_visits"]["hub"])
        self.assertTrue(0 <= report["better_than_all_six_on_graphs"] <= 11)
        with self.assertRaises(ValueError):
            evaluate_genesis(memory, [])

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            train_genesis([])
        with self.assertRaises(ValueError):
            train_genesis([(0, [])], generations=0)
        with self.assertRaises(ValueError):
            train_genesis([(0, [])], population=5)
        with self.assertRaises(ValueError):
            GenesisMemory.from_json(json.dumps({"schema_version": 999}))

    def test_small_complete_benchmark(self):
        result = experiment(train_per_family=3, test_per_family=5,
                            generations=3, population=9)
        self.assertEqual(result["training_graphs"], 12)
        self.assertEqual(result["evaluations"]["in_family_mixture"]["graphs"], 20)
        self.assertEqual(result["exact_certificates_checked"], 25)


if __name__ == "__main__":
    unittest.main()
