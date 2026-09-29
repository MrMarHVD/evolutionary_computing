"""Fast checks for controller semantics, variation, persistence and continuation.

Run from the repository root with:
uv run python -m unittest discover -s assignments/assignment_2 -p test_A2_2026.py -v
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import A2_template_2026 as a


def settings(**overrides):
    values = dict(population=4, generations=2, duration=0.04, workers=1,
                  mutation_rate=0.1, mutation_sigma=0.15, crossover_rate=0.2,
                  tournament=3, search_mode="focused", algorithm="ea",
                  patience=0, min_improvement=0.0001, target_radius=0.1,
                  resume=None)
    return argparse.Namespace(**(values | overrides))


class FixedScale:
    """Use a real RNG, but select one mutation scale for operator edge cases."""

    def __init__(self, scale):
        self.rng = a.np.random.default_rng(99)
        self.scale = scale

    def choice(self, *args, **kwargs):
        return self.scale

    def __getattr__(self, name):
        return getattr(self.rng, name)


class Assignment2Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        a.mj.set_mjcb_control(None)
        world = a.build_world()
        world.spawn(a.build_robot().spec, position=a.SPAWN_POS, rotation=[0, 0, 180],
                    correct_collision_with_floor=True)
        cls.model = world.spec.compile()
        cls.length = a._input_size(cls.model) * a.HIDDEN_SIZE + (a.HIDDEN_SIZE + 1) * cls.model.nu

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_path = self.root / "model.mjb"
        a.mj.mj_saveModel(self.model, str(self.model_path), None)
        a._worker_init(str(self.model_path))

    def tearDown(self):
        a.mj.set_mjcb_control(None)
        self.temporary.cleanup()

    def train(self, name, args, seed=42):
        path = self.root / name
        path.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            result, history = a._train(self.model, args, seed, path, None, self.model_path)
        return path, result, history

    def test_inputs_heading_and_actuator_contract(self):
        data = a.mj.MjData(self.model)
        a.mj.mj_forward(self.model, data)
        self.assertEqual(a._controller_inputs(data).shape, (36,))
        self.assertEqual(self.length, 360)
        a.np.testing.assert_allclose(a._heading_features(data), [0, 1], atol=1e-12)
        # Target is now to the snake's left, rather than straight ahead.
        data.qpos[1] = -2
        sine, cosine = a._heading_features(data)
        self.assertGreater(sine, 0)
        self.assertAlmostEqual(sine, cosine)
        data.qpos[:2] = a.TARGET_POSITION[:2]
        self.assertEqual(a._heading_features(data), (0.0, 1.0))
        weights = a._decode(a.np.full(self.length, 5.0), self.model)
        actions = a.nn_controller(self.model, data, weights)
        self.assertEqual(actions.shape, (self.model.nu,))
        self.assertTrue(a.np.isfinite(actions).all())
        self.assertTrue((abs(actions) <= a.np.pi / 2).all())

    def test_fitness_is_final_planar_distance(self):
        self.assertEqual(a.fitness_function(None, a.np.array([2, 0, 100])), 0)
        self.assertEqual(a.fitness_function(None, a.np.array([5, 4, -20])), 5)

    def test_changed_clock_count_keeps_controller_dimensions_consistent(self):
        with patch.object(a, "CLOCK_FREQUENCIES", (0.5, 1.0, 2.0)):
            data = a.mj.MjData(self.model)
            a.mj.mj_forward(self.model, data)
            inputs = a._controller_inputs(data)
            self.assertEqual(len(inputs), a._input_size(self.model))
            self.assertEqual(len(inputs), 38)
            length = len(inputs) * a.HIDDEN_SIZE + (a.HIDDEN_SIZE + 1) * self.model.nu
            weights = a._decode(a.np.zeros(length), self.model)
            self.assertEqual(a.nn_controller(self.model, data, weights).shape, (self.model.nu,))

    def test_saved_brain_rejects_changed_controller_semantics(self):
        saved = a._metadata(self.model, settings(), 42, self.model_path)
        saved["genotype"] = a.np.zeros(self.length).tolist()
        # New and legacy schema-3 brains both remain readable at the defaults.
        a._load_brain(saved, self.model)
        legacy = {key: value for key, value in saved.items() if key != "controller_settings"}
        a._load_brain(legacy, self.model)
        # Dimensions alone cannot detect a change to the clock frequencies.
        with patch.object(a, "CLOCK_FREQUENCIES", (0.5, 1.0)):
            for record in (saved, legacy):
                with self.assertRaisesRegex(ValueError, "controller settings"):
                    a._load_brain(record, self.model)

    def test_file_defaults_and_command_line_override_reach_training(self):
        with (patch.multiple(a, POPULATION_SIZE=2, GENERATIONS=0, SEEDS=[7],
                             SIM_DURATION=0.004, WORKERS=1, NO_VIEW=True,
                             OUTPUT_DIR=str(self.root / "cli")),
              patch.object(a.sys, "argv", ["A2_template_2026.py", "--population", "3"]),
              patch.object(a, "_plot_histories"),
              contextlib.redirect_stdout(io.StringIO())):
            a._cli(self.model)
        pointer = json.loads((self.root / "cli" / "latest_best.json").read_text())
        saved = json.loads(Path(pointer["brain"]).read_text())
        self.assertEqual(saved["seed"], 7)
        self.assertEqual(saved["duration"], 0.004)
        self.assertEqual(saved["evaluations"], 3)
        self.assertEqual(saved["settings"]["population"], 3)
        self.assertEqual(saved["generation"], 0)

    def test_bad_chromosomes_are_rejected(self):
        for genes in ([0.0], a.np.full(self.length, a.np.nan)):
            with self.assertRaises(ValueError):
                a._decode(genes, self.model)

    def test_evaluation_resets_state_and_clears_callback(self):
        data = a.mj.MjData(self.model)
        weights = a._decode(a.np.random.default_rng(3).normal(0, 0.5, self.length), self.model)
        first = a._simulate(self.model, data, weights, 0.06, trace=True)
        data.qpos[:] = 100
        data.qvel[:] = -20
        data.ctrl[:] = 1
        second = a._simulate(self.model, data, weights, 0.06, trace=True)
        self.assertEqual(first, second)
        self.assertTrue(first[2])
        self.assertAlmostEqual(data.time, 0.06)
        self.assertIsNone(a.mj.get_mjcb_control())

    def test_nonfinite_controller_fails_without_leaking_callback(self):
        data = a.mj.MjData(self.model)
        with patch.object(a, "nn_controller", return_value=a.np.full(self.model.nu, a.np.nan)):
            score, _, valid = a._simulate(self.model, data, None, 0.04)
        self.assertEqual(score, a.BAD_FITNESS)
        self.assertFalse(valid)
        self.assertIsNone(a.mj.get_mjcb_control())

    def test_fine_offspring_change_one_weight_and_need_evaluation(self):
        parents = a.Population([a._new_individual(a.np.zeros(self.length)) for _ in range(4)])
        for ind in parents:
            ind.fitness = 2
        a._breed(parents, FixedScale(0.2), settings(population=40), self.length)
        children = list(parents)[4:]
        self.assertEqual(len(children), 40)
        for child in children:
            self.assertEqual(a.np.count_nonzero(child.genotype), 1)
            self.assertTrue(child.requires_eval)
            self.assertIsNone(child.fitness_)
        for parent in list(parents)[:4]:
            self.assertEqual(a.np.count_nonzero(parent.genotype), 0)
            self.assertEqual(parent.fitness, 2)

    def test_focused_mutation_changes_a_clipped_or_empty_mask(self):
        for scale in (0.2, 1.0, 3.0):
            parent = a._new_individual(a.np.full(self.length, a.WEIGHT_LIMIT))
            parent.fitness = 2
            pop = a.Population([parent])
            a._breed(pop, FixedScale(scale), settings(population=30, mutation_rate=1e-30), self.length)
            for child in list(pop)[1:]:
                genes = a.np.asarray(child.genotype)
                self.assertFalse(a.np.array_equal(genes, parent.genotype))
                self.assertTrue((abs(genes) <= a.WEIGHT_LIMIT).all())

    def test_distinct_survival_preserves_elite_and_archive(self):
        individuals = [a._new_individual([gene]) for gene in (0, 0, 1, 2)]
        for individual, score in zip(individuals, (0.0, 0.0, 1.0, 2.0)):
            individual.fitness = score
        pop = a._survive(a.Population(individuals), 3, unique=True)
        self.assertEqual(len(pop), 4)  # Rejections still reach the SQLite archive.
        self.assertEqual([i.genotype for i in pop.alive], [[0.0], [1.0], [2.0]])
        self.assertTrue(individuals[0].alive)
        # Even a completely duplicate input population keeps the requested size.
        duplicates = a.Population([a._new_individual([0]) for _ in range(5)])
        for ind in duplicates:
            ind.fitness = 0
        self.assertEqual(len(a._survive(duplicates, 3, unique=True).alive), 3)

    def test_random_search_does_not_depend_on_parent_fitness(self):
        outputs = []
        for scores in ((0, 1), (100, -100)):
            pop = a.Population([a._new_individual(a.np.zeros(self.length)) for _ in scores])
            for ind, score in zip(pop, scores):
                ind.fitness = score
            a._breed(pop, a.np.random.default_rng(8), settings(algorithm="random"), self.length)
            outputs.append([ind.genotype for ind in list(pop)[2:]])
        self.assertEqual(outputs[0], outputs[1])

    def test_arena_seed_reproduces_terrain_independently_of_ea_seed(self):
        with patch.object(a, "WORLD_FACTORY", a.OlympicArena):
            first = a.build_world().spec.compile()
            a.set_seed(9876)
            second = a.build_world().spec.compile()
            a.np.testing.assert_array_equal(first.hfield_data, second.hfield_data)
            with patch.object(a, "TERRAIN_SEED", a.TERRAIN_SEED + 1):
                different = a.build_world().spec.compile()
            self.assertFalse(a.np.array_equal(first.hfield_data, different.hfield_data))

    def test_adaptive_children_inherit_sigma_and_evaluation_preserves_it(self):
        parent = a._new_individual(a.np.zeros(self.length))
        parent.fitness = 2.0
        parent.tags = {"mutation_sigma": 0.01}
        pop = a.Population([parent])
        # No sigma drift makes inheritance directly observable; a nearly empty
        # mutation mask must still produce a new chromosome for evaluation.
        with patch.object(a, "ADAPTIVE_SIGMA_TAU", 0.0):
            a._breed(pop, a.np.random.default_rng(9),
                     settings(search_mode="adaptive", mutation_rate=1e-30), self.length)
        for child in list(pop)[1:]:
            self.assertEqual(child.tags["mutation_sigma"], 0.01)
            self.assertFalse(a.np.array_equal(child.genotype, parent.genotype))
            self.assertTrue(child.requires_eval)
        a._evaluate_population(pop, None, 0.004)
        for child in list(pop)[1:]:
            self.assertEqual(child.tags["mutation_sigma"], 0.01)
            self.assertIn("final_position", child.tags)
            self.assertTrue(child.tags["valid"])
        self.assertEqual(parent.genotype, a.np.zeros(self.length).tolist())
        self.assertEqual(parent.fitness, 2.0)

    def test_adaptive_sigma_and_weight_bounds_with_clipped_parents(self):
        parent = a._new_individual(a.np.full(self.length, a.WEIGHT_LIMIT))
        parent.fitness = 2.0
        for sigma in (1e-9, 100):
            parent.tags = {"mutation_sigma": sigma}
            pop = a.Population([parent])
            a._breed(pop, a.np.random.default_rng(8),
                     settings(search_mode="adaptive", population=40), self.length)
            for child in list(pop)[1:]:
                self.assertGreaterEqual(child.tags["mutation_sigma"], a.ADAPTIVE_SIGMA_MIN)
                self.assertLessEqual(child.tags["mutation_sigma"], a.ADAPTIVE_SIGMA_MAX)
                self.assertTrue((abs(a.np.asarray(child.genotype)) <= a.WEIGHT_LIMIT).all())
                self.assertFalse(a.np.array_equal(child.genotype, parent.genotype))

    def test_split_resume_matches_full_run_and_counts_evaluations(self):
        for mode in ("standard", "mixed", "focused", "adaptive"):
            full, result, history = self.train(mode + "_full", settings(generations=3, search_mode=mode))
            part, _, _ = self.train(mode + "_part", settings(generations=1, search_mode=mode))
            continuation = settings(generations=2, search_mode=mode, resume=str(part))
            continuation._continuation = a._continuation_state(part)
            resumed, resumed_result, _ = self.train(mode + "_resumed", continuation)
            full_state = a._continuation_state(full)
            resumed_state = a._continuation_state(resumed)
            for field in ("population", "rng_state", "generation", "evaluations",
                          "last_improvement", "previous_best"):
                self.assertEqual(full_state[field], resumed_state[field], (mode, field))
            self.assertEqual(result["fitness"], resumed_result["fitness"])
            self.assertTrue(result["replay_verified"])
            self.assertEqual(result["evaluations"], 16)
            self.assertEqual(resumed_result["new_evaluations"], 8)
            self.assertTrue(all(right["best"] <= left["best"] for left, right in zip(history, history[1:])))
            with contextlib.closing(sqlite3.connect(full / "evolution.sqlite")) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM individual").fetchone()[0], 16)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM individual WHERE alive=1").fetchone()[0], 4)

    def test_baseline_initial_population_and_budget_match_ea(self):
        ea_run, ea_result, _ = self.train("ea", settings())
        random_run, random_result, _ = self.train("random", settings(algorithm="random"))
        self.assertEqual(ea_result["evaluations"], random_result["evaluations"])
        initial = []
        for run in (ea_run, random_run):
            with contextlib.closing(sqlite3.connect(run / "evolution.sqlite")) as db:
                initial.append(db.execute("SELECT genotype_, fitness_ FROM individual ORDER BY id LIMIT 4").fetchall())
        self.assertEqual(initial[0], initial[1])

    def test_explicit_adaptive_sigma_change_resets_strategy_but_keeps_brains(self):
        part, _, _ = self.train("adaptive_source", settings(search_mode="adaptive", generations=1))
        state = a._continuation_state(part)
        continuation = settings(search_mode="adaptive", generations=0, mutation_sigma=0.025,
                                resume=str(part))
        continuation._continuation = state
        resumed, _, _ = self.train("adaptive_reset", continuation)
        restored = a._continuation_state(resumed)
        self.assertEqual([r["genotype"] for r in state["population"]],
                         [r["genotype"] for r in restored["population"]])
        self.assertTrue(all(r["tags"]["mutation_sigma"] == 0.025 for r in restored["population"]))
        self.assertEqual(restored["last_improvement"], state["generation"])
        self.assertEqual(restored["metadata"]["terrain_seed"], state["metadata"]["terrain_seed"])

    def test_direct_replay_uses_archived_physics_and_checks_checksum(self):
        run, result, _ = self.train("saved", settings(generations=0))
        pointer = self.root / "latest_best.json"
        pointer.write_text(json.dumps({"brain": result["brain"]}), encoding="utf-8")
        original_world = a.build_world

        def changed_world():
            world = original_world()
            world.spec.option.gravity[:] = [0, 0, -4.0]
            return world

        with patch.object(a, "LATEST_BRAIN", pointer), patch.object(a, "build_world", changed_world):
            with patch.object(a, "_simulate", wraps=a._simulate) as simulate:
                score = a.run_experiment("simple")
                a.np.testing.assert_array_equal(simulate.call_args.args[0].opt.gravity, self.model.opt.gravity)
        self.assertEqual(score, result["fitness"])
        saved = json.loads((run / "best_brain.json").read_text())
        saved["model_sha256"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "checksum"):
            a._archived_model(run / "best_brain.json", saved)


if __name__ == "__main__":
    unittest.main()
