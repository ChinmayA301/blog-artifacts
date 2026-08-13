import importlib.util
import sys
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "pipeline" / "05_persona_conditioned_ic.py"
SPEC = importlib.util.spec_from_file_location("persona_ic", MODULE_PATH)
persona_ic = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = persona_ic
SPEC.loader.exec_module(persona_ic)


class PersonaConditionedICTest(unittest.TestCase):
    def test_activation_probability_is_topic_conditional_and_bounded(self):
        sender = {
            "affinity_ml_paper": 0.9,
            "affinity_sports_clip": 0.1,
            "share_propensity": 0.7,
        }
        receiver = dict(sender)
        ml_probability = persona_ic.activation_probability(
            sender, receiver, persona_ic.TOPICS["ml_paper"]
        )
        sports_probability = persona_ic.activation_probability(
            sender, receiver, persona_ic.TOPICS["sports_clip"]
        )
        self.assertGreater(ml_probability, sports_probability)
        self.assertGreaterEqual(sports_probability, 0.005)
        self.assertLessEqual(ml_probability, 0.60)

    def test_assignment_coupling_changes_only_synthetic_join(self):
        graph = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)])
        signals = pd.DataFrame(
            {
                "source": ["synthetic"] * 6,
                "age_bracket": ["25-34"] * 6,
                "region": ["North America"] * 6,
                "share_propensity": np.linspace(0.1, 0.9, 6),
                "affinity_ml_paper": np.linspace(0.9, 0.1, 6),
                "affinity_sports_clip": np.linspace(0.1, 0.9, 6),
                "age_eligible": [True] * 6,
                "platform_fit": np.linspace(0.1, 0.9, 6),
                "sampling_weight": [1 / 6] * 6,
            }
        )
        _, _, random_diagnostics = persona_ic.assign_personas_to_nodes(
            graph, signals, 0.0, np.random.default_rng(4)
        )
        _, _, coupled_diagnostics = persona_ic.assign_personas_to_nodes(
            graph, signals, 0.9, np.random.default_rng(4)
        )
        self.assertGreater(
            coupled_diagnostics["realized_spearman_activity_platform_fit"],
            random_diagnostics["realized_spearman_activity_platform_fit"],
        )

    def test_conditional_ic_is_reproducible_for_fixed_rng(self):
        graph = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
        assigned = pd.DataFrame(
            {
                "share_propensity": [0.8] * 4,
                "affinity_ml_paper": [0.9] * 4,
                "affinity_sports_clip": [0.1] * 4,
            }
        )
        first = persona_ic.conditional_ic(
            graph,
            [0],
            list(graph.nodes()),
            assigned,
            persona_ic.TOPICS["ml_paper"],
            n_sims=30,
            rng=np.random.default_rng(9),
        )
        second = persona_ic.conditional_ic(
            graph,
            [0],
            list(graph.nodes()),
            assigned,
            persona_ic.TOPICS["ml_paper"],
            n_sims=30,
            rng=np.random.default_rng(9),
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
