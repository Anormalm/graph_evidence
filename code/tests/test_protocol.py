import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_results import (  # noqa: E402
    answer_space_profile,
    cohen_kappa,
    conditional_faithfulness_m,
)
from groundlm_serialization import (  # noqa: E402
    DISTRACTOR_FUNCTIONS,
    TASKS,
    compute_answer_for_source_nodes,
    compute_ground_truth,
    generate_graphs,
    relabel_misleading,
)


class ProtocolTests(unittest.TestCase):
    def test_dataset_profile_and_workload(self):
        profile = answer_space_profile()
        self.assertEqual(profile["graphs"], 100)
        self.assertEqual(profile["graph_task_pairs"], 400)
        self.assertEqual(profile["query_instances"], 1200)
        self.assertEqual(profile["intended_prompts_per_model"]["total"], 16800)
        self.assertEqual(
            profile["ground_truth_counts"]["connectivity"], {"yes": 300}
        )
        self.assertEqual(
            profile["ground_truth_counts"]["shortest_path"],
            {"1": 101, "2": 143, "3": 50, "4": 4, "5": 2},
        )

    def test_current_hub_mapping_is_maximum_degree(self):
        for graph_info in generate_graphs():
            graph = graph_info["graph"]
            _, mapping = relabel_misleading(graph)
            hub_source = next(node for node, label in mapping.items() if label == "Hub")
            self.assertEqual(
                graph.degree(hub_source), max(dict(graph.degree()).values())
            )

    def test_all_distractors_preserve_exact_query_answers(self):
        for graph_info in generate_graphs():
            graph = graph_info["graph"]
            for task in TASKS:
                queries = compute_ground_truth(
                    graph, task, seed=graph_info["seed"]
                )
                for query in queries:
                    for add_distractor in DISTRACTOR_FUNCTIONS.values():
                        transformed = add_distractor(graph, seed=graph_info["seed"])
                        observed = compute_answer_for_source_nodes(
                            transformed, task, query["source_nodes"]
                        )
                        self.assertEqual(observed, query["answer"])

    def test_cohen_kappa(self):
        stat = cohen_kappa(["yes", "yes", "no", "no"], ["yes", "yes", "no", "no"])
        self.assertAlmostEqual(stat["kappa"], 1.0)
        degenerate = cohen_kappa(["yes", "yes"], ["yes", "yes"])
        self.assertIsNone(degenerate["kappa"])

    def test_conditional_faithfulness(self):
        records = [
            {
                "base_key": "a",
                "task": "connectivity",
                "ground_truth": "yes",
                "perturbation": "original",
                "model_answer": "Final answer: yes",
            },
            {
                "base_key": "a",
                "task": "connectivity",
                "ground_truth": "yes",
                "perturbation": "misleading_relabel",
                "model_answer": "Final answer: yes",
            },
            {
                "base_key": "b",
                "task": "connectivity",
                "ground_truth": "yes",
                "perturbation": "original",
                "model_answer": "Final answer: yes",
            },
            {
                "base_key": "b",
                "task": "connectivity",
                "ground_truth": "yes",
                "perturbation": "misleading_relabel",
                "model_answer": "Final answer: no",
            },
            {
                "base_key": "c",
                "task": "connectivity",
                "ground_truth": "yes",
                "perturbation": "original",
                "model_answer": "Final answer: no",
            },
            {
                "base_key": "c",
                "task": "connectivity",
                "ground_truth": "yes",
                "perturbation": "misleading_relabel",
                "model_answer": "Final answer: yes",
            },
        ]
        stat = conditional_faithfulness_m(records)
        self.assertEqual(stat["eligible"], 2)
        self.assertEqual(stat["retained"], 1)
        self.assertAlmostEqual(stat["CF_M"], 0.5)


if __name__ == "__main__":
    unittest.main()
