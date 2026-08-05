import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.metal_research.budget import (
    BudgetExhausted,
    admit_tier,
    charge_attempt,
    empty_usage,
    validate_budget,
)
from scripts.metal_research.contracts import (
    validate_goal_contract,
    validate_template,
)
from scripts.metal_research.paired import (
    paired_summary,
    validate_paired_result,
    validate_replication,
)


ROOT = Path(__file__).resolve().parents[2]


def replication(pairs: int = 5) -> dict[str, object]:
    return {
        "mode": "internal_paired",
        "included_pairs": pairs,
        "excluded_warmup_pairs": 1,
        "order_policy": "alternating",
        "first_order": ["control", "treatment"],
        "minimum_pairs_per_order_stratum": 2 if pairs >= 5 else 1,
        "input_policy": {
            "within_pair": "identical",
            "across_pairs": "distinct_deterministic",
        },
        "effect": "control_over_treatment",
        "aggregate": "median_of_pair_effects",
    }


def pairs() -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "input_id": f"tape-{index}",
            "order": (
                ["control", "treatment"]
                if index % 2 == 0
                else ["treatment", "control"]
            ),
            "arms": {
                "control": {"primary_ns": 400 + 20 * index},
                "treatment": {"primary_ns": 100 + 5 * index},
            },
            "effect": 4.0,
            "guards": {"exact": True},
        }
        for index in range(5)
    ]


def budget() -> dict[str, object]:
    return {
        "total": {
            "max_candidates_admitted": 12,
            "max_calendar_seconds": 1200,
            "max_active_evaluator_seconds": 1000,
            "max_exclusive_machine_seconds": 900,
            "max_gpu_active_seconds": 300,
            "max_tokens": 0,
            "max_monetary_usd": 0,
        },
        "reserves": [
            {
                "id": "representative_revalidation",
                "invocations": 1,
                "resources": {
                    "active_evaluator_seconds": 200,
                    "exclusive_machine_seconds": 200,
                    "gpu_active_seconds": 50,
                },
            },
            {
                "id": "piop_holdout",
                "invocations": 1,
                "resources": {
                    "active_evaluator_seconds": 500,
                    "exclusive_machine_seconds": 500,
                    "gpu_active_seconds": 100,
                },
            },
        ],
    }


class PairedContractTests(unittest.TestCase):
    def test_recomputes_paired_summary_and_order_strata(self) -> None:
        descriptor = replication()
        observed = paired_summary(pairs(), descriptor)

        self.assertEqual(observed["median"], 4.0)
        self.assertEqual(observed["mad"], 0.0)
        self.assertEqual(observed["control_first_median"], 4.0)
        self.assertEqual(observed["treatment_first_median"], 4.0)

        result = {
            "mode": "internal_paired",
            "warmups": [{}],
            "pairs": pairs(),
            "summary": observed,
        }
        validate_paired_result(result, descriptor)

    def test_rejects_reported_effect_or_order_drift(self) -> None:
        descriptor = replication()
        result = {
            "mode": "internal_paired",
            "warmups": [{}],
            "pairs": pairs(),
            "summary": paired_summary(pairs(), descriptor),
        }
        result["pairs"][0]["effect"] = 3.9
        with self.assertRaisesRegex(ValueError, "effect"):
            validate_paired_result(result, descriptor)

        result["pairs"] = pairs()
        result["pairs"][1]["order"] = ["control", "treatment"]
        with self.assertRaisesRegex(ValueError, "order"):
            validate_paired_result(result, descriptor)

    def test_representative_tiers_require_five_pairs(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least five"):
            validate_replication(replication(3), "representative")
        validate_replication(replication(3), "proxy")


class BudgetContractTests(unittest.TestCase):
    def test_reserves_are_protected_from_screening(self) -> None:
        contract = budget()
        validate_budget(contract)
        usage = empty_usage()
        usage["active_evaluator_seconds"] = 290
        usage["exclusive_machine_seconds"] = 190

        with self.assertRaisesRegex(BudgetExhausted, "reserved"):
            admit_tier(
                contract,
                usage,
                {
                    "active_evaluator_seconds": 20,
                    "exclusive_machine_seconds": 20,
                    "gpu_active_seconds": 0,
                },
            )

        admit_tier(
            contract,
            usage,
            {
                "active_evaluator_seconds": 200,
                "exclusive_machine_seconds": 200,
                "gpu_active_seconds": 50,
            },
            "representative_revalidation",
        )

    def test_failed_attempt_wall_time_is_charged(self) -> None:
        usage = empty_usage()
        charge_attempt(
            usage,
            {
                "outcome": "timeout",
                "controller": {
                    "queue_wait_seconds": 2.0,
                    "exclusive_lease_seconds": 11.0,
                    "subprocess_wall_seconds": 10.0,
                },
                "resources": {
                    "gpu_active_seconds": None,
                    "gpu_active_charge_seconds": 10.0,
                    "gpu_active_charge_kind": "conservative_wall_upper_bound",
                },
            },
        )

        self.assertEqual(usage["active_evaluator_seconds"], 10.0)
        self.assertEqual(usage["exclusive_machine_seconds"], 11.0)
        self.assertEqual(usage["gpu_active_seconds"], 10.0)
        self.assertEqual(usage["gpu_active_estimated_seconds"], 10.0)
        self.assertEqual(usage["gpu_active_validated_seconds"], 0.0)
        self.assertEqual(usage["failed_attempts"], 1)

    def test_spent_reserve_does_not_forbid_a_budgeted_retry(self) -> None:
        contract = budget()
        usage = empty_usage()
        charge_attempt(
            usage,
            {
                "outcome": "timeout",
                "budget_reserve": "representative_revalidation",
                "controller": {
                    "queue_wait_seconds": 0.0,
                    "exclusive_lease_seconds": 1.0,
                    "subprocess_wall_seconds": 1.0,
                },
                "resources": {
                    "gpu_active_seconds": None,
                    "gpu_active_charge_seconds": 1.0,
                },
            },
        )

        admit_tier(
            contract,
            usage,
            {
                "active_evaluator_seconds": 200,
                "exclusive_machine_seconds": 200,
                "gpu_active_seconds": 50,
            },
            "representative_revalidation",
        )
        self.assertEqual(
            usage["reserve_invocations"]["representative_revalidation"], 1
        )


class VersionedContractTests(unittest.TestCase):
    def test_repository_goal_uses_a_five_x_floor_everywhere(self) -> None:
        goal = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/piop_goal.v2.json"
            ).read_text()
        )
        validate_goal_contract(goal)

        self.assertEqual(goal["primary_metric"]["minimum_accepted_speedup"], 5.0)
        self.assertTrue(
            all(
                item["minimum_hybrid_speedup"] >= 5.0
                for item in goal["kernel_promotion"]
            )
        )
        self.assertEqual(
            goal["kernel_overrides"]["instruction_ra_virtualization"][
                "minimum_hybrid_speedup"
            ],
            7.0,
        )

    def test_template_binds_a_canonical_registry_slot(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        validate_template(template, ROOT)

        tampered = copy.deepcopy(template)
        tampered["slot_id"] = "OuterRemainder"
        with self.assertRaisesRegex(ValueError, "registry slot"):
            validate_template(tampered, ROOT)

        tampered = copy.deepcopy(template)
        transfer = next(
            tier
            for tier in tampered["evaluation"]["tiers"]
            if tier.get("role") == "transfer"
        )
        transfer["promotion"]["log_n"] = 26
        with self.assertRaisesRegex(ValueError, "transfer acceptance"):
            validate_template(tampered, ROOT)

        tampered = copy.deepcopy(template)
        transfer = next(
            tier
            for tier in tampered["evaluation"]["tiers"]
            if tier.get("role") == "transfer"
        )
        transfer["replication"]["included_pairs"] = 3
        transfer["replication"]["minimum_pairs_per_order_stratum"] = 1
        with self.assertRaisesRegex(ValueError, "transfer acceptance"):
            validate_template(tampered, ROOT)

    def test_schema_one_goal_and_template_remain_readable_by_legacy_controller(self) -> None:
        import scripts.metal_autoresearch as legacy

        legacy.validate_goal_contract(
            json.loads(
                (
                    ROOT / "crates/jolt-kernels/autoresearch/piop_goal.json"
                ).read_text()
            )
        )
        legacy.validate_template(
            json.loads(
                (
                    ROOT
                    / "crates/jolt-kernels/autoresearch/outer_remainder.template.json"
                ).read_text()
            ),
            ROOT,
        )


if __name__ == "__main__":
    unittest.main()
