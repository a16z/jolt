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
        validate_replication(replication(4), "proxy")


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
    def test_piop_evaluator_uses_the_v2_kernel_floors(self) -> None:
        from scripts import metal_piop_eval

        self.assertEqual(metal_piop_eval.BYTECODE_MIN_SPEEDUP, 5.0)
        self.assertEqual(metal_piop_eval.INSTRUCTION_INPUT_MIN_SPEEDUP, 5.0)
        self.assertEqual(metal_piop_eval.BOOLEANITY_ADDRESS_MIN_SPEEDUP, 5.0)
        self.assertEqual(metal_piop_eval.HAMMING_WEIGHT_MIN_SPEEDUP, 5.0)
        self.assertEqual(metal_piop_eval.OUTER_REMAINDER_MIN_SPEEDUP, 5.0)

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
        tampered["evaluation"]["tiers"][0]["evaluator"] = "malformed"
        with self.assertRaisesRegex(ValueError, "evaluator is invalid"):
            validate_template(tampered, ROOT)

    def test_runtime_artifact_contract_closes_source_plan_and_env(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        plan_parameter = "JOLT_METAL_OUTER_REMAINDER_BINDING_PLAN"
        plans = ["b_only_v1", "split_ab_v1"]
        template["search_space"][plan_parameter] = plans
        template["baseline_params"][plan_parameter] = "b_only_v1"
        template["runtime_artifact"] = {
            "kind": "outer_msl_v1",
            "source_path": (
                "crates/jolt-kernels/src/metal/solinas/outer_remainder/"
                "shader.metal"
            ),
            "plan_parameter": plan_parameter,
            "plans": plans,
            "tier_id": "screen",
        }
        source_path = template["runtime_artifact"]["source_path"]
        host_paths = set(template["scope"]["editable"]) - {source_path}
        template["scope"]["editable"] = [source_path]
        template["scope"]["frozen"] = sorted(
            set(template["scope"]["frozen"]) | host_paths
        )
        screen = next(
            tier
            for tier in template["evaluation"]["tiers"]
            if tier["id"] == "screen"
        )
        screen["evaluator"][
            "result_adapter"
        ] = "outer_remainder_successor_v1"
        screen["promotion"].update(
            {
                "kind": "successor_screen",
                "clear_loss_ratio": 0.98,
                "minimum_uncertainty": 0.01,
                "maximum_calibration_relative_mad": 0.02,
            }
        )

        validate_template(template, ROOT)

        tampered = copy.deepcopy(template)
        tampered["runtime_artifact"]["plans"] = ["arbitrary_host_plan"]
        with self.assertRaisesRegex(ValueError, "source or plans"):
            validate_template(tampered, ROOT)

        tampered = copy.deepcopy(template)
        host_path = sorted(host_paths)[0]
        tampered["scope"]["editable"].append(host_path)
        tampered["scope"]["frozen"].remove(host_path)
        with self.assertRaisesRegex(ValueError, "source or plans"):
            validate_template(tampered, ROOT)

        tampered = copy.deepcopy(template)
        screen = next(
            tier
            for tier in tampered["evaluation"]["tiers"]
            if tier["id"] == "screen"
        )
        screen["evaluator"]["env"][
            "JOLT_AUTORESEARCH_PARENT_ARTIFACT"
        ] = "/forged"
        with self.assertRaisesRegex(ValueError, "evaluator is invalid"):
            validate_template(tampered, ROOT)

        tampered = copy.deepcopy(template)
        tampered["search_space"]["JOLT_AUTORESEARCH_FORGED"] = ["value"]
        tampered["baseline_params"]["JOLT_AUTORESEARCH_FORGED"] = "value"
        with self.assertRaisesRegex(ValueError, "baseline parameters"):
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

    def test_repository_template_protects_one_validation_retry(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        reserves = {
            reserve["id"]: reserve for reserve in template["budget"]["reserves"]
        }
        tiers = {
            tier["role"]: tier
            for tier in template["evaluation"]["tiers"]
            if tier.get("applicable") is True
        }
        tier_by_reserve = {
            "representative_revalidation": tiers["representative"],
            "piop_holdout": tiers["holdout"],
            "piop_transfer": tiers["transfer"],
        }

        for reserve_id, tier in tier_by_reserve.items():
            reserve = reserves[reserve_id]
            self.assertGreaterEqual(reserve["invocations"], 2)
            for resource, cost_limit in tier["cost_limit"].items():
                self.assertGreaterEqual(
                    reserve["resources"][resource],
                    reserve["invocations"] * cost_limit,
                )

    def test_kernel_validation_separates_local_and_portfolio_floors(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        for tier in template["evaluation"]["tiers"]:
            if tier.get("role") not in {"holdout", "transfer"}:
                continue
            promotion = tier["promotion"]
            self.assertGreaterEqual(promotion["minimum_local_speedup"], 5.0)
            self.assertLess(promotion["minimum_portfolio_speedup"], 5.0)

    def test_outer_template_enforces_the_calibrated_latency_bar(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        tiers = {
            tier["role"]: tier
            for tier in template["evaluation"]["tiers"]
            if tier.get("applicable") is True
        }

        self.assertEqual(
            tiers["representative"]["promotion"]["maximum_treatment_ms"],
            170.5,
        )
        self.assertEqual(
            tiers["holdout"]["promotion"]["maximum_local_treatment_ms"],
            170.5,
        )
        self.assertEqual(
            tiers["transfer"]["promotion"]["maximum_local_treatment_ms"],
            341.1,
        )

    def test_outer_template_can_edit_the_pre_registered_kernel_dataflow(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        editable = set(template["scope"]["editable"])

        self.assertTrue(
            {
                "crates/jolt-kernels/src/metal/solinas/outer_remainder/plan.rs",
                "crates/jolt-kernels/src/metal/solinas/outer_remainder/sequence.rs",
                "crates/jolt-kernels/src/metal/solinas/outer_remainder/shader.metal",
                "crates/jolt-kernels/src/metal/solinas/outer_remainder/tests.rs",
            }
            <= editable
        )

    def test_outer_template_stages_a_log_25_exact_screen(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        screen = next(
            tier
            for tier in template["evaluation"]["tiers"]
            if tier["role"] == "proxy"
        )

        self.assertTrue(screen["applicable"])
        self.assertEqual(
            screen["evaluator"]["result_adapter"],
            "outer_remainder_screen_v1",
        )
        self.assertEqual(screen["promotion"]["log_n"], 25)
        self.assertEqual(screen["replication"]["included_pairs"], 3)

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
