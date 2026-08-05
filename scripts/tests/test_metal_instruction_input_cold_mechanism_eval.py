import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "metal_instruction_input_cold_mechanism_eval.py"
)
SPEC = importlib.util.spec_from_file_location(
    "metal_instruction_input_cold_mechanism_eval", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
cold = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cold)


def arm_result(
    arm: str,
    seed: int,
    member_ns: int,
    round_0_wall_ns: int,
    round_0_active_ns: int = 28_000_000,
) -> dict[str, object]:
    later_wall = [43_000_000, 15_000_000] + [2_000_000] * 8
    later_active = [42_000_000, 14_000_000] + [1_800_000] * 8
    command_wall = [round_0_wall_ns] + later_wall
    command_active = [round_0_active_ns] + later_active
    storage_mode = cold.STORAGE_MODE[arm]
    expects_control = arm in {"compute_control", "native_primer"}
    is_primer = arm == "native_primer"
    control_wall = 110_000_000 if expects_control else 0
    control_active = 100_000 if expects_control else 0
    identities = [1, 2, 3, 4, 5, 6]
    timings = {
        "cpu_control_ns": 800_000_000,
        "sequence_preparation_ns": 900_000_000,
        "storage_initialization_wall_ns": 1 if storage_mode == "lazy" else 114_000_000,
        "storage_initialization_gpu_active_ns": 0 if storage_mode == "lazy" else 10_000,
        "control_wall_ns": control_wall,
        "control_gpu_active_ns": control_active,
        "member_wall_ns": member_ns,
        "gpu_dispatch_wall_ns": sum(command_wall),
        "gpu_active_ns": sum(command_active),
        "host_round_ns": 500_000,
        "readback_ns": 1_000_000,
        "cpu_tail_ns": 1_000_000,
        "round_0_gpu_command_wall_ns": round_0_wall_ns,
        "round_0_gpu_command_active_ns": round_0_active_ns,
        "round_0_nonactive_ns": round_0_wall_ns - round_0_active_ns,
        "first_three_gpu_command_wall_ns": sum(command_wall[:3]),
        "first_three_gpu_command_active_ns": sum(command_active[:3]),
        "later_gpu_command_wall_ns": sum(command_wall[3:]),
        "later_gpu_command_active_ns": sum(command_active[3:]),
        "gpu_command_wall_ns": command_wall,
        "gpu_command_active_ns": command_active,
    }
    return {
        "schema": cold.SCHEMA,
        "schema_version": cold.SCHEMA_VERSION,
        "kernel": "instruction_input",
        "arm": arm,
        "metrics": {
            "member_wall_ns": member_ns,
            "round_0_nonactive_ns": round_0_wall_ns - round_0_active_ns,
            "control_plus_member_ns": control_wall + member_ns,
        },
        "timings": timings,
        "guards": dict.fromkeys(cold.GUARD_FIELDS, True),
        "all_exact": True,
        "resources": {
            "sequence_owned_storage_bytes": 6_443_433_984,
            "storage_initialization_bytes": cold.INITIALIZATION_BYTES[storage_mode],
            "storage_initialization_device_buffers": cold.INITIALIZATION_BUFFERS[storage_mode],
            "storage_buffer_identities": identities,
            "resident_row_identity": 7,
            "primer_source_elements": 64 if is_primer else 0,
            "primer_e_in_elements": 1 if is_primer else 0,
            "primer_e_out_elements": 32 if is_primer else 0,
            "primer_resident_row_identity": 7 if is_primer else 0,
            "primer_storage_buffer_identities": identities if is_primer else [0] * 6,
            "cutoff_readback_bytes": 8 * (1 << cold.CUTOFF_LOG2) * 16,
            "persistent_device_buffers": 6,
            "round_device_buffer_allocations": 0,
        },
        "workload": {
            "log_n": cold.LOG_N,
            "rows": 1 << cold.LOG_N,
            "cutoff_log2": cold.CUTOFF_LOG2,
            "cutoff_elements": 1 << cold.CUTOFF_LOG2,
            "tables": 8,
            "host_fiat_shamir": True,
            "target_sequences": 1,
            "excluded_target_warmups": 0,
            "cpu_control_before_sequence_preparation": True,
            "storage_initialization_outside_member_timer": True,
            "control_outside_member_timer": True,
        },
        "fingerprint": {
            "device": "Apple M4 Max",
            "max_buffer_length": 86_586_540_032,
            "recommended_max_working_set_size": 115_448_725_504,
            "cpu_threads": 16,
            "seed": seed,
            "log_n": cold.LOG_N,
            "cutoff_log2": cold.CUTOFF_LOG2,
            "native_message_threads": 256,
            "native_transition_threads": 128,
            "dense_transition_threads": 128,
            "storage_initialization": storage_mode,
            "control": arm,
            "gpu_command_count": cold.LOG_N - cold.CUTOFF_LOG2 + 1,
            "process_model": "one_cold_target_sequence_per_process",
        },
    }


def blocks(compute_member: int, compute_round_0: int, primer_member: int, primer_round_0: int):
    output = []
    for index in range(4):
        seed = index + 1
        output.append(
            {
                "lazy": arm_result("lazy", seed, 380_000_000, 228_000_000),
                "minimal": arm_result("minimal", seed, 220_000_000, 138_000_000),
                "compute_control": arm_result(
                    "compute_control", seed, compute_member, compute_round_0
                ),
                "native_primer": arm_result(
                    "native_primer", seed, primer_member, primer_round_0
                ),
            }
        )
    return output


class ColdMechanismEvaluatorTests(unittest.TestCase):
    def test_closed_result_validator_accepts_lazy_and_primer_arms(self) -> None:
        cold.validate_result(
            arm_result("lazy", 7, 380_000_000, 228_000_000), "lazy", 7
        )
        cold.validate_result(
            arm_result("native_primer", 7, 112_000_000, 33_000_000),
            "native_primer",
            7,
        )

    def test_validator_rejects_a_primer_on_a_different_row_allocation(self) -> None:
        result = arm_result("native_primer", 7, 112_000_000, 33_000_000)
        result["resources"]["primer_resident_row_identity"] = 8
        with self.assertRaisesRegex(ValueError, "primer geometry"):
            cold.validate_result(result, "native_primer", 7)

    def test_validator_rejects_any_timing_on_an_inactive_control(self) -> None:
        for field in ("control_wall_ns", "control_gpu_active_ns"):
            with self.subTest(field=field):
                result = arm_result("minimal", 7, 220_000_000, 138_000_000)
                result["timings"][field] = 1
                if field == "control_wall_ns":
                    result["metrics"]["control_plus_member_ns"] += 1
                with self.assertRaisesRegex(ValueError, "inactive control"):
                    cold.validate_result(result, "minimal", 7)

    def test_validator_rejects_nested_schema_and_alias_drift(self) -> None:
        result = arm_result("minimal", 7, 220_000_000, 138_000_000)
        result["guards"].pop("exact_final_relation")
        with self.assertRaisesRegex(ValueError, "guard"):
            cold.validate_result(result, "minimal", 7)

        result = arm_result("minimal", 7, 220_000_000, 138_000_000)
        result["metrics"]["member_wall_ns"] += 1
        with self.assertRaisesRegex(ValueError, "aliases"):
            cold.validate_result(result, "minimal", 7)

        result = arm_result("minimal", 7, 220_000_000, 138_000_000)
        result["workload"]["rows"] //= 2
        with self.assertRaisesRegex(ValueError, "workload"):
            cold.validate_result(result, "minimal", 7)

    def test_exact_native_primer_is_selected_when_generic_compute_is_inactive(self) -> None:
        summary = cold.summarize(
            blocks(218_000_000, 136_000_000, 112_000_000, 33_000_000)
        )
        self.assertEqual(summary["decision"], "native_pipeline_or_row_binding")
        self.assertTrue(summary["minimal_storage_selected"])
        self.assertTrue(summary["native_primer"]["clears"])
        self.assertFalse(summary["compute_control"]["clears"])

    def test_equal_compute_and_native_effects_select_general_startup(self) -> None:
        summary = cold.summarize(
            blocks(110_000_000, 33_000_000, 112_000_000, 34_000_000)
        )
        self.assertEqual(summary["decision"], "general_compute_startup")
        self.assertTrue(summary["compute_control"]["clears"])
        self.assertTrue(summary["native_primer"]["clears"])

    def test_crossed_member_and_wait_effects_do_not_clear(self) -> None:
        samples = blocks(218_000_000, 136_000_000, 112_000_000, 33_000_000)
        primer_members = [80_000_000, 80_000_000, 140_000_000, 140_000_000]
        primer_waits = [120_000_000, 120_000_000, 60_000_000, 60_000_000]
        for sample, member, wait in zip(samples, primer_members, primer_waits):
            sample["minimal"]["timings"]["round_0_nonactive_ns"] = 200_000_000
            sample["native_primer"]["timings"]["member_wall_ns"] = member
            sample["native_primer"]["timings"]["round_0_nonactive_ns"] = wait
        comparison = cold.contrast(samples, "minimal", "native_primer")
        self.assertFalse(comparison["clears"])
        self.assertFalse(
            comparison["criteria"][
                "three_of_four_member_and_wait_reductions_align_within_20_ms"
            ]
        )

    def test_bimodal_incremental_effect_is_not_inactive(self) -> None:
        comparison = {
            "paired_member_effect_ns": [30_000_001, -30_000_001] * 2,
            "paired_member_ratios": [0.7, 1.3] * 2,
        }
        self.assertFalse(cold.inactive(comparison))

    def test_storage_cannot_be_selected_without_the_raw_phenomenon(self) -> None:
        samples = blocks(105_000_000, 33_000_000, 102_000_000, 32_000_000)
        for sample in samples:
            sample["lazy"]["timings"]["member_wall_ns"] = 170_000_000
            sample["minimal"]["timings"]["member_wall_ns"] = 110_000_000
        summary = cold.summarize(samples)
        self.assertFalse(summary["phenomenon_reproduced"])
        self.assertTrue(all(summary["minimal_storage_criteria"].values()))
        self.assertFalse(summary["minimal_storage_selected"])

    def test_cyclic_orders_are_latin(self) -> None:
        orders = [
            list(cold.ARMS[index:] + cold.ARMS[:index])
            for index in range(cold.BLOCKS)
        ]
        for arm in cold.ARMS:
            self.assertEqual(
                sorted(order.index(arm) for order in orders),
                list(range(len(cold.ARMS))),
            )


if __name__ == "__main__":
    unittest.main()
