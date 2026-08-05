import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "metal_instruction_input_residency_eval.py"
SPEC = importlib.util.spec_from_file_location(
    "metal_instruction_input_residency_eval", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
residency = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(residency)


def arm_result(
    arm: str,
    seed: int,
    member_ns: int,
    first_three_ns: int,
    gpu_wall_ns: int,
    gpu_active_ns: int,
) -> dict[str, object]:
    first = [first_three_ns // 3, first_three_ns // 3]
    first.append(first_three_ns - sum(first))
    later_total = gpu_wall_ns - first_three_ns
    later = [later_total // 8] * 7
    later.append(later_total - sum(later))
    command_wall = first + later
    active = [gpu_active_ns // len(command_wall)] * (len(command_wall) - 1)
    active.append(gpu_active_ns - sum(active))
    assert all(active_ns <= wall_ns for active_ns, wall_ns in zip(active, command_wall))
    initialization_bytes = residency.INITIALIZATION_BYTES[arm]
    timings = {
        "cpu_control_ns": 700_000_000,
        "sequence_preparation_ns": 300_000_000,
        "storage_initialization_wall_ns": 20_000_000,
        "storage_initialization_gpu_active_ns": 15_000_000,
        "member_wall_ns": member_ns,
        "gpu_dispatch_wall_ns": gpu_wall_ns,
        "gpu_active_ns": gpu_active_ns,
        "host_round_ns": 1_000_000,
        "readback_ns": 1_000_000,
        "cpu_tail_ns": 3_000_000,
        "first_three_gpu_command_wall_ns": first_three_ns,
        "first_three_gpu_command_active_ns": sum(active[:3]),
        "later_gpu_command_wall_ns": later_total,
        "later_gpu_command_active_ns": sum(active[3:]),
        "gpu_command_wall_ns": command_wall,
        "gpu_command_active_ns": active,
    }
    return {
        "schema": residency.SCHEMA,
        "schema_version": residency.SCHEMA_VERSION,
        "kernel": "instruction_input",
        "arm": arm,
        "metrics": {},
        "timings": timings,
        "guards": {"exact": True},
        "all_exact": True,
        "resources": {
            "sequence_owned_storage_bytes": residency.INITIALIZATION_BYTES["full"],
            "storage_initialization_bytes": initialization_bytes,
            "storage_initialization_device_buffers": 6,
            "storage_buffer_identities": [1, 2, 3, 4, 5, 6],
            "cutoff_readback_bytes": 8 * (1 << residency.CUTOFF_LOG2) * 16,
            "persistent_device_buffers": 6,
            "round_device_buffer_allocations": 0,
        },
        "workload": {
            "log_n": residency.LOG_N,
            "cutoff_log2": residency.CUTOFF_LOG2,
            "target_sequences": 1,
            "excluded_target_warmups": 0,
            "storage_initialization_outside_member_timer": True,
        },
        "fingerprint": {
            "seed": seed,
            "log_n": residency.LOG_N,
            "cutoff_log2": residency.CUTOFF_LOG2,
            "storage_initialization": arm,
            "process_model": "one_cold_target_sequence_per_process",
        },
    }


class ResidencyEvaluatorTests(unittest.TestCase):
    def test_closed_result_validator_accepts_exact_arm(self) -> None:
        result = arm_result("full", 7, 120_000_000, 70_000_000, 100_000_000, 20_000_000)
        residency.validate_result(result, "full", 7)

    def test_closed_result_validator_rejects_wrong_touch_bytes(self) -> None:
        result = arm_result("minimal", 7, 380_000_000, 300_000_000, 350_000_000, 20_000_000)
        result["resources"]["storage_initialization_bytes"] = 95
        with self.assertRaisesRegex(ValueError, "resource accounting"):
            residency.validate_result(result, "minimal", 7)

    def test_full_touch_wins_only_when_every_mechanism_gate_clears(self) -> None:
        pairs = []
        for index in range(3):
            seed = index + 1
            pairs.append(
                {
                    "minimal": arm_result(
                        "minimal", seed, 380_000_000, 300_000_000, 350_000_000, 20_000_000
                    ),
                    "full": arm_result(
                        "full", seed, 120_000_000, 70_000_000, 100_000_000, 20_000_000
                    ),
                }
            )
        summary = residency.summarize(pairs)
        self.assertEqual(summary["decision"], "full")
        self.assertTrue(all(summary["full_criteria"].values()))

    def test_small_full_touch_gain_selects_minimal_when_both_are_fast(self) -> None:
        pairs = []
        for index in range(3):
            seed = index + 1
            pairs.append(
                {
                    "minimal": arm_result(
                        "minimal", seed, 140_000_000, 90_000_000, 120_000_000, 20_000_000
                    ),
                    "full": arm_result(
                        "full", seed, 138_000_000, 88_000_000, 118_000_000, 20_000_000
                    ),
                }
            )
        self.assertEqual(residency.summarize(pairs)["decision"], "minimal")


if __name__ == "__main__":
    unittest.main()
