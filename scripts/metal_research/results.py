from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .paired import paired_summary, validate_paired_result
from .versions import TIER_RESULT_SCHEMA, TIER_RESULT_SCHEMA_VERSION


def _input_id(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _semantic_order(
    order: Any, control_label: str, treatment_label: str
) -> list[str]:
    if order == [control_label, treatment_label]:
        return ["control", "treatment"]
    if order == [treatment_label, control_label]:
        return ["treatment", "control"]
    raise ValueError("evaluator returned a non-alternating arm order")


def _pair(
    index: int,
    input_id: str,
    order: Any,
    control_ns: Any,
    treatment_ns: Any,
    exact: bool,
    control_label: str,
    treatment_label: str,
) -> dict[str, Any]:
    if (
        isinstance(control_ns, bool)
        or not isinstance(control_ns, (int, float))
        or not math.isfinite(control_ns)
        or control_ns <= 0
        or isinstance(treatment_ns, bool)
        or not isinstance(treatment_ns, (int, float))
        or not math.isfinite(treatment_ns)
        or treatment_ns <= 0
    ):
        raise ValueError("evaluator returned invalid raw arm timing")
    return {
        "index": index,
        "input_id": input_id,
        "order": _semantic_order(order, control_label, treatment_label),
        "arms": {
            "control": {"primary_ns": control_ns},
            "treatment": {"primary_ns": treatment_ns},
        },
        "effect": float(control_ns) / float(treatment_ns),
        "guards": {"exact": exact},
    }


def _envelope(
    tier: dict[str, Any],
    kernel: str,
    contract: str,
    output: dict[str, Any],
    pairs: list[dict[str, Any]],
    warmups: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = paired_summary(pairs, tier["replication"])
    return {
        "schema": TIER_RESULT_SCHEMA,
        "schema_version": TIER_RESULT_SCHEMA_VERSION,
        "tier_id": tier["id"],
        "kernel": kernel,
        "result_contract": contract,
        "primary": {
            "name": (
                "piop_speedup"
                if tier["evaluator"]["result_adapter"] == "metal_piop_v7"
                else (
                    "successor_speedup"
                    if tier["evaluator"]["result_adapter"]
                    == "outer_remainder_successor_v1"
                    else "hybrid_speedup"
                )
            ),
            "unit": "x",
            "direction": "max",
            "value": summary["median"],
        },
        "replication": {
            "mode": "internal_paired",
            "warmups": warmups,
            "pairs": pairs,
            "summary": summary,
        },
        "guards": output.get("guards", {}),
        "telemetry": {},
        "fingerprint": output.get("fingerprint", {}),
        "payload": output,
    }


def _adapt_outer(
    tier: dict[str, Any], output: dict[str, Any], kernel: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    adapter = tier["evaluator"]["result_adapter"]
    contracts = {
        "outer_remainder_v3": ("outer_remainder_v3", 3),
        "outer_remainder_screen_v1": ("outer_remainder_screen_v1", 1),
    }
    expected_schema, expected_version = contracts[adapter]
    if (
        output.get("schema") != expected_schema
        or output.get("schema_version") != expected_version
        or output.get("kernel") != "OuterRemainder"
    ):
        raise ValueError("OuterRemainder evaluator returned the wrong contract")
    samples = output.get("samples")
    if not isinstance(samples, list):
        raise ValueError("OuterRemainder evaluator samples are missing")
    input_id = _input_id(output.get("fingerprint", {}).get("fixture"))
    exact = output.get("all_exact") is True
    pairs = []
    for index, sample in enumerate(samples):
        if sample.get("pair") != index:
            raise ValueError("OuterRemainder sample index is invalid")
        pairs.append(
            _pair(
                index,
                input_id,
                sample.get("order"),
                sample.get("optimized", {}).get("member_ns"),
                sample.get("metal", {}).get("member_ns"),
                exact,
                "optimized",
                "metal",
            )
        )
    warmups = (
        [{"payload": output.get("excluded_warmup")}]
        if tier["replication"]["excluded_warmup_pairs"] == 1
        else []
    )
    result = _envelope(tier, kernel, expected_schema, output, pairs, warmups)
    reported = output.get("metrics", {}).get("hybrid_speedup")
    if isinstance(reported, bool) or not isinstance(reported, (int, float)):
        raise ValueError("OuterRemainder primary metric is invalid")
    if not math.isclose(
        float(reported), result["primary"]["value"], rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("OuterRemainder primary metric disagrees with raw pairs")
    charge = float(output.get("resources", {}).get("gpu_seconds", 0.0))
    return result, {
        "gpu_active_seconds": None,
        "gpu_active_charge_seconds": charge,
        "gpu_active_charge_kind": "treatment_wall_upper_bound",
    }


def _adapt_outer_successor(
    tier: dict[str, Any], output: dict[str, Any], kernel: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        output.get("schema") != "outer_remainder_successor_v1"
        or output.get("schema_version") != 1
        or output.get("kernel") != "OuterRemainder"
    ):
        raise ValueError("OuterRemainder successor returned the wrong contract")
    samples = output.get("samples")
    if not isinstance(samples, list):
        raise ValueError("OuterRemainder successor samples are missing")
    input_id = _input_id(output.get("fingerprint", {}).get("fixture"))
    exact = output.get("all_exact") is True
    pairs = []
    timed_gpu_active_ns = 0.0
    for index, sample in enumerate(samples):
        if sample.get("pair") != index:
            raise ValueError("OuterRemainder successor sample index is invalid")
        parent_ns = sample.get("parent", {}).get("gpu_active_ns")
        candidate_ns = sample.get("candidate", {}).get("gpu_active_ns")
        pairs.append(
            _pair(
                index,
                input_id,
                sample.get("order"),
                parent_ns,
                candidate_ns,
                exact,
                "parent",
                "candidate",
            )
        )
        timed_gpu_active_ns += float(parent_ns) + float(candidate_ns)
    warmup = output.get("excluded_warmup")
    if not isinstance(warmup, dict):
        raise ValueError("OuterRemainder successor warmup is missing")
    warmup_parent_ns = warmup.get("parent", {}).get("gpu_active_ns")
    warmup_candidate_ns = warmup.get("candidate", {}).get("gpu_active_ns")
    _pair(
        -1,
        input_id,
        warmup.get("order"),
        warmup_parent_ns,
        warmup_candidate_ns,
        exact,
        "parent",
        "candidate",
    )
    total_gpu_active_ns = (
        timed_gpu_active_ns
        + float(warmup_parent_ns)
        + float(warmup_candidate_ns)
    )
    warmups = (
        [{"payload": warmup}]
        if tier["replication"]["excluded_warmup_pairs"] == 1
        else []
    )
    result = _envelope(
        tier,
        kernel,
        "outer_remainder_successor_v1",
        output,
        pairs,
        warmups,
    )
    reported = output.get("metrics", {}).get("successor_speedup")
    if isinstance(reported, bool) or not isinstance(reported, (int, float)):
        raise ValueError("OuterRemainder successor metric is invalid")
    if not math.isclose(
        float(reported), result["primary"]["value"], rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("OuterRemainder successor metric disagrees with raw pairs")
    reported_charge = output.get("resources", {}).get("gpu_seconds")
    if (
        isinstance(reported_charge, bool)
        or not isinstance(reported_charge, (int, float))
        or not math.isfinite(reported_charge)
        or reported_charge < 0
    ):
        raise ValueError("OuterRemainder successor GPU charge is invalid")
    if not math.isclose(
        float(reported_charge) * 1e9,
        total_gpu_active_ns,
        rel_tol=1e-12,
        abs_tol=0.5,
    ):
        raise ValueError(
            "OuterRemainder successor GPU charge disagrees with raw arms"
        )
    charge = total_gpu_active_ns / 1e9
    return result, {
        "gpu_active_seconds": charge,
        "gpu_active_charge_seconds": charge,
        "gpu_active_charge_kind": "validated",
    }


def _adapt_piop(
    tier: dict[str, Any], output: dict[str, Any], kernel: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if output.get("schema_version") != 7 or output.get("kernel") != "akita_piop":
        raise ValueError("PIOP evaluator returned the wrong contract")
    records = output.get("pairs")
    if not isinstance(records, list):
        raise ValueError("PIOP evaluator pair records are missing")
    fingerprint = output.get("fingerprint", {})
    if fingerprint.get("log_n") != tier["promotion"].get("log_n"):
        raise ValueError("PIOP evaluator ran at the wrong trace size")
    input_id = _input_id(
        {"workload": fingerprint.get("workload"), "log_n": fingerprint.get("log_n")}
    )
    guards = output.get("guards", {})
    proofs_exact = (
        isinstance(guards, dict)
        and guards.get("cpu_proofs_verified") is True
        and guards.get("metal_proofs_verified") is True
    )
    run_class = output.get("run_class")
    if run_class != {"mode": "production", "acceptance_eligible": True}:
        raise ValueError("PIOP evaluator did not run the production contract")
    expected_local = tier["promotion"].get("local_kernel")
    if output.get("local_kernel") != expected_local:
        raise ValueError("PIOP evaluator selected the wrong local kernel")
    pairs = []
    local_pairs = []
    for index, record in enumerate(records):
        if record.get("index") != index + 1:
            raise ValueError("PIOP pair index is invalid")
        arms = record.get("arms", {})
        pairs.append(
            _pair(
                index,
                input_id,
                record.get("order"),
                arms.get("optimized", {}).get("piop_ns"),
                arms.get("metal", {}).get("piop_ns"),
                proofs_exact,
                "optimized",
                "metal",
            )
        )
        optimized_local = arms.get("optimized", {}).get("local")
        metal_local = arms.get("metal", {}).get("local")
        if (
            not isinstance(optimized_local, dict)
            or not isinstance(metal_local, dict)
            or optimized_local.get("kernel") != expected_local
            or metal_local.get("kernel") != expected_local
        ):
            raise ValueError("PIOP local arm evidence is missing")
        local_pairs.append(
            _pair(
                index,
                input_id,
                record.get("order"),
                optimized_local.get("primary_ns"),
                metal_local.get("primary_ns"),
                proofs_exact,
                "optimized",
                "metal",
            )
        )
    result = _envelope(tier, kernel, "metal_piop_v7", output, pairs, [])
    reported = output.get("metrics", {}).get("piop_speedup")
    if isinstance(reported, bool) or not isinstance(reported, (int, float)):
        raise ValueError("PIOP primary metric is invalid")
    if not math.isclose(
        float(reported), result["primary"]["value"], rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("PIOP primary metric disagrees with raw pairs")
    local_summary = paired_summary(local_pairs, tier["replication"])
    local_metric_name = tier["promotion"].get("local_metric")
    reported_local = output.get("metrics", {}).get(local_metric_name)
    if (
        not isinstance(local_metric_name, str)
        or isinstance(reported_local, bool)
        or not isinstance(reported_local, (int, float))
        or not math.isclose(
            float(reported_local),
            local_summary["median"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("PIOP local metric disagrees with raw pairs")
    result["local"] = {
        "kernel": expected_local,
        "primary": {
            "name": local_metric_name,
            "unit": "x",
            "direction": "max",
            "value": local_summary["median"],
        },
        "replication": {"pairs": local_pairs, "summary": local_summary},
    }
    charge = sum(
        float(pair["arms"]["treatment"]["primary_ns"]) for pair in pairs
    ) / 1_000_000_000.0
    reported_charge = output.get("resources", {}).get("metal_piop_seconds")
    rounding_tolerance = len(pairs) * 0.500001 / 1_000_000_000.0
    if (
        isinstance(reported_charge, bool)
        or not isinstance(reported_charge, (int, float))
        or not math.isclose(
            float(reported_charge),
            charge,
            rel_tol=1e-9,
            abs_tol=rounding_tolerance,
        )
    ):
        raise ValueError("PIOP resource charge disagrees with raw treatment arms")
    return result, {
        "gpu_active_seconds": None,
        "gpu_active_charge_seconds": charge,
        "gpu_active_charge_kind": "treatment_wall_upper_bound",
    }


def adapt_result(
    tier: dict[str, Any], output: dict[str, Any], kernel: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    adapter = tier["evaluator"]["result_adapter"]
    if adapter == "outer_remainder_successor_v1":
        return _adapt_outer_successor(tier, output, kernel)
    if adapter in {"outer_remainder_v3", "outer_remainder_screen_v1"}:
        return _adapt_outer(tier, output, kernel)
    if adapter == "metal_piop_v7":
        return _adapt_piop(tier, output, kernel)
    raise ValueError(f"unsupported evaluator result adapter: {adapter}")


def validate_tier_result(
    result: dict[str, Any], tier: dict[str, Any]
) -> dict[str, float]:
    if (
        result.get("schema") != TIER_RESULT_SCHEMA
        or result.get("schema_version") != TIER_RESULT_SCHEMA_VERSION
        or result.get("tier_id") != tier["id"]
    ):
        raise ValueError("tier result envelope is invalid")
    primary = result.get("primary")
    if (
        not isinstance(primary, dict)
        or primary.get("direction") != "max"
        or primary.get("unit") != "x"
    ):
        raise ValueError("tier primary metric is invalid")
    observed = validate_paired_result(result.get("replication", {}), tier["replication"])
    value = primary.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isclose(
            float(value), observed["median"], rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise ValueError("tier primary metric disagrees with paired evidence")
    return observed
