from __future__ import annotations

import math
import statistics
from typing import Any


_PAIRED_FIELDS = {
    "mode",
    "included_pairs",
    "excluded_warmup_pairs",
    "order_policy",
    "first_order",
    "minimum_pairs_per_order_stratum",
    "input_policy",
    "effect",
    "aggregate",
}


def _finite_positive(value: Any, description: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{description} must be finite and positive")
    return float(value)


def validate_replication(descriptor: dict[str, Any], role: str) -> None:
    if descriptor == {"mode": "single"}:
        return
    if not isinstance(descriptor, dict) or set(descriptor) != _PAIRED_FIELDS:
        raise ValueError("internal paired replication fields are incomplete")
    if descriptor["mode"] != "internal_paired":
        raise ValueError("unsupported replication mode")

    pairs = descriptor["included_pairs"]
    minimum = 5 if role in {"representative", "holdout"} else 3
    if type(pairs) is not int or pairs < minimum or pairs % 2 == 0:
        qualifier = "five" if minimum == 5 else "three"
        raise ValueError(
            f"{role} internal paired replication requires at least {qualifier} odd pairs"
        )
    warmups = descriptor["excluded_warmup_pairs"]
    if type(warmups) is not int or warmups < 0:
        raise ValueError("excluded warmup pairs must be nonnegative")
    if descriptor["order_policy"] != "alternating":
        raise ValueError("paired evaluation order must alternate")
    first_order = descriptor["first_order"]
    if first_order not in (["control", "treatment"], ["treatment", "control"]):
        raise ValueError("first_order must contain the control and treatment arms")
    minimum_stratum = descriptor["minimum_pairs_per_order_stratum"]
    if (
        type(minimum_stratum) is not int
        or minimum_stratum < 1
        or minimum_stratum > pairs // 2
    ):
        raise ValueError("minimum order-stratum pair count is invalid")
    if descriptor["input_policy"] not in (
        {"within_pair": "identical", "across_pairs": "distinct_deterministic"},
        {"within_pair": "identical", "across_pairs": "same_fixture"},
    ):
        raise ValueError("paired input policy is invalid")
    if descriptor["effect"] != "control_over_treatment":
        raise ValueError("paired effect must be control_over_treatment")
    if descriptor["aggregate"] != "median_of_pair_effects":
        raise ValueError("paired aggregation must use the median of pair effects")


def _expected_order(descriptor: dict[str, Any], index: int) -> list[str]:
    order = list(descriptor["first_order"])
    return order if index % 2 == 0 else list(reversed(order))


def paired_summary(
    pairs: list[dict[str, Any]], descriptor: dict[str, Any]
) -> dict[str, float]:
    validate_replication(descriptor, "proxy")
    if len(pairs) != descriptor["included_pairs"]:
        raise ValueError("paired result has the wrong pair count")

    effects: list[float] = []
    strata: dict[str, list[float]] = {"control": [], "treatment": []}
    input_ids: list[str] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict) or pair.get("index") != index:
            raise ValueError("paired result index is invalid")
        input_id = pair.get("input_id")
        if not isinstance(input_id, str) or not input_id:
            raise ValueError("paired result input_id is invalid")
        input_ids.append(input_id)
        expected_order = _expected_order(descriptor, index)
        if pair.get("order") != expected_order:
            raise ValueError("paired result order is invalid")
        arms = pair.get("arms")
        if not isinstance(arms, dict) or set(arms) != {"control", "treatment"}:
            raise ValueError("paired result arms are invalid")
        try:
            control = _finite_positive(
                arms["control"]["primary_ns"], "control arm primary_ns"
            )
            treatment = _finite_positive(
                arms["treatment"]["primary_ns"], "treatment arm primary_ns"
            )
        except (KeyError, TypeError) as error:
            raise ValueError("paired result arm timing is invalid") from error
        effect = control / treatment
        reported = _finite_positive(pair.get("effect"), "paired effect")
        if not math.isclose(reported, effect, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("reported paired effect disagrees with raw arms")
        guards = pair.get("guards")
        if (
            not isinstance(guards, dict)
            or not guards
            or any(value is not True for value in guards.values())
        ):
            raise ValueError("paired result guards did not all pass")
        effects.append(effect)
        strata[expected_order[0]].append(effect)

    if descriptor["input_policy"]["across_pairs"] == "distinct_deterministic":
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("paired result inputs are not distinct")
    elif len(set(input_ids)) != 1:
        raise ValueError("same-fixture paired result changed its input")

    minimum_stratum = descriptor["minimum_pairs_per_order_stratum"]
    if any(len(values) < minimum_stratum for values in strata.values()):
        raise ValueError("paired result has too few observations in an order stratum")
    median = statistics.median(effects)
    return {
        "median": median,
        "mad": statistics.median(abs(value - median) for value in effects),
        "control_first_median": statistics.median(strata["control"]),
        "treatment_first_median": statistics.median(strata["treatment"]),
    }


def validate_paired_result(
    result: dict[str, Any], descriptor: dict[str, Any]
) -> dict[str, float]:
    if result.get("mode") != "internal_paired":
        raise ValueError("paired result mode is invalid")
    warmups = result.get("warmups")
    if (
        not isinstance(warmups, list)
        or len(warmups) != descriptor["excluded_warmup_pairs"]
    ):
        raise ValueError("paired result warmup count is invalid")
    pairs = result.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("paired result pairs are invalid")
    observed = paired_summary(pairs, descriptor)
    reported = result.get("summary")
    if not isinstance(reported, dict) or set(reported) != set(observed):
        raise ValueError("paired result summary fields are invalid")
    for name, value in observed.items():
        if (
            name == "mad"
            and not isinstance(reported[name], bool)
            and reported[name] == 0
        ):
            candidate = 0.0
        else:
            candidate = _finite_positive(reported[name], f"paired summary {name}")
        if not math.isclose(candidate, value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"paired result {name} disagrees with raw pairs")
    return observed
