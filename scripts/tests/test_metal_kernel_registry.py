import copy
import json
import unittest
from pathlib import Path

from scripts import metal_kernel_registry


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "crates/jolt-kernels/src/metal/kernel_registry.json"


class MetalKernelRegistryTests(unittest.TestCase):
    def test_repository_registry_is_complete(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        metal_kernel_registry.validate_registry(ROOT, registry)

    def test_duplicate_module_id_is_rejected(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        registry["components"].append(copy.deepcopy(registry["components"][0]))

        with self.assertRaisesRegex(ValueError, "duplicate component"):
            metal_kernel_registry.validate_registry(ROOT, registry)

    def test_missing_artifact_is_rejected(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        registry["artifacts"]["documents"][0]["path"] = "missing.md"

        with self.assertRaisesRegex(ValueError, "missing artifact"):
            metal_kernel_registry.validate_registry(ROOT, registry)

    def test_entry_point_has_one_owner(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        sources = [source for source in registry["sources"] if source["entry_points"]]
        entry_point = sources[0]["entry_points"][0]
        sources[1]["entry_points"].append(entry_point)

        with self.assertRaisesRegex(ValueError, "duplicate entry point"):
            metal_kernel_registry.validate_registry(ROOT, registry)

    def test_component_includes_the_source_that_owns_each_pipeline(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        component = next(
            component
            for component in registry["components"]
            if component["id"] == "address_phase_sequence"
        )
        component["source_ids"].remove("product5")

        with self.assertRaisesRegex(ValueError, "does not include source product5"):
            metal_kernel_registry.validate_registry(ROOT, registry)

    def test_template_is_owned_by_its_declared_slot(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        template = next(
            artifact
            for artifact in registry["artifacts"]["templates"]
            if artifact["id"] == "outer_remainder_search"
        )
        template["slot_id"] = "instruction_input"

        with self.assertRaisesRegex(ValueError, "template ownership"):
            metal_kernel_registry.validate_registry(ROOT, registry)

    def test_template_binding_resolves_canonical_slot_and_digest(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        binding = metal_kernel_registry.resolve_template_binding(
            ROOT,
            registry,
            ROOT
            / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
        )

        self.assertEqual(binding["artifact_id"], "outer_remainder_search_v2")
        self.assertEqual(binding["slot_id"], "spartan_outer_remainder")
        self.assertEqual(binding["contract_schema"], 2)
        self.assertEqual(binding["lifecycle"], "fresh_init")
        self.assertTrue(binding["fresh_init_eligible"])
        self.assertEqual(len(binding["registry_sha256"]), 64)
        self.assertEqual(
            binding["registry_sha256"],
            metal_kernel_registry.sha256(
                json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
            ),
        )

    def test_template_lifecycle_matches_its_contract_schema(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        template = next(
            artifact
            for artifact in registry["artifacts"]["templates"]
            if artifact["id"] == "outer_remainder_search"
        )
        template["lifecycle"] = "fresh_init"

        with self.assertRaisesRegex(ValueError, "lifecycle"):
            metal_kernel_registry.validate_registry(ROOT, registry)

    def test_slot_has_at_most_one_fresh_template(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        template = copy.deepcopy(
            next(
                artifact
                for artifact in registry["artifacts"]["templates"]
                if artifact["id"] == "outer_remainder_search_v2"
            )
        )
        template["id"] = "outer_remainder_second_fresh"
        registry["artifacts"]["templates"].append(template)
        slot = next(
            slot
            for slot in registry["slots"]
            if slot["id"] == "spartan_outer_remainder"
        )
        slot["artifacts"]["templates"].append(template["id"])

        with self.assertRaisesRegex(ValueError, "at most one fresh-init"):
            metal_kernel_registry.validate_registry(ROOT, registry)

    def test_registry_requires_a_fresh_template(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        registry["artifacts"]["templates"] = [
            artifact
            for artifact in registry["artifacts"]["templates"]
            if artifact["id"] != "outer_remainder_search_v2"
        ]
        slot = next(
            slot
            for slot in registry["slots"]
            if slot["id"] == "spartan_outer_remainder"
        )
        slot["artifacts"]["templates"].remove("outer_remainder_search_v2")

        with self.assertRaisesRegex(ValueError, "must contain a fresh-init"):
            metal_kernel_registry.validate_registry(ROOT, registry)

    def test_library_source_order_is_bound_to_the_fragment_manifest(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        source_order = registry["library"]["source_order"]
        source_order[2], source_order[3] = source_order[3], source_order[2]

        with self.assertRaisesRegex(ValueError, "fragment manifest"):
            metal_kernel_registry.validate_registry(ROOT, registry)

    def test_library_source_path_must_own_the_fragment_manifest(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        registry["library"]["source_path"] = registry["library"]["facade_path"]

        with self.assertRaisesRegex(ValueError, "fragment manifest"):
            metal_kernel_registry.validate_registry(ROOT, registry)

    def test_source_dependencies_are_known_and_acyclic(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        source = next(
            source
            for source in registry["sources"]
            if source["id"] == "spartan_outer_common"
        )
        source["requires"] = ["missing"]

        with self.assertRaisesRegex(ValueError, "invalid dependency"):
            metal_kernel_registry.validate_registry(ROOT, registry)

    def test_library_source_order_must_be_topological(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        order = registry["library"]["source_order"]
        common = order.index("spartan_outer_common")
        field = order.index("fp128")
        order[common], order[field] = order[field], order[common]

        with self.assertRaisesRegex(ValueError, "not topological"):
            metal_kernel_registry.validate_registry(ROOT, registry)

    def test_source_role_is_closed(self) -> None:
        registry = metal_kernel_registry.read_registry(REGISTRY)
        registry["sources"][0]["role"] = "misc"

        with self.assertRaisesRegex(ValueError, "source role"):
            metal_kernel_registry.validate_registry(ROOT, registry)


if __name__ == "__main__":
    unittest.main()
