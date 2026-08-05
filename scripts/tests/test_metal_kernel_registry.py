import copy
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


if __name__ == "__main__":
    unittest.main()
