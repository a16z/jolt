use super::super::super::types::ProtocolRustArtifact;

pub(super) fn generated_stage_inventory(artifacts: &[ProtocolRustArtifact]) -> String {
    artifacts
        .iter()
        .map(|artifact| {
            format!(
                "    GeneratedStage {{\n        name: \"{}\",\n        module: \"{}\",\n        ordinal: {},\n    }},",
                artifact.stage.name(),
                artifact.stage.module_name(),
                artifact.stage.order()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
