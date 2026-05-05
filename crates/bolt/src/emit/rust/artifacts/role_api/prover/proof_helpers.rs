use super::super::StageRustApi;

pub(super) fn push_proof_helpers(
    source: &mut String,
    stages: &[StageRustApi],
    field_type: &str,
    stage_proof_type: &str,
    sumcheck_output_type: &str,
    named_eval_type: &str,
) {
    for stage in stages {
        let kernel_module = stage
            .kernel_module
            .as_deref()
            .unwrap_or(stage.module_alias.as_str());
        source.push_str(&format!(
            "pub fn {field}_proof(artifacts: &{kernel}::{artifacts_ty}<{field_type}>) -> {stage_proof_type} {{\n    {stage_proof_type} {{\n        sumchecks: artifacts.sumchecks.iter().map({field}_sumcheck).collect(),\n    }}\n}}\n\n",
            field = stage.field_name,
            kernel = kernel_module,
            artifacts_ty = stage.artifacts_type
        ));
        source.push_str(&format!(
            "fn {field}_sumcheck(output: &{kernel}::{output_ty}<{field_type}>) -> {sumcheck_output_type} {{\n    {sumcheck_output_type} {{\n        driver: output.driver,\n        point: output.point.clone(),\n        evals: output.evals.iter().map({field}_eval).collect(),\n        proof: output.proof.clone(),\n    }}\n}}\n\n",
            field = stage.field_name,
            kernel = kernel_module,
            output_ty = stage.output_type
        ));
        source.push_str(&format!(
            "fn {field}_eval(eval: &{kernel}::{eval_ty}<{field_type}>) -> {named_eval_type} {{\n    {named_eval_type} {{\n        name: eval.name,\n        oracle: eval.oracle,\n        value: eval.value,\n    }}\n}}\n\n",
            field = stage.field_name,
            kernel = kernel_module,
            eval_ty = stage.eval_type
        ));
    }
}
