use std::error::Error;
use std::fmt::{self, Display, Formatter};

use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;

use crate::ir::{BoltModule, Concrete, Party, Phase, Protocol, Role};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{verify_concrete_schema, verify_party_schema, verify_protocol_schema};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyError {
    message: String,
}

impl VerifyError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for VerifyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for VerifyError {}

pub fn lower_piop_and_fiat_shamir<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
) -> Result<BoltModule<'c, Concrete>, MlirError> {
    verify_protocol_schema(module)?;
    let source = phase_copy_source(module, Concrete::NAME, None, &[]);
    let concrete = context.parse_module::<Concrete>(&source)?;
    verify_module(&concrete)?;
    verify_concrete_schema(&concrete)?;
    Ok(concrete)
}

pub fn derive_prover_role<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Concrete>,
) -> Result<BoltModule<'c, Concrete>, MlirError> {
    derive_role(context, module, Role::Prover)
}

pub fn derive_verifier_role<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Concrete>,
) -> Result<BoltModule<'c, Concrete>, MlirError> {
    derive_role(context, module, Role::Verifier)
}

pub fn project_prover_party<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Concrete>,
) -> Result<BoltModule<'c, Party>, MlirError> {
    project_party(context, module, Role::Prover)
}

pub fn project_verifier_party<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Concrete>,
) -> Result<BoltModule<'c, Party>, MlirError> {
    project_party(context, module, Role::Verifier)
}

pub fn project_party<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Concrete>,
    role: Role,
) -> Result<BoltModule<'c, Party>, MlirError> {
    verify_concrete_schema(module)?;
    require_declared_role(module, &role)?;
    let party_function = format!(
        "  \"party.function\"() {{role = \"{}\", source = @{}, sym_name = \"{}.{}\"}} : () -> ()",
        role.as_str(),
        module.name(),
        module.name(),
        role.as_str()
    );
    let source = phase_copy_source(module, Party::NAME, Some(&role), &[party_function]);
    let party = context.parse_module::<Party>(&source)?;
    verify_module(&party)?;
    verify_party_schema(&party)?;
    Ok(party)
}

pub fn verify_concrete_transcript<P>(module: &BoltModule<'_, P>) -> Result<(), VerifyError>
where
    P: Phase,
{
    let mut current_state = None;
    let mut error = None;
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        match operation_name(op).as_str() {
            "transcript.state" => {
                if current_state.is_some() {
                    error = Some(VerifyError::new("multiple transcript.state ops"));
                    break;
                }
                let Ok(result) = op.result(0) else {
                    error = Some(VerifyError::new("transcript.state requires one result"));
                    break;
                };
                current_state = Some(result.to_string());
            }
            "transcript.absorb" | "transcript.absorb_optional" => {
                let Some(expected_input) = current_state.as_deref() else {
                    error = Some(VerifyError::new(
                        "transcript absorb requires a prior transcript.state result",
                    ));
                    break;
                };
                let Ok(input) = op.operand(0) else {
                    error = Some(VerifyError::new(format!(
                        "{} requires transcript-state operand 0",
                        operation_name(op)
                    )));
                    break;
                };
                let input = input.to_string();
                if input != expected_input {
                    error = Some(VerifyError::new(format!(
                        "{} consumed transcript state {input}, expected {expected_input}",
                        operation_name(op)
                    )));
                    break;
                }
                if op.operand(1).is_err() {
                    error = Some(VerifyError::new(format!(
                        "{} requires commitment artifact operand 1",
                        operation_name(op)
                    )));
                    break;
                }
                let Ok(result) = op.result(0) else {
                    error = Some(VerifyError::new(format!(
                        "{} requires one transcript-state result",
                        operation_name(op)
                    )));
                    break;
                };
                current_state = Some(result.to_string());
            }
            "transcript.absorb_bytes" => {
                let Some(expected_input) = current_state.as_deref() else {
                    error = Some(VerifyError::new(
                        "transcript absorb_bytes requires a prior transcript.state result",
                    ));
                    break;
                };
                let Ok(input) = op.operand(0) else {
                    error = Some(VerifyError::new(
                        "transcript.absorb_bytes requires transcript-state operand 0",
                    ));
                    break;
                };
                let input = input.to_string();
                if input != expected_input {
                    error = Some(VerifyError::new(format!(
                        "transcript.absorb_bytes consumed transcript state {input}, expected {expected_input}",
                    )));
                    break;
                }
                let Ok(result) = op.result(0) else {
                    error = Some(VerifyError::new(
                        "transcript.absorb_bytes requires one transcript-state result",
                    ));
                    break;
                };
                current_state = Some(result.to_string());
            }
            "transcript.squeeze" | "piop.sumcheck" | "pcs.batch_open" | "pcs.batch_verify" => {
                let Some(expected_input) = current_state.as_deref() else {
                    error = Some(VerifyError::new(format!(
                        "{} requires a prior transcript.state result",
                        operation_name(op)
                    )));
                    break;
                };
                let Ok(input) = op.operand(0) else {
                    error = Some(VerifyError::new(format!(
                        "{} requires transcript-state operand 0",
                        operation_name(op)
                    )));
                    break;
                };
                let input = input.to_string();
                if input != expected_input {
                    error = Some(VerifyError::new(format!(
                        "{} consumed transcript state {input}, expected {expected_input}",
                        operation_name(op)
                    )));
                    break;
                }
                let Ok(result) = op.result(0) else {
                    error = Some(VerifyError::new(format!(
                        "{} requires transcript-state result 0",
                        operation_name(op)
                    )));
                    break;
                };
                current_state = Some(result.to_string());
            }
            _ => {}
        }
    }

    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn derive_role<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Concrete>,
    role: Role,
) -> Result<BoltModule<'c, Concrete>, MlirError> {
    let source = phase_copy_source(module, Concrete::NAME, Some(&role), &[]);
    context.parse_module::<Concrete>(&source)
}

fn phase_copy_source<P: Phase>(
    module: &BoltModule<'_, P>,
    target_phase: &str,
    role: Option<&Role>,
    prefix_ops: &[String],
) -> String {
    let mut source = format!(
        "module @{} attributes {{bolt.phase = \"{target_phase}\"",
        module.name()
    );
    if let Some(role) = role {
        source.push_str(", bolt.role = \"");
        source.push_str(role.as_str());
        source.push('"');
    }
    source.push_str("} {\n");
    for op in prefix_ops {
        source.push_str(op);
        source.push('\n');
    }
    append_body_text(&mut source, module);
    source.push_str("}\n");
    source
}

fn require_declared_role(module: &BoltModule<'_, Concrete>, role: &Role) -> Result<(), MlirError> {
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        if operation_name(op) != "protocol.boundary" {
            continue;
        }
        let roles = op
            .attribute("roles")
            .ok()
            .and_then(|attribute| parse_string_array(&attribute.to_string()))
            .ok_or_else(|| MlirError::Schema {
                message: "protocol.boundary requires string array attr `roles`".to_owned(),
            })?;
        if roles.iter().any(|declared| declared == role.as_str()) {
            return Ok(());
        }
        return Err(MlirError::Schema {
            message: format!("protocol.boundary does not declare role `{role}`"),
        });
    }

    Err(MlirError::Schema {
        message: "module missing required op `protocol.boundary`".to_owned(),
    })
}

fn parse_string_array(attribute: &str) -> Option<Vec<String>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| {
            item.trim()
                .strip_prefix('"')
                .and_then(|item| item.strip_suffix('"'))
                .map(ToOwned::to_owned)
        })
        .collect()
}

fn operation_name<'c: 'a, 'a>(operation: impl OperationLike<'c, 'a>) -> String {
    operation
        .name()
        .as_string_ref()
        .as_str()
        .unwrap_or("<invalid-operation-name>")
        .to_owned()
}

fn append_body_text<P: Phase>(module_source: &mut String, module: &BoltModule<'_, P>) {
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        module_source.push_str("  ");
        module_source.push_str(&op.to_string());
        module_source.push('\n');
    }
}
