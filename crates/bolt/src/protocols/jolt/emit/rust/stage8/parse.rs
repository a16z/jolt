use std::collections::BTreeMap;

use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;

use super::{
    Stage8CpuProgram, Stage8OpeningBatchPlan, Stage8OpeningClaimPlan, Stage8OpeningInputPlan,
    Stage8Params, Stage8PcsProofPlan,
};
use crate::emit::rust::EmitError;
use crate::ir::{BoltModule, Cpu};
use crate::protocols::jolt::emit::rust::mlir::{
    i64_int_attr as int_attr, operand_symbol, operand_symbols, operation_name, string_attr,
    symbol_array_attr, symbol_reference_attr as symbol_attr,
};

impl Stage8CpuProgram {
    pub(super) fn from_module(module: &BoltModule<'_, Cpu>) -> Result<Self, EmitError> {
        let role = module
            .role()
            .ok_or_else(|| EmitError::new("stage8 CPU module missing role"))?;
        let mut params = None;
        let mut function = None;
        let mut opening_inputs = Vec::new();
        let mut opening_claims = Vec::new();
        let mut opening_batches = Vec::new();
        let mut pcs_proofs = Vec::new();

        let mut operation = module.as_mlir_module().body().first_operation();
        while let Some(op) = operation {
            operation = op.next_in_block();
            match operation_name(op).as_str() {
                "cpu.params" => {
                    params = Some(Stage8Params {
                        field: symbol_attr(op, "field")?,
                        pcs: symbol_attr(op, "pcs")?,
                        transcript: symbol_attr(op, "transcript")?,
                    });
                }
                "cpu.function" => {
                    function = Some(string_attr(op, "sym_name")?);
                }
                "cpu.opening_input" => {
                    opening_inputs.push(Stage8OpeningInputPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source_stage: symbol_attr(op, "source_stage")?,
                        source_claim: symbol_attr(op, "source_claim")?,
                        oracle: symbol_attr(op, "oracle")?,
                        domain: symbol_attr(op, "domain")?,
                        point_arity: int_attr(op, "point_arity")?,
                        claim_kind: string_attr(op, "claim_kind")?,
                    });
                }
                "cpu.pcs_opening_claim" => {
                    opening_claims.push(Stage8OpeningClaimPlan {
                        symbol: string_attr(op, "sym_name")?,
                        oracle: symbol_attr(op, "oracle")?,
                        family: symbol_attr(op, "family")?,
                        domain: symbol_attr(op, "domain")?,
                        point_arity: int_attr(op, "point_arity")?,
                        point_source: operand_symbol(op, 0)?,
                        eval_source: operand_symbol(op, 1)?,
                        source_stage: String::new(),
                        source_claim: String::new(),
                    });
                }
                "cpu.pcs_opening_batch" => {
                    opening_batches.push(Stage8OpeningBatchPlan {
                        symbol: string_attr(op, "sym_name")?,
                        proof_slot: symbol_attr(op, "proof_slot")?,
                        policy: string_attr(op, "policy")?,
                        count: int_attr(op, "count")?,
                        ordered_claims: symbol_array_attr(op, "ordered_claims")?,
                        claim_operands: operand_symbols(op, 0)?,
                    });
                }
                "cpu.pcs_batch_open" | "cpu.pcs_batch_verify" => {
                    let mode = match operation_name(op).as_str() {
                        "cpu.pcs_batch_open" => "open",
                        "cpu.pcs_batch_verify" => "verify",
                        _ => unreachable!(),
                    };
                    pcs_proofs.push(Stage8PcsProofPlan {
                        symbol: string_attr(op, "sym_name")?,
                        mode: mode.to_owned(),
                        pcs: symbol_attr(op, "pcs")?,
                        proof_slot: symbol_attr(op, "proof_slot")?,
                        transcript_label: string_attr(op, "transcript_label")?,
                        batch: operand_symbol(op, 1)?,
                    });
                }
                _ => {}
            }
        }

        let input_by_symbol = opening_inputs
            .iter()
            .map(|input| (input.symbol.as_str(), input))
            .collect::<BTreeMap<_, _>>();
        for claim in &mut opening_claims {
            let input = input_by_symbol
                .get(claim.point_source.as_str())
                .ok_or_else(|| {
                    EmitError::new(format!(
                        "stage8 opening claim `{}` references missing point source `{}`",
                        claim.symbol, claim.point_source
                    ))
                })?;
            claim.source_stage = input.source_stage.clone();
            claim.source_claim = input.source_claim.clone();
        }

        Ok(Self {
            role,
            params: params.ok_or_else(|| EmitError::new("stage8 program missing cpu.params"))?,
            function: function
                .ok_or_else(|| EmitError::new("stage8 program missing cpu.function"))?,
            opening_inputs,
            opening_claims,
            opening_batches,
            pcs_proofs,
        })
    }
}
