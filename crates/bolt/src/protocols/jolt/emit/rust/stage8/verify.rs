use std::collections::BTreeSet;

use super::{Stage8CpuProgram, EVALUATION_POINT_SOURCE_SYMBOL};
use crate::emit::rust::EmitError;
use crate::ir::Role;

impl Stage8CpuProgram {
    pub(super) fn verify_supported_target(&self) -> Result<(), EmitError> {
        if self.function != "jolt.stage8" {
            return Err(EmitError::new(format!(
                "stage8 emitter expected function `jolt.stage8`, got `{}`",
                self.function
            )));
        }
        if self.opening_batches.len() != 1 {
            return Err(EmitError::new(format!(
                "stage8 emitter expects one PCS opening batch, got {}",
                self.opening_batches.len()
            )));
        }
        if self.pcs_proofs.len() != 1 {
            return Err(EmitError::new(format!(
                "stage8 emitter expects one PCS proof op, got {}",
                self.pcs_proofs.len()
            )));
        }
        let expected_mode = match self.role {
            Role::Prover => "open",
            Role::Verifier => "verify",
        };
        if self.pcs_proofs[0].mode != expected_mode {
            return Err(EmitError::new(format!(
                "stage8 {} artifact expected PCS mode `{expected_mode}`, got `{}`",
                self.role, self.pcs_proofs[0].mode
            )));
        }
        let batch = &self.opening_batches[0];
        if batch.count != self.opening_claims.len() {
            return Err(EmitError::new(format!(
                "stage8 opening batch count {} does not match {} opening claims",
                batch.count,
                self.opening_claims.len()
            )));
        }
        if batch.ordered_claims != batch.claim_operands {
            return Err(EmitError::new(
                "stage8 opening batch ordered claims do not match SSA operands",
            ));
        }
        if !self
            .opening_inputs
            .iter()
            .any(|input| input.symbol == EVALUATION_POINT_SOURCE_SYMBOL)
        {
            return Err(EmitError::new(format!(
                "stage8 program missing `{EVALUATION_POINT_SOURCE_SYMBOL}` opening-point source"
            )));
        }
        let input_symbols = self
            .opening_inputs
            .iter()
            .map(|input| input.symbol.as_str())
            .collect::<BTreeSet<_>>();
        for claim in &self.opening_claims {
            if !input_symbols.contains(claim.point_source.as_str()) {
                return Err(EmitError::new(format!(
                    "stage8 claim `{}` point source `{}` is not an opening input",
                    claim.symbol, claim.point_source
                )));
            }
            if claim.point_source != claim.eval_source {
                return Err(EmitError::new(format!(
                    "stage8 claim `{}` must take point and eval from the same opening input",
                    claim.symbol
                )));
            }
        }
        Ok(())
    }
}
