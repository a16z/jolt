use std::collections::BTreeMap;

use super::Stage1CpuProgram;
use crate::emit::rust::EmitError;
use crate::ir::Role;
use crate::protocols::jolt::emit::rust::checks::{
    require_supported_symbol_for_emitter, symbols, verify_count,
};

impl Stage1CpuProgram {
    pub(super) fn verify_supported_target(&self) -> Result<(), EmitError> {
        require_supported_symbol_for_emitter("stage1", "field", &self.params.field, "bn254_fr")?;
        require_supported_symbol_for_emitter("stage1", "pcs", &self.params.pcs, "dory")?;
        require_supported_symbol_for_emitter(
            "stage1",
            "transcript",
            &self.params.transcript,
            "blake2b_transcript",
        )?;
        self.verify_transcript_squeezes()?;
        self.verify_claim_batches()?;
        match self.role {
            Role::Prover => {
                self.verify_kernel_definitions()?;
                self.verify_prover_driver_bindings()?;
            }
            Role::Verifier => self.verify_verifier_driver_bindings()?,
        }
        self.verify_opening_flow()
    }

    fn verify_transcript_squeezes(&self) -> Result<(), EmitError> {
        for squeeze in &self.transcript_squeezes {
            if squeeze.kind != "challenge_vector" {
                return Err(EmitError::new(format!(
                    "stage1 transcript squeeze @{} has unsupported kind `{}`",
                    squeeze.symbol, squeeze.kind
                )));
            }
            if squeeze.count == 0 {
                return Err(EmitError::new(format!(
                    "stage1 transcript squeeze @{} has zero count",
                    squeeze.symbol
                )));
            }
        }
        Ok(())
    }

    fn verify_kernel_definitions(&self) -> Result<(), EmitError> {
        for kernel in &self.kernels {
            if kernel.backend != "cpu" {
                return Err(EmitError::new(format!(
                    "stage1 kernel @{} targets unsupported backend `{}`",
                    kernel.symbol, kernel.backend
                )));
            }
            if kernel.kind != "sumcheck" {
                return Err(EmitError::new(format!(
                    "stage1 kernel @{} has unsupported kind `{}`",
                    kernel.symbol, kernel.kind
                )));
            }
            let expected_abi = match kernel.relation.as_str() {
                "jolt.stage1.outer.uniskip" => "jolt_stage1_outer_uniskip",
                "jolt.stage1.outer.remaining" => "jolt_stage1_outer_remaining",
                _ => {
                    return Err(EmitError::new(format!(
                        "unsupported stage1 kernel relation @{}",
                        kernel.relation
                    )));
                }
            };
            if kernel.abi != expected_abi {
                return Err(EmitError::new(format!(
                    "stage1 kernel @{} ABI `{}` does not match relation @{}",
                    kernel.symbol, kernel.abi, kernel.relation
                )));
            }
        }
        Ok(())
    }

    fn verify_claim_batches(&self) -> Result<(), EmitError> {
        let claims = symbols(self.claims.iter().map(|claim| &claim.symbol));
        for batch in &self.batches {
            verify_count(
                "sumcheck batch",
                &batch.symbol,
                batch.count,
                batch.ordered_claims.len(),
            )?;
            verify_count(
                "sumcheck batch operands",
                &batch.symbol,
                batch.count,
                batch.claim_operands.len(),
            )?;
            if batch.ordered_claims != batch.claim_operands {
                return Err(EmitError::new(format!(
                    "sumcheck batch @{} operand order does not match ordered_claims",
                    batch.symbol
                )));
            }
            for claim in &batch.ordered_claims {
                if !claims.contains(claim) {
                    return Err(EmitError::new(format!(
                        "sumcheck batch @{} references missing claim @{claim}",
                        batch.symbol
                    )));
                }
            }
        }
        Ok(())
    }

    fn verify_prover_driver_bindings(&self) -> Result<(), EmitError> {
        let kernels = symbols(self.kernels.iter().map(|kernel| &kernel.symbol));
        let batches: BTreeMap<_, _> = self
            .batches
            .iter()
            .map(|batch| (batch.symbol.as_str(), batch))
            .collect();
        let claims: BTreeMap<_, _> = self
            .claims
            .iter()
            .map(|claim| (claim.symbol.as_str(), claim))
            .collect();
        for claim in &self.claims {
            let Some(kernel) = claim.kernel.as_deref() else {
                return Err(EmitError::new(format!(
                    "prover sumcheck claim @{} is missing kernel",
                    claim.symbol
                )));
            };
            if !kernels.contains(kernel) {
                return Err(EmitError::new(format!(
                    "sumcheck claim @{} references missing kernel @{kernel}",
                    claim.symbol
                )));
            }
        }
        for driver in &self.drivers {
            let Some(kernel) = driver.kernel.as_deref() else {
                return Err(EmitError::new(format!(
                    "prover sumcheck driver @{} is missing kernel",
                    driver.symbol
                )));
            };
            if !kernels.contains(kernel) {
                return Err(EmitError::new(format!(
                    "sumcheck driver @{} references missing kernel @{kernel}",
                    driver.symbol
                )));
            }
            let batch = batches.get(driver.batch.as_str()).ok_or_else(|| {
                EmitError::new(format!(
                    "sumcheck driver @{} references missing batch @{}",
                    driver.symbol, driver.batch
                ))
            })?;
            verify_count(
                "sumcheck driver round_schedule",
                &driver.symbol,
                driver.num_rounds,
                driver.round_schedule.iter().sum(),
            )?;
            if driver.round_schedule != batch.round_schedule {
                return Err(EmitError::new(format!(
                    "sumcheck driver @{} round_schedule differs from batch @{}",
                    driver.symbol, batch.symbol
                )));
            }
            for claim in &batch.ordered_claims {
                let claim = claims.get(claim.as_str()).ok_or_else(|| {
                    EmitError::new(format!(
                        "sumcheck driver @{} references missing claim @{claim}",
                        driver.symbol
                    ))
                })?;
                if claim.kernel.as_deref() != Some(kernel) {
                    return Err(EmitError::new(format!(
                        "sumcheck driver @{} kernel @{kernel} differs from claim @{} kernel {:?}",
                        driver.symbol, claim.symbol, claim.kernel
                    )));
                }
            }
        }
        Ok(())
    }

    fn verify_verifier_driver_bindings(&self) -> Result<(), EmitError> {
        if !self.kernels.is_empty() {
            return Err(EmitError::new(
                "verifier stage1 program must not contain kernels",
            ));
        }
        let batches: BTreeMap<_, _> = self
            .batches
            .iter()
            .map(|batch| (batch.symbol.as_str(), batch))
            .collect();
        let claims: BTreeMap<_, _> = self
            .claims
            .iter()
            .map(|claim| (claim.symbol.as_str(), claim))
            .collect();
        for claim in &self.claims {
            if claim.kernel.is_some() || claim.relation.is_none() {
                return Err(EmitError::new(format!(
                    "verifier sumcheck claim @{} must carry relation and no kernel",
                    claim.symbol
                )));
            }
        }
        for driver in &self.drivers {
            let Some(relation) = driver.relation.as_deref() else {
                return Err(EmitError::new(format!(
                    "verifier sumcheck driver @{} is missing relation",
                    driver.symbol
                )));
            };
            if driver.kernel.is_some() {
                return Err(EmitError::new(format!(
                    "verifier sumcheck driver @{} must not carry kernel",
                    driver.symbol
                )));
            }
            let batch = batches.get(driver.batch.as_str()).ok_or_else(|| {
                EmitError::new(format!(
                    "sumcheck driver @{} references missing batch @{}",
                    driver.symbol, driver.batch
                ))
            })?;
            verify_count(
                "sumcheck driver round_schedule",
                &driver.symbol,
                driver.num_rounds,
                driver.round_schedule.iter().sum(),
            )?;
            if driver.round_schedule != batch.round_schedule {
                return Err(EmitError::new(format!(
                    "sumcheck driver @{} round_schedule differs from batch @{}",
                    driver.symbol, batch.symbol
                )));
            }
            for claim in &batch.ordered_claims {
                let claim = claims.get(claim.as_str()).ok_or_else(|| {
                    EmitError::new(format!(
                        "sumcheck driver @{} references missing claim @{claim}",
                        driver.symbol
                    ))
                })?;
                if claim.relation.as_deref() != Some(relation) {
                    return Err(EmitError::new(format!(
                        "sumcheck driver @{} relation @{relation} differs from claim @{} relation {:?}",
                        driver.symbol, claim.symbol, claim.relation
                    )));
                }
            }
        }
        Ok(())
    }

    fn verify_opening_flow(&self) -> Result<(), EmitError> {
        let drivers = symbols(self.drivers.iter().map(|driver| &driver.symbol));
        let instance_results = symbols(
            self.instance_results
                .iter()
                .map(|instance| &instance.symbol),
        );
        for instance in &self.instance_results {
            if !drivers.contains(&instance.source) {
                return Err(EmitError::new(format!(
                    "sumcheck instance result @{} references missing driver @{}",
                    instance.symbol, instance.source
                )));
            }
        }
        let mut point_sources = drivers.clone();
        point_sources.extend(instance_results);
        let evals = symbols(self.evals.iter().map(|eval| &eval.symbol));
        let openings = symbols(self.opening_claims.iter().map(|claim| &claim.symbol));
        for eval in &self.evals {
            if !drivers.contains(&eval.source) {
                return Err(EmitError::new(format!(
                    "sumcheck eval @{} references missing driver @{}",
                    eval.symbol, eval.source
                )));
            }
        }
        for claim in &self.opening_claims {
            if !point_sources.contains(&claim.point_source) {
                return Err(EmitError::new(format!(
                    "opening claim @{} uses missing point source @{}",
                    claim.symbol, claim.point_source
                )));
            }
            if !evals.contains(&claim.eval_source) {
                return Err(EmitError::new(format!(
                    "opening claim @{} uses missing eval source @{}",
                    claim.symbol, claim.eval_source
                )));
            }
        }
        for batch in &self.opening_batches {
            verify_count(
                "opening batch",
                &batch.symbol,
                batch.count,
                batch.ordered_claims.len(),
            )?;
            verify_count(
                "opening batch operands",
                &batch.symbol,
                batch.count,
                batch.claim_operands.len(),
            )?;
            if batch.ordered_claims != batch.claim_operands {
                return Err(EmitError::new(format!(
                    "opening batch @{} operand order does not match ordered_claims",
                    batch.symbol
                )));
            }
            for claim in &batch.ordered_claims {
                if !openings.contains(claim) {
                    return Err(EmitError::new(format!(
                        "opening batch @{} references missing opening @{claim}",
                        batch.symbol
                    )));
                }
            }
        }
        Ok(())
    }
}
