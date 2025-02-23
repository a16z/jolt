use crate::{
    jolt::vm::{
        bytecode::{BytecodeProof, BytecodeStuff},
        instruction_lookups::{
            InstructionLookupStuff, InstructionLookupsProof, PrimarySumcheck,
            PrimarySumcheckOpenings,
        },
        read_write_memory::{OutputSumcheckProof, ReadWriteMemoryProof, ReadWriteMemoryStuff},
        rv32i_vm::{RV32ISubtables, C, M, RV32I},
        timestamp_range_check::{TimestampRangeCheckStuff, TimestampValidityProof},
        JoltPreprocessing, JoltProof, JoltStuff,
    },
    lasso::memory_checking::{MultisetHashes, StructuredPolynomialData},
    poly::{
        commitment::hyperkzg::{HyperKZG, HyperKZGCommitment},
        opening_proof::ReducedOpeningProof,
        unipoly::UniPoly,
    },
    r1cs::{
        inputs::{AuxVariableStuff, JoltR1CSInputs, R1CSStuff},
        spartan::UniformSpartanProof,
    },
    subprotocols::{
        grand_product::{BatchedGrandProductLayerProof, BatchedGrandProductProof},
        sumcheck::SumcheckInstanceProof,
    },
    utils::poseidon_transcript::PoseidonTranscript,
};

use super::Parse;
use ark_bn254::Bn254;
use ark_ff::{BigInt, BigInteger, PrimeField};
use serde_json::json;
use tracer::JoltDevice;

pub(crate) type Fr = ark_bn254::Fr;
pub(crate) type Fq = ark_bn254::Fq;
pub(crate) type ProofTranscript = PoseidonTranscript<Fr, Fr>;
pub(crate) type PCS = HyperKZG<ark_bn254::Bn254, ProofTranscript>;

pub fn to_limbs<F: PrimeField, K: PrimeField>(r: F) -> [K; 3] {
    let mut limbs = [K::ZERO; 3];
    let r_bits = r.into_bigint().to_bits_le();
    limbs[0] = K::from_le_bytes_mod_order(
        &BigInt::<4>::from_bits_le(&r_bits.iter().take(125).cloned().collect::<Vec<bool>>())
            .to_bytes_le(),
    );
    limbs[1] = K::from_le_bytes_mod_order(
        &BigInt::<4>::from_bits_le(
            &r_bits
                .iter()
                .skip(125)
                .take(125)
                .cloned()
                .collect::<Vec<bool>>(),
        )
        .to_bytes_le(),
    );
    limbs[2] = K::from_le_bytes_mod_order(
        &BigInt::<4>::from_bits_le(
            &r_bits
                .iter()
                .skip(250)
                .take(4)
                .cloned()
                .collect::<Vec<bool>>(),
        )
        .to_bytes_le(),
    );
    limbs
}

impl Parse for Fr {
    fn format_non_native(&self) -> serde_json::Value {
        let limbs = to_limbs::<Fr, Fq>(*self);
        json!({
            "limbs": [limbs[0].to_string(), limbs[1].to_string(), limbs[2].to_string()]
        })
    }
}

impl Parse for ProofTranscript {
    fn format(&self) -> serde_json::Value {
        json!({
            "state": self.state.state[1].to_string(),
            "nRounds": self.n_rounds.to_string(),
        })
    }
}

//Parse Jolt
impl Parse
    for JoltProof<{ C }, { M }, JoltR1CSInputs, Fr, PCS, RV32I, RV32ISubtables<Fr>, ProofTranscript>
{
    fn format(&self) -> serde_json::Value {
        json!({
            "trace_length": self.trace_length.to_string(),
            "program_io": self.program_io.format(),
            "bytecode": self.bytecode.format(),
            "read_write_memory": self.read_write_memory.format(),
            "instruction_lookups": self.instruction_lookups.format(),
            "r1cs": self.r1cs.format(),
            "opening_proof":self.opening_proof.format()
        })
    }
}
impl Parse for JoltDevice {
    fn format(&self) -> serde_json::Value {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .map(|input| (*input as usize).to_string())
            .collect();
        let outputs: Vec<String> = self
            .outputs
            .iter()
            .map(|output| (*output as usize).to_string())
            .collect();
        json!({
                "inputs": inputs,
                "outputs": outputs,
                "panic": (self.panic as u8).to_string()
        })
    }
}

impl Parse for BytecodeProof<Fr, HyperKZG<Bn254, ProofTranscript>, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        let openings: Vec<String> = self
            .openings
            .read_write_values()
            .iter()
            .chain(self.openings.init_final_values().iter())
            .map(|opening| opening.to_string())
            .collect();
        json!({
            "multiset_hashes": self.multiset_hashes.format(),
            "read_write_grand_product": self.read_write_grand_product.format(),
            "init_final_grand_product": self.init_final_grand_product.format(),
            "openings":openings
        })
    }
}
impl Parse for ReadWriteMemoryProof<Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        let mut openings = Vec::new();
        openings.push(self.memory_checking_proof.openings.a_ram.to_string());
        openings.push(self.memory_checking_proof.openings.v_read_rd.to_string());
        openings.push(self.memory_checking_proof.openings.v_read_rs1.to_string());
        openings.push(self.memory_checking_proof.openings.v_read_rs2.to_string());
        openings.push(self.memory_checking_proof.openings.v_read_ram.to_string());
        openings.push(self.memory_checking_proof.openings.v_write_rd.to_string());
        openings.push(self.memory_checking_proof.openings.v_write_ram.to_string());
        openings.push(self.memory_checking_proof.openings.v_final.to_string());
        openings.push(self.memory_checking_proof.openings.t_read_rd.to_string());
        openings.push(self.memory_checking_proof.openings.t_read_rs1.to_string());
        openings.push(self.memory_checking_proof.openings.t_read_rs2.to_string());
        openings.push(self.memory_checking_proof.openings.t_read_ram.to_string());
        openings.push(self.memory_checking_proof.openings.t_final.to_string());

        let mut exogenous_openings = Vec::new();
        exogenous_openings.push(
            self.memory_checking_proof
                .exogenous_openings
                .a_rd
                .to_string(),
        );
        exogenous_openings.push(
            self.memory_checking_proof
                .exogenous_openings
                .a_rs1
                .to_string(),
        );
        exogenous_openings.push(
            self.memory_checking_proof
                .exogenous_openings
                .a_rs2
                .to_string(),
        );

        json!({
            "memory_checking_proof": { "multiset_hashes": self.memory_checking_proof.multiset_hashes.format(),
                                       "read_write_grand_product" : self.memory_checking_proof.read_write_grand_product.format(),
                                       "init_final_grand_product" : self.memory_checking_proof.init_final_grand_product.format(),
                                       "openings": openings,
                                       "exogenous_openings": exogenous_openings
                            },
            "timestamp_validity_proof": self.timestamp_validity_proof.format(),
            "output_proof": self.output_proof.format()
        })
    }
}
impl Parse
    for InstructionLookupsProof<{ C }, { M }, Fr, PCS, RV32I, RV32ISubtables<Fr>, ProofTranscript>
{
    fn format(&self) -> serde_json::Value {
        let openings: Vec<String> = self
            .memory_checking
            .openings
            .dim
            .iter()
            .chain(self.memory_checking.openings.read_cts.iter())
            .chain(self.memory_checking.openings.final_cts.iter())
            .chain(self.memory_checking.openings.E_polys.iter())
            .chain(self.memory_checking.openings.instruction_flags.iter())
            .chain([self.memory_checking.openings.lookup_outputs].iter())
            .map(|com| com.to_string())
            .collect();

        json!({
            "primary_sumcheck": self.primary_sumcheck.format(),
            "memory_checking_proof": { "multiset_hashes": self.memory_checking.multiset_hashes.format(),
                                       "read_write_grand_product" : self.memory_checking.read_write_grand_product.format(),
                                       "init_final_grand_product" : self.memory_checking.init_final_grand_product.format(),
                                       "openings": openings,
                            },
        })
    }
}
impl Parse for UniformSpartanProof<{ C }, JoltR1CSInputs, Fr, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "outer_sumcheck_proof": self.outer_sumcheck_proof.format(),
            "inner_sumcheck_proof": self.inner_sumcheck_proof.format(),
            "outer_sumcheck_claims": [self.outer_sumcheck_claims.0.to_string(), self.outer_sumcheck_claims.1.to_string(), self.outer_sumcheck_claims.2.to_string()],
            "claimed_witness_evals": self.claimed_witness_evals.iter().map(|eval|eval.to_string()).collect::<Vec<String>>()
        })
    }
}
impl Parse for ReducedOpeningProof<Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "sumcheck_proof":self.sumcheck_proof.format(),
            "sumcheck_claims": self.sumcheck_claims.iter().map(|claim|claim.to_string()).collect::<Vec<String>>(),
            "joint_opening_proof": self.joint_opening_proof.format_non_native()
        })
    }
}
impl Parse for PrimarySumcheck<Fr, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "sumcheck_proof": self.sumcheck_proof.format(),
            "openings":self.openings.format(),
        })
    }
}
impl Parse for PrimarySumcheckOpenings<Fr> {
    fn format(&self) -> serde_json::Value {
        let E_poly_openings: Vec<String> = self
            .E_poly_openings
            .iter()
            .map(|opening| opening.to_string())
            .collect();
        let flag_openings: Vec<String> = self
            .flag_openings
            .iter()
            .map(|opening| opening.to_string())
            .collect();
        json!({
            "E_poly_openings": E_poly_openings,
            "flag_openings": flag_openings,
            "lookup_outputs_opening": self.lookup_outputs_opening.to_string(),
        })
    }
}
impl Parse for MultisetHashes<Fr> {
    fn format(&self) -> serde_json::Value {
        let read_hashes: Vec<String> = self
            .read_hashes
            .iter()
            .map(|hash| hash.to_string())
            .collect();
        let write_hashes: Vec<String> = self
            .write_hashes
            .iter()
            .map(|hash| hash.to_string())
            .collect();
        let init_hashes: Vec<String> = self
            .init_hashes
            .iter()
            .map(|hash| hash.to_string())
            .collect();
        let final_hashes: Vec<String> = self
            .final_hashes
            .iter()
            .map(|hash| hash.to_string())
            .collect();

        json!({
            "read_hashes": read_hashes,
            "write_hashes": write_hashes,
            "init_hashes": init_hashes,
            "final_hashes": final_hashes
        })
    }
}
impl Parse for TimestampValidityProof<Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "multiset_hashes": self.multiset_hashes.format(),
            "openings": self.openings.format(),
            "exogenous_openings": self.exogenous_openings.iter().map(|opening|opening.to_string()).collect::<Vec<String>>(),
            "batched_grand_product": self.batched_grand_product.format()
        })
    }
}

impl Parse for OutputSumcheckProof<Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "sumcheck_proof": self.sumcheck_proof.format(),
            "opening": self.opening.to_string()
        })
    }
}

impl Parse for JoltStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        json!({
            "bytecode": self.bytecode.format(),
            "read_write_memory": self.read_write_memory.format(),
            "instruction_lookups": self.instruction_lookups.format(),
            "timestamp_range_check": self.timestamp_range_check.format(),
            "r1cs": self.r1cs.format()
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "bytecode": self.bytecode.format_non_native(),
            "read_write_memory": self.read_write_memory.format_non_native(),
            "instruction_lookups": self.instruction_lookups.format_non_native(),
            "timestamp_range_check": self.timestamp_range_check.format_non_native(),
            "r1cs": self.r1cs.format_non_native()
        })
    }
}

impl Parse for BatchedGrandProductProof<PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        let num_gkr_layers = self.gkr_layers.len();

        let num_coeffs = self.gkr_layers[num_gkr_layers - 1].proof.uni_polys[0]
            .coeffs
            .len();

        let max_no_polys = self.gkr_layers[num_gkr_layers - 1].proof.uni_polys.len();

        let mut updated_gkr_layers = Vec::new();

        for idx in 0..num_gkr_layers {
            let zero_poly = UniPoly::from_coeff(vec![Fr::from(0u8); num_coeffs]);
            let len = self.gkr_layers[idx].proof.uni_polys.len();
            let updated_uni_poly: Vec<_> = self.gkr_layers[idx]
                .proof
                .uni_polys
                .clone()
                .into_iter()
                .chain(vec![zero_poly; max_no_polys - len].into_iter())
                .collect();
            let layer = BatchedGrandProductLayerProof {
                proof: SumcheckInstanceProof::new(updated_uni_poly),
                left_claim: self.gkr_layers[idx].left_claim,
                right_claim: self.gkr_layers[idx].right_claim,
            };
            updated_gkr_layers.push(layer.format());
        }

        json!({
            "gkr_layers": updated_gkr_layers
        })
    }
}

impl Parse for BatchedGrandProductLayerProof<Fr, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "proof": self.proof.format(),
            "left_claim": self.left_claim.to_string(),
            "right_claim": self.right_claim.to_string(),
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "proof": self.proof.format_non_native(),
            "left_claim": self.left_claim.format(),
            "right_claim": self.right_claim.format(),
        })
    }
}
impl Parse for SumcheckInstanceProof<Fr, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        let uni_polys: Vec<serde_json::Value> =
            self.uni_polys.iter().map(|poly| poly.format()).collect();
        json!({
            "uni_polys": uni_polys,
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        let uni_polys: Vec<serde_json::Value> = self
            .uni_polys
            .iter()
            .map(|poly| poly.format_non_native())
            .collect();
        json!({
            "uni_polys": uni_polys,
        })
    }
}

impl Parse for UniPoly<Fr> {
    fn format(&self) -> serde_json::Value {
        let coeffs: Vec<String> = self.coeffs.iter().map(|coeff| coeff.to_string()).collect();
        json!({
            "coeffs": coeffs,
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        let coeffs: Vec<serde_json::Value> = self
            .coeffs
            .iter()
            .map(|coeff| coeff.format_non_native())
            .collect();
        json!({
            "coeffs": coeffs,
        })
    }
}
impl Parse for BytecodeStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        let v_read_write: Vec<serde_json::Value> =
            self.v_read_write.iter().map(|v| v.format()).collect();
        json!({
            "a_read_write": self.a_read_write.format(),
            "v_read_write": v_read_write,
            "t_read": self.t_read.format(),
            "t_final": self.t_final.format()
        })
    }

    fn format_non_native(&self) -> serde_json::Value {
        let v_read_write: Vec<serde_json::Value> = self
            .v_read_write
            .iter()
            .map(|v| v.format_non_native())
            .collect();
        json!({
            "a_read_write": self.a_read_write.format_non_native(),
            "v_read_write": v_read_write,
            "t_read": self.t_read.format_non_native(),
            "t_final": self.t_final.format_non_native()
        })
    }
}

impl Parse for ReadWriteMemoryStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        json!({
            "a_ram": self.a_ram.format(),
                "v_read_rd": self.v_read_rd.format(),
                "v_read_rs1": self.v_read_rs1.format(),
                "v_read_rs2": self.v_read_rs2.format(),
                "v_read_ram": self.v_read_ram.format(),
                "v_write_rd": self.v_write_rd.format(),
                "v_write_ram": self.v_write_ram.format(),
                "v_final": self.v_final.format(),
                "t_read_rd": self.t_read_rd.format(),
                "t_read_rs1": self.t_read_rs1.format(),
                "t_read_rs2": self.t_read_rs2.format(),
                "t_read_ram": self.t_read_ram.format(),
                "t_final": self.t_final.format()
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "a_ram": self.a_ram.format_non_native(),
                "v_read_rd": self.v_read_rd.format_non_native(),
                "v_read_rs1": self.v_read_rs1.format_non_native(),
                "v_read_rs2": self.v_read_rs2.format_non_native(),
                "v_read_ram": self.v_read_ram.format_non_native(),
                "v_write_rd": self.v_write_rd.format_non_native(),
                "v_write_ram": self.v_write_ram.format_non_native(),
                "v_final": self.v_final.format_non_native(),
                "t_read_rd": self.t_read_rd.format_non_native(),
                "t_read_rs1": self.t_read_rs1.format_non_native(),
                "t_read_rs2": self.t_read_rs2.format_non_native(),
                "t_read_ram": self.t_read_ram.format_non_native(),
                "t_final": self.t_final.format_non_native()
        })
    }
}
impl Parse for InstructionLookupStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        let dim: Vec<serde_json::Value> = self.dim.iter().map(|com| com.format()).collect();
        let read_cts: Vec<serde_json::Value> =
            self.read_cts.iter().map(|com| com.format()).collect();
        let final_cts: Vec<serde_json::Value> =
            self.final_cts.iter().map(|com| com.format()).collect();
        let E_polys: Vec<serde_json::Value> = self.E_polys.iter().map(|com| com.format()).collect();
        let instruction_flags: Vec<serde_json::Value> = self
            .instruction_flags
            .iter()
            .map(|com| com.format())
            .collect();
        json!({
            "dim": dim,
            "read_cts": read_cts,
            "final_cts": final_cts,
            "E_polys": E_polys,
            "instruction_flags": instruction_flags,
            "lookup_outputs": self.lookup_outputs.format()
        })
    }

    fn format_non_native(&self) -> serde_json::Value {
        let dim: Vec<serde_json::Value> =
            self.dim.iter().map(|com| com.format_non_native()).collect();
        let read_cts: Vec<serde_json::Value> = self
            .read_cts
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let final_cts: Vec<serde_json::Value> = self
            .final_cts
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let E_polys: Vec<serde_json::Value> = self
            .E_polys
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let instruction_flags: Vec<serde_json::Value> = self
            .instruction_flags
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        json!({
            "dim": dim,
            "read_cts": read_cts,
            "final_cts": final_cts,
            "E_polys": E_polys,
            "instruction_flags": instruction_flags,
            "lookup_outputs": self.lookup_outputs.format_non_native()
        })
    }
}

impl Parse for TimestampRangeCheckStuff<Fr> {
    fn format(&self) -> serde_json::Value {
        let read_cts_read_timestamp: Vec<String> = self
            .read_cts_read_timestamp
            .iter()
            .map(|com| com.to_string())
            .collect();
        let read_cts_global_minus_read: Vec<String> = self
            .read_cts_global_minus_read
            .iter()
            .map(|com| com.to_string())
            .collect();
        let final_cts_read_timestamp: Vec<String> = self
            .final_cts_read_timestamp
            .iter()
            .map(|com| com.to_string())
            .collect();
        let final_cts_global_minus_read: Vec<String> = self
            .final_cts_global_minus_read
            .iter()
            .map(|com| com.to_string())
            .collect();
        json!({
             "read_cts_read_timestamp": read_cts_read_timestamp,
                "read_cts_global_minus_read":read_cts_global_minus_read,
                "final_cts_read_timestamp": final_cts_read_timestamp,
                "final_cts_global_minus_read": final_cts_global_minus_read
        })
    }
}

impl Parse for TimestampRangeCheckStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        let read_cts_read_timestamp: Vec<serde_json::Value> = self
            .read_cts_read_timestamp
            .iter()
            .map(|com| com.format())
            .collect();
        let read_cts_global_minus_read: Vec<serde_json::Value> = self
            .read_cts_global_minus_read
            .iter()
            .map(|com| com.format())
            .collect();
        let final_cts_read_timestamp: Vec<serde_json::Value> = self
            .final_cts_read_timestamp
            .iter()
            .map(|com| com.format())
            .collect();
        let final_cts_global_minus_read: Vec<serde_json::Value> = self
            .final_cts_global_minus_read
            .iter()
            .map(|com| com.format())
            .collect();
        json!({
             "read_cts_read_timestamp": read_cts_read_timestamp,
                "read_cts_global_minus_read":read_cts_global_minus_read,
                "final_cts_read_timestamp": final_cts_read_timestamp,
                "final_cts_global_minus_read": final_cts_global_minus_read
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        let read_cts_read_timestamp: Vec<serde_json::Value> = self
            .read_cts_read_timestamp
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let read_cts_global_minus_read: Vec<serde_json::Value> = self
            .read_cts_global_minus_read
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let final_cts_read_timestamp: Vec<serde_json::Value> = self
            .final_cts_read_timestamp
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let final_cts_global_minus_read: Vec<serde_json::Value> = self
            .final_cts_global_minus_read
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        json!({
             "read_cts_read_timestamp": read_cts_read_timestamp,
                "read_cts_global_minus_read":read_cts_global_minus_read,
                "final_cts_read_timestamp": final_cts_read_timestamp,
                "final_cts_global_minus_read": final_cts_global_minus_read
        })
    }
}
impl Parse for R1CSStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        let chunks_x: Vec<serde_json::Value> =
            self.chunks_x.iter().map(|com| com.format()).collect();
        let chunks_y: Vec<serde_json::Value> =
            self.chunks_y.iter().map(|com| com.format()).collect();
        let circuit_flags: Vec<serde_json::Value> =
            self.circuit_flags.iter().map(|com| com.format()).collect();
        json!({
            "chunks_x": chunks_x,
            "chunks_y": chunks_y,
            "circuit_flags": circuit_flags,
            "aux": self.aux.format()
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        let chunks_x: Vec<serde_json::Value> = self
            .chunks_x
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let chunks_y: Vec<serde_json::Value> = self
            .chunks_y
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let circuit_flags: Vec<serde_json::Value> = self
            .circuit_flags
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        json!({
            "chunks_x": chunks_x,
            "chunks_y": chunks_y,
            "circuit_flags": circuit_flags,
            "aux": self.aux.format_non_native()
        })
    }
}
impl Parse for AuxVariableStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        let relevant_y_chunks: Vec<serde_json::Value> = self
            .relevant_y_chunks
            .iter()
            .map(|com| com.format())
            .collect();
        json!({
                "left_lookup_operand": self.left_lookup_operand.format(),
                "right_lookup_operand":  self.right_lookup_operand.format(),
                "product": self.product.format(),
                "relevant_y_chunks": relevant_y_chunks,
                "write_lookup_output_to_rd": self.write_lookup_output_to_rd.format(),
                "write_pc_to_rd":  self.write_pc_to_rd.format(),
                "next_pc_jump":  self.next_pc_jump.format(),
                "should_branch":  self.should_branch.format(),
                "next_pc": self.next_pc.format(),
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        let relevant_y_chunks: Vec<serde_json::Value> = self
            .relevant_y_chunks
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        json!({
                "left_lookup_operand": self.left_lookup_operand.format_non_native(),
                "right_lookup_operand":  self.right_lookup_operand.format_non_native(),
                "product": self.product.format_non_native(),
                "relevant_y_chunks": relevant_y_chunks,
                "write_lookup_output_to_rd": self.write_lookup_output_to_rd.format_non_native(),
                "write_pc_to_rd":  self.write_pc_to_rd.format_non_native(),
                "next_pc_jump":  self.next_pc_jump.format_non_native(),
                "should_branch":  self.should_branch.format_non_native(),
                "next_pc": self.next_pc.format_non_native(),
        })
    }
}

impl Parse for JoltPreprocessing<{ C }, Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        let v_init_final: Vec<Vec<String>> = self
            .bytecode
            .v_init_final
            .iter()
            .map(|poly| poly.Z.iter().map(|elem| elem.to_string()).collect())
            .collect();
        json!({
           "bytecode" : {
                        "v_init_final": v_init_final
                        },
           "read_write_memory": {"bytecode_words": self.read_write_memory.bytecode_words.iter().map(|elem|elem.to_string()).collect::<Vec<String>>() }
        })
    }
}
