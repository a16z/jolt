//! Dory polynomial commitment scheme implementation

use super::dory_globals::DoryGlobals;
use super::jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
use super::wrappers::{
    jolt_to_ark, ArkDoryProof, ArkFr, ArkG1, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    JoltToDoryTranscript, BN254,
};
use crate::{
    field::JoltField,
    msm::VariableBaseMSM,
    poly::commitment::commitment_scheme::{CommitmentScheme, StreamingCommitmentScheme},
    poly::multilinear_polynomial::MultilinearPolynomial,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use ark_bn254::{G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::Zero;
use dory::primitives::{
    arithmetic::{Group, PairingCurve},
    poly::Polynomial,
};
use rand_core::OsRng;
use rayon::prelude::*;
use std::{borrow::Borrow, collections::HashMap};
use tracing::trace_span;

#[derive(Clone)]
pub struct DoryCommitmentScheme;

impl CommitmentScheme for DoryCommitmentScheme {
    type Field = ark_bn254::Fr;
    type ProverSetup = ArkworksProverSetup;
    type VerifierSetup = ArkworksVerifierSetup;
    type Commitment = ArkGT;
    type Proof = ArkDoryProof;
    type BatchedProof = Vec<ArkDoryProof>;
    type OpeningProofHint = Vec<ArkG1>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_prover").entered();
        let setup = ArkworksProverSetup::new(&mut OsRng, max_num_vars);

        // Initialize the prepared point cache for faster multi-pairings
        // Skips cache initialization during tests to avoid shared state issues
        #[cfg(not(test))]
        DoryGlobals::init_prepared_cache(&setup.g1_vec, &setup.g2_vec);

        setup
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_verifier").entered();
        setup.to_verifier_setup()
    }

    fn commit(
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let _span = trace_span!("DoryCommitmentScheme::commit").entered();

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        let (tier_2, row_commitments) = <MultilinearPolynomial<ark_bn254::Fr> as Polynomial<
            ArkFr,
        >>::commit::<BN254, JoltG1Routines>(
            poly, nu, sigma, setup
        )
        .expect("commitment should succeed");

        (tier_2, row_commitments)
    }

    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<ark_bn254::Fr>> + Sync,
    {
        let _span = trace_span!("DoryCommitmentScheme::batch_commit").entered();

        polys
            .par_iter()
            .map(|poly| Self::commit(poly.borrow(), gens))
            .collect()
    }

    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        row_commitments: Self::OpeningProofHint,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let _span = trace_span!("DoryCommitmentScheme::prove").entered();

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = opening_point
            .iter()
            .rev()  // Reverse the order for Dory
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        dory::prove::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _, _>(
            poly,
            &ark_point,
            row_commitments,
            nu,
            sigma,
            setup,
            &mut dory_transcript,
        )
        .expect("proof generation should succeed")
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        opening: &ark_bn254::Fr,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let _span = trace_span!("DoryCommitmentScheme::verify").entered();

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = opening_point
            .iter()
            .rev()  // Reverse the order for Dory
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();
        let ark_eval: ArkFr = jolt_to_ark(opening);

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        dory::verify::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _>(
            *commitment,
            ark_eval,
            &ark_point,
            proof,
            setup.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| ProofVerifyError::InternalError)?;

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Dory"
    }

    /// In Dory, the opening proof hint consists of the Pedersen commitments to the rows
    /// of the polynomial coefficient matrix. In the context of a batch opening proof, we
    /// can homomorphically combine the row commitments for multiple polynomials into the
    /// row commitments for the RLC of those polynomials. This is more efficient than computing
    /// the row commitments for the RLC from scratch.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_hints")]
    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        let num_rows = DoryGlobals::get_max_num_rows();

        let mut rlc_hint = vec![ArkG1(G1Projective::zero()); num_rows];
        for (coeff, mut hint) in coeffs.iter().zip(hints.into_iter()) {
            hint.resize(num_rows, ArkG1(G1Projective::zero()));

            let row_commitments: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(hint.as_mut_ptr() as *mut G1Projective, hint.len())
            };

            let rlc_row_commitments: &[G1Projective] = unsafe {
                std::slice::from_raw_parts(rlc_hint.as_ptr() as *const G1Projective, rlc_hint.len())
            };

            let _span = trace_span!("vector_scalar_mul_add_gamma_g1_online");
            let _enter = _span.enter();

            // Scales the row commitments for the current polynomial by
            // its coefficient
            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                row_commitments,
                *coeff,
                rlc_row_commitments,
            );

            let _ = std::mem::replace(&mut rlc_hint, hint);
        }

        rlc_hint
    }

    /// Homomorphically combines multiple commitments using a random linear combination.
    /// Computes: sum_i(coeff_i * commitment_i) for the GT elements.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_commitments")]
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let _span = trace_span!("DoryCommitmentScheme::combine_commitments").entered();

        // Combine GT elements using parallel RLC
        let commitments_vec: Vec<&ArkGT> = commitments.iter().map(|c| c.borrow()).collect();
        coeffs
            .par_iter()
            .zip(commitments_vec.par_iter())
            .map(|(coeff, commitment)| {
                let ark_coeff = jolt_to_ark(coeff);
                ark_coeff * **commitment
            })
            .reduce(|| ArkGT::identity(), |a, b| a + b)
    }
}

impl StreamingCommitmentScheme for DoryCommitmentScheme {
    type Tier1Commitment = ArkG1;

    fn compute_tier_2_commit(
        tier_1_commitments: &[Self::Tier1Commitment],
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let _span = trace_span!("DoryCommitmentScheme::compute_tier_2_commit").entered();

        let row_commitments = tier_1_commitments.to_vec();
        let num_rows = row_commitments.len();
        let g2_bases = &setup.g2_vec[..num_rows];

        let tier_2 = <BN254 as PairingCurve>::multi_pair(&row_commitments, g2_bases);

        (tier_2, row_commitments)
    }

    fn streaming_batch_commit<F, PCS>(
        polynomial_specs: &[crate::zkvm::witness::CommittedPolynomial],
        lazy_trace: &mut tracer::LazyTraceIterator,
        preprocessing: &crate::zkvm::JoltProverPreprocessing<F, PCS>,
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        F: crate::field::JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        use crate::zkvm::witness::{CommittedPolynomial, DTH_ROOT_OF_K};
        use crate::zkvm::instruction_lookups;

        let _span = trace_span!("DoryCommitmentScheme::streaming_batch_commit").entered();

        let row_len = DoryGlobals::get_num_columns();
        let T = DoryGlobals::get_T();

        // Calculate dimensions for RA polynomials once
        let mut ram_d = 0;
        for poly in polynomial_specs {
            match poly {
                CommittedPolynomial::RamRa(i) => ram_d = ram_d.max(*i + 1),
                _ => {}
            }
        }

        // Get G1 bases for MSM computations
        let g1_slice = unsafe {
            std::slice::from_raw_parts(setup.g1_vec.as_ptr() as *const ArkG1, setup.g1_vec.len())
        };
        let bases: Vec<G1Affine> = g1_slice
            .iter()
            .take(row_len)
            .map(|g| g.0.into_affine())
            .collect();

        // Separate polynomials into regular and one-hot
        let mut regular_polys = Vec::new();
        let mut onehot_polys = Vec::new();

        for poly in polynomial_specs {
            match poly {
                CommittedPolynomial::RdInc | CommittedPolynomial::RamInc => {
                    regular_polys.push(*poly);
                }
                CommittedPolynomial::InstructionRa(_)
                | CommittedPolynomial::BytecodeRa(_)
                | CommittedPolynomial::RamRa(_) => {
                    onehot_polys.push(*poly);
                }
            }
        }

        // Initialize storage for results
        let mut poly_tier1_commitments: HashMap<
            crate::zkvm::witness::CommittedPolynomial,
            Vec<Self::Tier1Commitment>,
        > = HashMap::new();

        // Initialize regular polynomial tier 1 commitments
        for poly in &regular_polys {
            poly_tier1_commitments.insert(*poly, Vec::new());
        }

        // Initialize one-hot polynomial indices accumulation
        let mut onehot_indices: HashMap<
            crate::zkvm::witness::CommittedPolynomial,
            Vec<Vec<usize>>,
        > = HashMap::new();

        for poly in &onehot_polys {
            let k = match poly {
                CommittedPolynomial::InstructionRa(_) => instruction_lookups::K_CHUNK,
                CommittedPolynomial::BytecodeRa(_) => {
                    let d = preprocessing.shared.bytecode.d;
                    let log_K = preprocessing.shared.bytecode.code_size.log_2();
                    let log_K_chunk = log_K.div_ceil(d);
                    1 << log_K_chunk
                }
                CommittedPolynomial::RamRa(_) => DTH_ROOT_OF_K,
                _ => unreachable!(),
            };
            onehot_indices.insert(*poly, vec![Vec::new(); k]);
        }

        // Process trace in row-sized chunks
        let mut trace_buffer = Vec::with_capacity(row_len);
        let mut chunk_index = 0;

        while let Some(cycle) = lazy_trace.next() {
            trace_buffer.push(cycle);

            // When we have a full row, process it
            if trace_buffer.len() == row_len {
                // Process regular polynomials (compute tier 1 commitment immediately)
                for poly in &regular_polys {
                    let tier1_commit = Self::compute_tier1_for_poly(
                        poly,
                        &trace_buffer,
                        &bases,
                        preprocessing,
                        ram_d,
                    );

                    poly_tier1_commitments
                        .get_mut(poly)
                        .unwrap()
                        .push(tier1_commit);
                }

                // Process one-hot polynomials (accumulate indices)
                Self::accumulate_onehot_indices(
                    &onehot_polys,
                    &trace_buffer,
                    chunk_index,
                    row_len,
                    &mut onehot_indices,
                    preprocessing,
                    ram_d,
                );

                chunk_index += 1;
                trace_buffer.clear();
            }
        }

        // Handle partial last row if exists
        if !trace_buffer.is_empty() {
            // Pad the row with NoOp cycles
            let noop_cycle = tracer::instruction::Cycle::NoOp;
            trace_buffer.resize(row_len, noop_cycle);

            // Process regular polynomials
            for poly in &regular_polys {
                let tier1_commit = Self::compute_tier1_for_poly(
                    poly, &trace_buffer, &bases, preprocessing, ram_d);

                poly_tier1_commitments
                    .get_mut(poly)
                    .unwrap()
                    .push(tier1_commit);
            }

            // Process one-hot polynomials
            Self::accumulate_onehot_indices(
                &onehot_polys,
                &trace_buffer,
                chunk_index,
                row_len,
                &mut onehot_indices,
                preprocessing,
                ram_d,
            );
        }

        // Compute tier 1 commitments for one-hot polynomials using batch addition
        for poly in &onehot_polys {
            let all_k_indices = onehot_indices.remove(poly).unwrap();
            let k = all_k_indices.len();

            // Calculate number of rows in tier 1 for this polynomial
            let num_tier1_rows = match poly {
                CommittedPolynomial::InstructionRa(_)
                | CommittedPolynomial::BytecodeRa(_)
                | CommittedPolynomial::RamRa(_) => {
                    // For one-hot polynomials: num_rows = (T * K) / row_len
                    (T as u128 * k as u128 / row_len as u128) as usize
                }
                _ => unreachable!(),
            };

            let rows_per_k = T / row_len;
            let mut tier1_commits = vec![ArkG1(G1Projective::zero()); num_tier1_rows];

            // Process each chunk
            for chunk_idx in 0..rows_per_k {
                // Collect indices for this chunk
                let mut indices_per_k: Vec<Vec<usize>> = vec![Vec::new(); k];

                // Find which column indices in this chunk have each k value
                for k_idx in 0..k {
                    for &global_idx in &all_k_indices[k_idx] {
                        let chunk_of_idx = global_idx / row_len;
                        if chunk_of_idx == chunk_idx {
                            let col_idx = global_idx % row_len;
                            indices_per_k[k_idx].push(col_idx);
                        }
                    }
                }

                // Compute batch additions for this chunk
                let results = jolt_optimizations::batch_g1_additions_multi(&bases, &indices_per_k);

                // Place results in the correct tier 1 positions
                for (k_idx, result) in results.into_iter().enumerate() {
                    if !indices_per_k[k_idx].is_empty() {
                        let row_idx = chunk_idx + k_idx * rows_per_k;
                        if row_idx < tier1_commits.len() {
                            tier1_commits[row_idx] = ArkG1(G1Projective::from(result));
                        }
                    }
                }
            }

            poly_tier1_commitments.insert(*poly, tier1_commits);
        }

        // Compute tier 2 commitments for each polynomial
        polynomial_specs
            .iter()
            .map(|poly| {
                let tier1_commits = poly_tier1_commitments.remove(poly).unwrap();
                Self::compute_tier_2_commit(&tier1_commits, setup)
            })
            .collect()
    }
}

impl DoryCommitmentScheme {
    /// Helper function to accumulate one-hot indices for a chunk
    fn accumulate_onehot_indices<F, PCS>(
        onehot_polys: &[crate::zkvm::witness::CommittedPolynomial],
        row_cycles: &[tracer::instruction::Cycle],
        chunk_index: usize,
        row_len: usize,
        onehot_indices: &mut HashMap<crate::zkvm::witness::CommittedPolynomial, Vec<Vec<usize>>>,
        preprocessing: &crate::zkvm::JoltProverPreprocessing<F, PCS>,
        ram_d: usize,
    )
    where
        F: crate::field::JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        use crate::zkvm::instruction::LookupQuery;
        use crate::zkvm::{
            instruction_lookups,
            ram::remap_address,
            witness::{CommittedPolynomial, DTH_ROOT_OF_K},
        };
        use common::constants::XLEN;

        for poly in onehot_polys {
            match poly {
                CommittedPolynomial::InstructionRa(idx) => {
                    for (col_index, cycle) in row_cycles.iter().enumerate() {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        let k = (lookup_index
                            >> (instruction_lookups::LOG_K_CHUNK
                                * (instruction_lookups::D - 1 - idx)))
                            % instruction_lookups::K_CHUNK as u128;
                        let global_index = chunk_index * row_len + col_index;
                        onehot_indices
                            .get_mut(poly)
                            .unwrap()[k as usize]
                            .push(global_index);
                    }
                }
                CommittedPolynomial::BytecodeRa(idx) => {
                    let d = preprocessing.shared.bytecode.d;
                    let log_K = preprocessing.shared.bytecode.code_size.log_2();
                    let log_K_chunk = log_K.div_ceil(d);

                    for (col_index, cycle) in row_cycles.iter().enumerate() {
                        let pc = preprocessing.shared.bytecode.get_pc(cycle);
                        let k = (pc >> (log_K_chunk * (d - 1 - idx))) % (1 << log_K_chunk);
                        let global_index = chunk_index * row_len + col_index;
                        onehot_indices
                            .get_mut(poly)
                            .unwrap()[k]
                            .push(global_index);
                    }
                }
                CommittedPolynomial::RamRa(idx) => {
                    for (col_index, cycle) in row_cycles.iter().enumerate() {
                        if let Some(address) = remap_address(
                            cycle.ram_access().address() as u64,
                            &preprocessing.shared.memory_layout,
                        ) {
                            let k = (address as usize >> (DTH_ROOT_OF_K.log_2() * (ram_d - 1 - idx)))
                                % DTH_ROOT_OF_K;
                            let global_index = chunk_index * row_len + col_index;
                            onehot_indices
                                .get_mut(poly)
                                .unwrap()[k]
                                .push(global_index);
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    /// Helper function to compute tier 1 commitment for a single polynomial and row
    fn compute_tier1_for_poly<F, PCS>(
        poly: &crate::zkvm::witness::CommittedPolynomial,
        row_cycles: &[tracer::instruction::Cycle],
        bases: &[G1Affine],
        preprocessing: &crate::zkvm::JoltProverPreprocessing<F, PCS>,
        ram_d: usize,
    ) -> ArkG1
    where
        F: crate::field::JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        use crate::zkvm::witness::CommittedPolynomial;
        use tracer::instruction::RAMAccess;

        match poly {
            CommittedPolynomial::RdInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write();
                        post_value as i128 - pre_value as i128
                    })
                    .collect();
                ArkG1(VariableBaseMSM::msm_i128(&bases[..row.len()], &row).unwrap())
            }
            CommittedPolynomial::RamInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| match cycle.ram_access() {
                        RAMAccess::Write(write) => {
                            write.post_value as i128 - write.pre_value as i128
                        }
                        _ => 0,
                    })
                    .collect();
                ArkG1(VariableBaseMSM::msm_i128(&bases[..row.len()], &row).unwrap())
            }
            _ => panic!("compute_tier1_for_poly should only be called for regular polynomials"),
        }
    }
}

#[cfg(test)]
#[path = "commitment_scheme_test.rs"]
mod tests;
