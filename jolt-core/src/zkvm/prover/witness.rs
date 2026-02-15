use std::collections::HashMap;

use itertools::{zip_eq, Itertools};
use rayon::prelude::*;
use tracer::{instruction::Cycle, ChunksIterator};

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{StreamingCommitmentScheme, ZkEvalCommitment};
use crate::poly::commitment::dory::{DoryContext, DoryGlobals, DoryLayout};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::transcripts::Transcript;
use crate::zkvm::ram::populate_memory_states;
use crate::zkvm::witness::{all_committed_polynomials, CommittedPolynomial};

use super::JoltCpuProver;

impl<
        'a,
        F: JoltField,
        C: JoltCurve,
        PCS: StreamingCommitmentScheme<Field = F> + ZkEvalCommitment<C>,
        ProofTranscript: Transcript,
    > JoltCpuProver<'a, F, C, PCS, ProofTranscript>
{
    #[tracing::instrument(skip_all, name = "generate_and_commit_witness_polynomials")]
    pub(super) fn generate_and_commit_witness_polynomials(
        &mut self,
    ) -> (
        Vec<PCS::Commitment>,
        HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
    ) {
        let _guard = DoryGlobals::initialize_context(
            1 << self.one_hot_params.log_k_chunk,
            self.padded_trace_len,
            DoryContext::Main,
            Some(DoryGlobals::get_layout()),
        );

        let polys = all_committed_polynomials(&self.one_hot_params);
        let T = DoryGlobals::get_T();

        let (commitments, hint_map) = if DoryGlobals::get_layout() == DoryLayout::AddressMajor {
            tracing::debug!(
                "Using non-streaming commit path for AddressMajor layout with {} polynomials",
                polys.len()
            );

            let trace: Vec<Cycle> = self
                .lazy_trace
                .clone()
                .pad_using(T, |_| Cycle::NoOp)
                .collect();

            let (commitments, hints): (Vec<_>, Vec<_>) = polys
                .par_iter()
                .map(|poly_id| {
                    let witness: MultilinearPolynomial<F> = poly_id.generate_witness(
                        &self.preprocessing.shared.bytecode,
                        &self.preprocessing.shared.memory_layout,
                        &trace,
                        Some(&self.one_hot_params),
                    );
                    PCS::commit(&witness, &self.preprocessing.generators)
                })
                .unzip();

            let hint_map = HashMap::from_iter(zip_eq(polys, hints));
            (commitments, hint_map)
        } else {
            let row_len = DoryGlobals::get_num_columns();
            let num_rows = T / DoryGlobals::get_max_num_rows();

            tracing::debug!(
                "Generating and committing {} witness polynomials with T={}, row_len={}, num_rows={}",
                polys.len(),
                T,
                row_len,
                num_rows
            );

            let mut row_commitments: Vec<Vec<PCS::ChunkState>> = vec![vec![]; num_rows];

            self.lazy_trace
                .clone()
                .pad_using(T, |_| Cycle::NoOp)
                .iter_chunks(row_len)
                .zip(row_commitments.iter_mut())
                .par_bridge()
                .for_each(|(chunk, row_tier1_commitments)| {
                    let res: Vec<_> = polys
                        .par_iter()
                        .map(|poly| {
                            poly.stream_witness_and_commit_rows::<_, PCS>(
                                &self.preprocessing.generators,
                                &self.preprocessing.shared,
                                &chunk,
                                &self.one_hot_params,
                            )
                        })
                        .collect();
                    *row_tier1_commitments = res;
                });

            let tier1_per_poly: Vec<Vec<PCS::ChunkState>> = (0..polys.len())
                .into_par_iter()
                .map(|poly_idx| {
                    row_commitments
                        .iter()
                        .flat_map(|row| row.get(poly_idx).cloned())
                        .collect()
                })
                .collect();

            let (commitments, hints): (Vec<_>, Vec<_>) = tier1_per_poly
                .into_par_iter()
                .zip(&polys)
                .map(|(tier1_commitments, poly)| {
                    let onehot_k = poly.get_onehot_k(&self.one_hot_params);
                    PCS::aggregate_chunks(
                        &self.preprocessing.generators,
                        onehot_k,
                        &tier1_commitments,
                    )
                })
                .unzip();

            let hint_map = HashMap::from_iter(zip_eq(polys, hints));
            (commitments, hint_map)
        };

        for commitment in &commitments {
            self.transcript
                .append_serializable(b"commitment", commitment);
        }

        (commitments, hint_map)
    }

    pub(super) fn generate_and_commit_untrusted_advice(&mut self) -> Option<PCS::Commitment> {
        if self.program_io.untrusted_advice.is_empty() {
            return None;
        }

        let mut untrusted_advice_vec =
            vec![0; self.program_io.memory_layout.max_untrusted_advice_size as usize / 8];

        populate_memory_states(
            0,
            &self.program_io.untrusted_advice,
            Some(&mut untrusted_advice_vec),
            None,
        );

        let poly = MultilinearPolynomial::from(untrusted_advice_vec);
        let advice_len = poly.len().next_power_of_two().max(1);

        let _guard =
            DoryGlobals::initialize_context(1, advice_len, DoryContext::UntrustedAdvice, None);
        let _ctx = DoryGlobals::with_context(DoryContext::UntrustedAdvice);
        let (commitment, hint) = PCS::commit(&poly, &self.preprocessing.generators);
        self.transcript
            .append_serializable(b"untrusted_advice", &commitment);

        self.advice.untrusted_advice_polynomial = Some(poly);
        self.advice.untrusted_advice_hint = Some(hint);

        Some(commitment)
    }

    pub(super) fn generate_and_commit_trusted_advice(&mut self) {
        if self.program_io.trusted_advice.is_empty() {
            return;
        }

        let mut trusted_advice_vec =
            vec![0; self.program_io.memory_layout.max_trusted_advice_size as usize / 8];

        populate_memory_states(
            0,
            &self.program_io.trusted_advice,
            Some(&mut trusted_advice_vec),
            None,
        );

        let poly = MultilinearPolynomial::from(trusted_advice_vec);
        self.advice.trusted_advice_polynomial = Some(poly);
        self.transcript.append_serializable(
            b"trusted_advice",
            self.advice.trusted_advice_commitment.as_ref().unwrap(),
        );
    }
}
