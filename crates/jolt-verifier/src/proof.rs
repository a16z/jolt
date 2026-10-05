//! Verifier-owned proof model types.
//!
//! A proof is its argument string. Its leading prover messages are the
//! [`ProofHeader`] and the [`ProofCommitments`]; every later message's width is
//! derived from them and from the verifier's preprocessing.

#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
pub use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
use jolt_openings::CommitmentScheme;
use jolt_transcript::{ProverTranscript, Sponge, VerifierTranscript};
use serde::{Deserialize, Serialize};

use crate::{config::JoltProtocolConfig, jolt_protocol_id, num, VerifierError, JOLT_SESSION};

/// A Jolt proof: the argument string, plus the protocol axes it was produced
/// under so a build mismatch is reported as such instead of as a transcript
/// failure. The verifier binds its own configuration into the transcript, so
/// `protocol` carries no soundness weight.
///
/// The argument string's regions, in order, each opened by its
/// [`sites`](crate::sites) label:
///
/// 1. [`ProofHeader`], then [`ProofCommitments`].
/// 2. Stages 1–7: per batch, any uni-skip round, the rounds, then the output
///    claims in the shape of the verifier-derived output points. A clear proof
///    sends the generated `wire_claim_values`; a committed proof sends row
///    commitments over the generated `committed_claim_layout`. A clear stage 4
///    sends its
///    [`RamValCheckStagedOpenings`](crate::stages::stage4::ram_val_check::RamValCheckStagedOpenings)
///    before its batch.
/// 3. Stage 8: the joint opening.
/// 4. BlindFold, in committed proofs only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProof {
    pub protocol: JoltProtocolConfig,
    pub narg: Vec<u8>,
}

impl JoltProof {
    /// Decodes the proof header (the argument string's first message) on
    /// sponge `H`, without verifying the proof.
    pub fn header<H: Sponge>(&self) -> Result<ProofHeader, VerifierError> {
        let mut transcript =
            VerifierTranscript::<H>::new(&jolt_protocol_id::<H>(), JOLT_SESSION, &self.narg);
        ProofHeader::receive(&mut transcript)
    }
}

/// The prover-chosen shape parameters, sent first.
#[expect(non_snake_case, reason = "Preserves the protocol's parameter name.")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofHeader {
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
    pub untrusted_advice: bool,
}

impl ProofHeader {
    pub fn send<H: Sponge>(&self, transcript: &mut ProverTranscript<H>) {
        transcript.send(&num::u64_from_usize(self.trace_length));
        transcript.send(&num::u64_from_usize(self.ram_K));
        transcript.send_all(&[
            self.rw_config.ram_rw_phase1_num_rounds,
            self.rw_config.ram_rw_phase2_num_rounds,
            self.rw_config.registers_rw_phase1_num_rounds,
            self.rw_config.registers_rw_phase2_num_rounds,
            self.one_hot_config.log_k_chunk,
            self.one_hot_config.lookups_ra_virtual_log_k_chunk,
        ]);
        transcript.send(&self.trace_polynomial_order.transcript_scalar());
        transcript.send(&u8::from(self.untrusted_advice));
    }

    /// Reads the header, rejecting values outside each field's encoding. The
    /// values themselves are validated against the preprocessing by
    /// [`validate_inputs`](crate::verifier::validate_inputs).
    pub fn receive<H: Sponge>(
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<Self, VerifierError> {
        let trace_length = usize_field(transcript.receive::<u64>()?, "trace_length")?;
        let ram_k = usize_field(transcript.receive::<u64>()?, "ram_K")?;
        let [ram_phase1, ram_phase2, registers_phase1, registers_phase2, log_k_chunk, lookups_chunk]: [u8; 6] =
            transcript.receive()?;
        let trace_polynomial_order = TracePolynomialOrder::from_transcript_scalar(
            transcript.receive::<u64>()?,
        )
        .ok_or(VerifierError::MalformedProofHeader {
            field: "trace_polynomial_order",
        })?;
        let untrusted_advice = match transcript.receive::<u8>()? {
            0 => false,
            1 => true,
            _ => {
                return Err(VerifierError::MalformedProofHeader {
                    field: "untrusted_advice",
                })
            }
        };
        Ok(Self {
            trace_length,
            ram_K: ram_k,
            rw_config: JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: ram_phase1,
                ram_rw_phase2_num_rounds: ram_phase2,
                registers_rw_phase1_num_rounds: registers_phase1,
                registers_rw_phase2_num_rounds: registers_phase2,
            },
            one_hot_config: JoltOneHotConfig {
                log_k_chunk,
                lookups_ra_virtual_log_k_chunk: lookups_chunk,
            },
            trace_polynomial_order,
            untrusted_advice,
        })
    }
}

fn usize_field(value: u64, field: &'static str) -> Result<usize, VerifierError> {
    usize::try_from(value).map_err(|_| VerifierError::MalformedProofHeader { field })
}

/// The field-register commitments of the field-inline extension. `FieldRdInc` is the
/// extension's single committed polynomial; the field-register access columns are virtual
/// (anchored through the bytecode read-RAF path), so this nest stays one deep until the
/// protocol commits more.
#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldRegistersCommitments<C> {
    pub rd_inc: C,
}

/// The field-inline extension's committed payload, grouped by component as the
/// protocol spec lays it out (`FieldInlineCommitments::field_registers`).
#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineCommitments<C> {
    pub field_registers: FieldRegistersCommitments<C>,
}

/// One commitment per committed trace polynomial on the homomorphic build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoltCommitments<C> {
    pub rd_inc: C,
    pub ram_inc: C,
    pub instruction_ra: Vec<C>,
    pub ram_ra: Vec<C>,
    pub bytecode_ra: Vec<C>,
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineCommitments<C>,
}

/// The polynomial commitments a proof sends, in send order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofCommitments<C> {
    /// One commitment per committed trace polynomial.
    #[cfg(not(feature = "akita"))]
    pub trace: JoltCommitments<C>,
    /// The single packed `OneHotTrace` commitment carrying every per-proof column.
    #[cfg(feature = "akita")]
    pub one_hot_trace: C,
    /// The direct commitment to the full field-register increment polynomial.
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    pub field_inc: C,
    /// Present exactly when the header declares untrusted advice.
    pub untrusted_advice: Option<C>,
}

impl<C> ProofCommitments<C> {
    pub fn send<PCS, H>(&self, transcript: &mut ProverTranscript<H>)
    where
        PCS: CommitmentScheme<Output = C>,
        H: Sponge,
    {
        #[cfg(not(feature = "akita"))]
        {
            let trace = &self.trace;
            PCS::send_commitment(&trace.rd_inc, transcript);
            PCS::send_commitment(&trace.ram_inc, transcript);
            for commitment in trace
                .instruction_ra
                .iter()
                .chain(&trace.ram_ra)
                .chain(&trace.bytecode_ra)
            {
                PCS::send_commitment(commitment, transcript);
            }
            #[cfg(feature = "field-inline")]
            PCS::send_commitment(&trace.field_inline.field_registers.rd_inc, transcript);
        }
        #[cfg(feature = "akita")]
        {
            PCS::send_commitment(&self.one_hot_trace, transcript);
            #[cfg(feature = "field-inline")]
            PCS::send_commitment(&self.field_inc, transcript);
        }
        if let Some(commitment) = &self.untrusted_advice {
            PCS::send_commitment(commitment, transcript);
        }
    }

    /// Reads the commitments at the counts `layout` (homomorphic build) and
    /// `header` fix.
    pub fn receive<PCS, H>(
        setup: &PCS::VerifierSetup,
        header: &ProofHeader,
        #[cfg(not(feature = "akita"))] layout: JoltRaPolynomialLayout,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<Self, VerifierError>
    where
        PCS: CommitmentScheme<Output = C>,
        H: Sponge,
    {
        let mut receive = || {
            PCS::receive_commitment(setup, transcript).map_err(|error| {
                VerifierError::MalformedCommitment {
                    reason: error.to_string(),
                }
            })
        };
        #[cfg(not(feature = "akita"))]
        let trace = {
            let rd_inc = receive()?;
            let ram_inc = receive()?;
            let instruction_ra = (0..layout.instruction())
                .map(|_| receive())
                .collect::<Result<_, _>>()?;
            let ram_ra = (0..layout.ram())
                .map(|_| receive())
                .collect::<Result<_, _>>()?;
            let bytecode_ra = (0..layout.bytecode())
                .map(|_| receive())
                .collect::<Result<_, _>>()?;
            JoltCommitments {
                rd_inc,
                ram_inc,
                instruction_ra,
                ram_ra,
                bytecode_ra,
                #[cfg(feature = "field-inline")]
                field_inline: FieldInlineCommitments {
                    field_registers: FieldRegistersCommitments { rd_inc: receive()? },
                },
            }
        };
        #[cfg(feature = "akita")]
        let one_hot_trace = receive()?;
        #[cfg(all(feature = "akita", feature = "field-inline"))]
        let field_inc = receive()?;
        let untrusted_advice = header.untrusted_advice.then(&mut receive).transpose()?;
        Ok(Self {
            #[cfg(not(feature = "akita"))]
            trace,
            #[cfg(feature = "akita")]
            one_hot_trace,
            #[cfg(all(feature = "akita", feature = "field-inline"))]
            field_inc,
            untrusted_advice,
        })
    }
}
