//! Fixed full-program/no-advice boundary. This is not complete Jolt acceptance.
use common::jolt_device::JoltDevice;
use jolt_akita::r1cs::{AkitaCommitmentShape, AkitaCommitmentVars, CommitmentR1csError};
use jolt_akita::{AkitaField, AkitaScheme};
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig, TracePolynomialOrder};
use jolt_crypto::VectorCommitment;
use jolt_field::{Field, Fr, Ring};
use jolt_r1cs::bn254_bits::ByteVar;
use jolt_r1cs::fp128_bn254::{Fp128Error, Fp128Var};
use jolt_r1cs::{LinearCombination, R1csBuilder, Variable};
use jolt_sumcheck::{ClearProof, SumcheckProof};
use jolt_transcript::r1cs::{Blake2bR1csError, LegacyBlake2bVar};
use thiserror::Error;

use super::{Stage1R1csError, Stage1UniskipShape, Stage1UniskipVars};
use crate::config::{validate_proof_config, JOLT_VERIFIER_CONFIG};
use crate::preprocessing::ProgramPreprocessing;
use crate::stages::uniskip::UniskipParams;
use crate::verifier::{
    validate_inputs, validate_proof_consistency, CheckedInputs, PreambleValue,
    ProofTranscriptConfig,
};
use crate::{JoltProof, JoltVerifierPreprocessing, VerifierError};

#[derive(Debug, Error)]
pub enum BoundaryError {
    #[error("unsupported fixed boundary profile")]
    Profile,
    #[error("boundary requires a fresh builder with only ONE")]
    PublicPrefix,
    #[error(transparent)]
    Native(#[from] VerifierError),
    #[error(transparent)]
    Commitment(#[from] CommitmentR1csError),
    #[error(transparent)]
    Field(#[from] Fp128Error),
    #[error(transparent)]
    Stage(#[from] Stage1R1csError),
    #[error(transparent)]
    Blake(#[from] Blake2bR1csError),
}

/// Application-visible statement; the shape fixes its lengths and normalization.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryPublicInputs {
    pub inputs: [u8; 3],
    pub output: u8,
    pub panic: bool,
}

impl BoundaryPublicInputs {
    pub fn packed(self) -> Result<Fr, BoundaryError> {
        if self.output == 0 {
            return Err(BoundaryError::Profile);
        }
        let bytes = [
            self.inputs[0],
            self.inputs[1],
            self.inputs[2],
            self.output,
            u8::from(self.panic),
            0,
            0,
            0,
        ];
        Ok(Fr::from_u64(u64::from_le_bytes(bytes)))
    }
}

/// Actual proof cells; all are constrained, without a native acceptance flag.
#[derive(Clone)]
pub struct AkitaStage1BoundaryWitness {
    pub public: BoundaryPublicInputs,
    pub commitment: [u8; 128],
    pub coefficients: [u128; 28],
    pub output_claim: u128,
}

/// A fixed public shape selected from trusted preprocessing and validated metadata.
pub struct AkitaStage1BoundaryShape {
    checked: CheckedInputs,
    config: ProofTranscriptConfig,
    commitment: AkitaCommitmentShape,
    uniskip: Stage1UniskipShape,
}

/// Shared handles for the next verifier stages, including the live transcript.
pub struct AkitaStage1BoundaryVars {
    pub public: Variable,
    pub commitment: AkitaCommitmentVars,
    pub coefficients: Vec<Fp128Var>,
    pub uniskip: Stage1UniskipVars,
    pub transcript: LegacyBlake2bVar,
}

impl AkitaStage1BoundaryShape {
    pub fn new<VC, Zk>(
        preprocessing: &JoltVerifierPreprocessing<AkitaScheme, VC>,
        io: &JoltDevice,
        proof: &JoltProof<AkitaScheme, VC, Zk>,
    ) -> Result<Self, BoundaryError>
    where
        VC: VectorCommitment<Field = AkitaField>,
    {
        Self::check_metadata(proof)?;
        if UniskipParams::spartan_outer().degree() != 27
            || UniskipParams::spartan_outer().domain_size() != 10
        {
            return Err(BoundaryError::Profile);
        }
        if !matches!(&preprocessing.program, ProgramPreprocessing::Full(_)) {
            return Err(BoundaryError::Profile);
        }
        validate_proof_consistency(proof, false)?;
        let mut checked = validate_inputs(preprocessing, io, proof, false)?;
        let _ = Self::public(&checked.public_io)?;
        let commitment = AkitaCommitmentShape::new(&preprocessing.pcs_setup, &proof.commitments)?;
        // Public statement values are allocated later; none becomes a matrix coefficient.
        checked.public_io.inputs.fill(0);
        checked.public_io.outputs.fill(1);
        checked.public_io.panic = false;
        Ok(Self {
            checked,
            config: ProofTranscriptConfig {
                rw_config: proof.rw_config,
                one_hot_config: proof.one_hot_config,
                trace_polynomial_order: proof.trace_polynomial_order,
            },
            commitment,
            uniskip: Stage1UniskipShape::new(12, 28)?,
        })
    }

    fn check_metadata<VC, Zk>(proof: &JoltProof<AkitaScheme, VC, Zk>) -> Result<(), BoundaryError>
    where
        VC: VectorCommitment<Field = AkitaField>,
    {
        validate_proof_config(&JOLT_VERIFIER_CONFIG, proof.protocol)?;
        if proof.trace_length != 4096
            || proof.ram_K != 8192
            || proof.untrusted_advice_commitment.is_some()
            || proof.trace_polynomial_order != TracePolynomialOrder::CycleMajor
            || proof.rw_config
                != (JoltReadWriteConfig {
                    ram_rw_phase1_num_rounds: 12,
                    ram_rw_phase2_num_rounds: 13,
                    registers_rw_phase1_num_rounds: 12,
                    registers_rw_phase2_num_rounds: 7,
                })
            || proof.one_hot_config
                != (JoltOneHotConfig {
                    log_k_chunk: 4,
                    lookups_ra_virtual_log_k_chunk: 16,
                })
        {
            return Err(BoundaryError::Profile);
        }
        match &proof.stages.stage1_uni_skip_first_round_proof {
            SumcheckProof::Clear(ClearProof::Full(full))
                if full.round_polynomials.len() == 1
                    && full
                        .round_polynomials
                        .first()
                        .is_some_and(|round| round.coefficients().len() == 28) =>
            {
                Ok(())
            }
            SumcheckProof::Clear(ClearProof::Full(_) | ClearProof::Compressed(_))
            | SumcheckProof::Committed(_) => Err(BoundaryError::Profile),
        }
    }

    fn public(io: &JoltDevice) -> Result<BoundaryPublicInputs, BoundaryError> {
        let inputs = io
            .inputs
            .as_slice()
            .try_into()
            .map_err(|_| BoundaryError::Profile)?;
        let end = io
            .outputs
            .iter()
            .rposition(|byte| *byte != 0)
            .ok_or(BoundaryError::Profile)?;
        let [output] = io.outputs.get(..=end).ok_or(BoundaryError::Profile)? else {
            return Err(BoundaryError::Profile);
        };
        let public = BoundaryPublicInputs {
            inputs,
            output: *output,
            panic: io.panic,
        };
        let _ = public.packed()?;
        Ok(public)
    }

    /// Extract typed cells from the original proof; metadata must match this profile.
    pub fn witness<VC, Zk>(
        &self,
        io: &JoltDevice,
        proof: &JoltProof<AkitaScheme, VC, Zk>,
    ) -> Result<AkitaStage1BoundaryWitness, BoundaryError>
    where
        VC: VectorCommitment<Field = AkitaField>,
    {
        Self::check_metadata(proof)?;
        if io.memory_layout != self.checked.public_io.memory_layout {
            return Err(BoundaryError::Profile);
        }
        let SumcheckProof::Clear(ClearProof::Full(full)) =
            &proof.stages.stage1_uni_skip_first_round_proof
        else {
            return Err(BoundaryError::Profile);
        };
        let coefficients = full
            .round_polynomials
            .first()
            .ok_or(BoundaryError::Profile)?
            .coefficients()
            .iter()
            .map(|x| x.to_canonical_u128())
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| BoundaryError::Profile)?;
        let commitment = self
            .commitment
            .witness_bytes(&proof.commitments)?
            .try_into()
            .map_err(|_| BoundaryError::Profile)?;
        Ok(AkitaStage1BoundaryWitness {
            public: Self::public(io)?,
            commitment,
            coefficients,
            output_claim: proof
                .clear_claims()?
                .stage1
                .uniskip_output_claim
                .to_canonical_u128(),
        })
    }

    pub fn commitment_shape(&self) -> &AkitaCommitmentShape {
        &self.commitment
    }

    /// Public descriptor for application-owned key identities, in declared wire order.
    pub fn profile_descriptor(&self, setup_digest: [u8; 32]) -> Vec<u8> {
        let layout = &self.checked.public_io.memory_layout;
        let mut out = self.checked.preprocessing_digest.to_vec();
        out.extend(setup_digest);
        for value in [
            layout.program_size,
            layout.max_trusted_advice_size,
            layout.trusted_advice_start,
            layout.trusted_advice_end,
            layout.max_untrusted_advice_size,
            layout.untrusted_advice_start,
            layout.untrusted_advice_end,
            layout.max_input_size,
            layout.max_output_size,
            layout.input_start,
            layout.input_end,
            layout.output_start,
            layout.output_end,
            layout.stack_size,
            layout.stack_end,
            layout.heap_size,
            layout.heap_end,
            layout.panic,
            layout.termination,
            layout.io_end,
            self.checked.entry_address,
            4096,
            8192,
            12,
            13,
            12,
            7,
            4,
            16,
            0,
            28,
            22,
            1,
            16,
            8,
        ] {
            out.extend(value.to_le_bytes());
        }
        out.extend(self.commitment.layout_digest());
        out
    }

    /// Build `[ONE,p,private...]`. The outer relation must fix ONE and use exactly
    /// two public columns. This relation authenticates only through the uni-skip.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "linear-combination arithmetic builds Fr constraints; byte weights are bounded by 2^40"
    )]
    pub fn constrain(
        &self,
        builder: &mut R1csBuilder<Fr>,
        witness: Option<&AkitaStage1BoundaryWitness>,
    ) -> Result<AkitaStage1BoundaryVars, BoundaryError> {
        if builder.num_vars() != 1 {
            return Err(BoundaryError::PublicPrefix);
        }
        let public = builder.alloc_witness(witness.map(|w| w.public.packed()).transpose()?);
        let mut bytes = Vec::with_capacity(5);
        for index in 0..3 {
            bytes.push(ByteVar::allocate(
                builder,
                witness.and_then(|w| w.public.inputs.get(index)).copied(),
            ));
        }
        bytes.push(ByteVar::allocate(builder, witness.map(|w| w.public.output)));
        bytes.push(ByteVar::allocate(
            builder,
            witness.map(|w| u8::from(w.public.panic)),
        ));
        let mut packed = LinearCombination::zero();
        let mut weight = Fr::from_u64(1);
        for byte in &bytes {
            packed = packed + byte.expression().scale(weight);
            weight *= Fr::from_u64(256);
        }
        builder.assert_equal(public, packed);
        let panic = bytes.last().ok_or(BoundaryError::Profile)?;
        builder.assert_product(
            panic.expression(),
            panic.expression() - LinearCombination::one(),
            LinearCombination::zero(),
        );
        let output = bytes.get(3).ok_or(BoundaryError::Profile)?;
        let inverse = builder.alloc_witness(witness.map(|w| {
            Fr::from_u64(u64::from(w.public.output))
                .inverse()
                .unwrap_or(Fr::from_u64(0))
        }));
        builder.assert_product(output.expression(), inverse, LinearCombination::one());
        let commitment = self
            .commitment
            .allocate(builder, witness.map(|w| w.commitment.as_slice()))?;
        let coefficients = (0..28)
            .map(|i| {
                Fp128Var::allocate(
                    builder,
                    witness.and_then(|w| w.coefficients.get(i)).copied(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let claim = Fp128Var::allocate(builder, witness.map(|w| w.output_claim))?;
        let mut transcript = LegacyBlake2bVar::new(builder, b"Jolt")?;
        transcript.check_schedule(90)?;
        for (label, value) in self.checked.preamble_values(self.config) {
            let payload = match value {
                PreambleValue::Bytes(value) => {
                    transcript.append_label_with_count(
                        builder,
                        label,
                        u64::try_from(value.len()).map_err(|_| BoundaryError::Profile)?,
                    )?;
                    value.iter().copied().map(ByteVar::constant).collect()
                }
                PreambleValue::Inputs(_) => {
                    transcript.append_label_with_count(builder, label, 3)?;
                    bytes.iter().take(3).cloned().collect()
                }
                PreambleValue::Outputs(_) => {
                    transcript.append_label_with_count(builder, label, 1)?;
                    vec![output.clone()]
                }
                PreambleValue::Word(value) => {
                    transcript.append_label(builder, label)?;
                    let mut word = vec![ByteVar::constant(0); 24];
                    word.extend(value.to_be_bytes().map(ByteVar::constant));
                    word
                }
                PreambleValue::Panic(_) => {
                    transcript.append_label(builder, label)?;
                    let mut word = vec![ByteVar::constant(0); 31];
                    word.push(panic.clone());
                    word
                }
            };
            transcript.append_bytes(builder, &payload)?;
        }
        transcript.append_label(builder, b"commitment")?;
        self.commitment
            .append(builder, &mut transcript, &commitment)?;
        let uniskip = self
            .uniskip
            .constrain(builder, &mut transcript, &coefficients, &claim)?;
        Ok(AkitaStage1BoundaryVars {
            public,
            commitment,
            coefficients,
            uniskip,
            transcript,
        })
    }
}

#[cfg(all(test, feature = "prover-fixtures"))]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "recorded fixture regression checks precise native cells and malformed profiles"
)]
pub(super) mod tests {
    use super::*;
    use crate::stages::uniskip::draw_spartan_outer_tau;
    use crate::verifier::validate_and_seed_transcript;
    use blake2::{digest::consts::U32, Blake2b, Digest};
    use jolt_field::CanonicalBytes;
    use jolt_prover_legacy::zkvm::packed::{AkitaTranscript, AkitaVc};
    type AkitaJoltProof = JoltProof<AkitaScheme, AkitaVc>;
    use jolt_r1cs::{fp128_bn254::MODULUS, ConstraintMatrices};
    use jolt_sumcheck::OPENING_CLAIM_TRANSCRIPT_LABEL;
    use jolt_sumcheck::{CenteredIntegerDomain, SumcheckClaim, UNISKIP_ROUND_TRANSCRIPT_LABEL};
    use jolt_transcript::{Label, Transcript};
    use serde::de::DeserializeOwned;

    fn decode<T: DeserializeOwned>(bytes: &[u8]) -> T {
        let (value, used) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard()).unwrap();
        assert_eq!(used, bytes.len());
        value
    }
    fn fixture() -> (
        JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
        JoltDevice,
        AkitaJoltProof,
    ) {
        (
            decode(include_bytes!(
                "../../tests/fixtures/akita-boundary/preprocessing.bin"
            )),
            decode(include_bytes!(
                "../../tests/fixtures/akita-boundary/public-io.bin"
            )),
            decode(include_bytes!(
                "../../tests/fixtures/akita-boundary/proof.bin"
            )),
        )
    }

    #[test]
    fn boundary_recorded_profile_and_native_frames() {
        let (preprocessing, io, mut proof) = fixture();
        crate::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &preprocessing,
            &io,
            &proof,
            None,
        )
        .unwrap();
        let shape = AkitaStage1BoundaryShape::new(&preprocessing, &io, &proof).unwrap();
        let witness = shape.witness(&io, &proof).unwrap();
        assert_eq!(witness.public.packed().unwrap(), Fr::from_u64(0x0f03_0509));
        let (_, mut transcript) = validate_and_seed_transcript::<_, _, AkitaTranscript, _>(
            &preprocessing,
            &io,
            &proof,
            None,
        )
        .unwrap();
        let tau = draw_spartan_outer_tau(&mut transcript, 12);
        assert_eq!(tau.len(), 14);
        assert_eq!(
            tau[0].to_bytes_le_vec(),
            [245, 15, 143, 233, 20, 137, 135, 211, 194, 87, 91, 143, 187, 34, 25, 145]
        );
        let params = UniskipParams::spartan_outer();
        let reduction = proof
            .stages
            .stage1_uni_skip_first_round_proof
            .verify(
                &SumcheckClaim::new(1, params.degree(), AkitaField::from_u64(0)),
                CenteredIntegerDomain::new(params.domain_size()),
                UNISKIP_ROUND_TRANSCRIPT_LABEL,
                &mut transcript,
            )
            .unwrap();
        assert_eq!(reduction.value.to_canonical_u128(), witness.output_claim);
        proof.trace_length = 2048;
        assert!(AkitaStage1BoundaryShape::new(&preprocessing, &io, &proof).is_err());
        proof.trace_length = 4096;
        proof.untrusted_advice_commitment = Some(proof.commitments.clone());
        assert!(AkitaStage1BoundaryShape::new(&preprocessing, &io, &proof).is_err());
        let mut builder = R1csBuilder::new();
        let _ = builder.alloc_unknown();
        assert!(matches!(
            shape.constrain(&mut builder, None),
            Err(BoundaryError::PublicPrefix)
        ));
        assert_eq!(builder.num_vars(), 2);
    }

    #[test]
    fn boundary_recorded_full_native_parity_and_frozen_mutations() {
        let (preprocessing, io, proof) = fixture();
        let shape = AkitaStage1BoundaryShape::new(&preprocessing, &io, &proof).unwrap();
        let input = shape.witness(&io, &proof).unwrap();
        let (_, mut native) = validate_and_seed_transcript::<_, _, AkitaTranscript, _>(
            &preprocessing,
            &io,
            &proof,
            None,
        )
        .unwrap();
        let tau = draw_spartan_outer_tau(&mut native, 12);
        let params = UniskipParams::spartan_outer();
        let reduction = proof
            .stages
            .stage1_uni_skip_first_round_proof
            .verify(
                &SumcheckClaim::new(1, params.degree(), AkitaField::from_u64(0)),
                CenteredIntegerDomain::new(params.domain_size()),
                UNISKIP_ROUND_TRANSCRIPT_LABEL,
                &mut native,
            )
            .unwrap();
        native.append(&Label(OPENING_CLAIM_TRANSCRIPT_LABEL));
        native.append(&AkitaField::from_u128(input.output_claim));
        let mut builder = R1csBuilder::new();
        let vars = shape.constrain(&mut builder, Some(&input)).unwrap();
        assert_eq!(vars.public.index(), 1);
        let mut assignment = builder.witness().unwrap();
        for (actual, expected) in vars.uniskip.tau.iter().zip(tau) {
            assert_eq!(
                assignment[actual.variable().index()],
                Fr::from_u128(expected.to_canonical_u128())
            );
        }
        assert_eq!(
            assignment[vars.uniskip.challenge.variable().index()],
            Fr::from_u128(reduction.point[0].to_canonical_u128())
        );
        for (byte, expected) in vars.transcript.state().iter().zip(native.state()) {
            assert_eq!(
                byte.expression()
                    .terms
                    .iter()
                    .fold(Fr::from_u64(0), |sum, (variable, coefficient)| sum
                        + assignment[variable.index()] * coefficient),
                Fr::from_u64(u64::from(expected))
            );
        }
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&assignment).is_ok());
        assignment[vars.public.index()] += Fr::from_u64(1);
        assert!(matrices.check_witness(&assignment).is_err());
        assignment[vars.public.index()] -= Fr::from_u64(1);
        assignment[vars.uniskip.challenge.variable().index()] += Fr::from_u64(1);
        assert!(matrices.check_witness(&assignment).is_err());
    }
    pub(in crate::r1cs) fn fingerprint(matrices: &ConstraintMatrices<Fr>) -> [u8; 32] {
        let mut hash = Blake2b::<U32>::new();
        hash.update(u64::try_from(matrices.num_vars).unwrap().to_le_bytes());
        hash.update(
            u64::try_from(matrices.num_constraints)
                .unwrap()
                .to_le_bytes(),
        );
        for matrix in [&matrices.a, &matrices.b, &matrices.c] {
            for row in matrix {
                hash.update(u64::try_from(row.len()).unwrap().to_le_bytes());
                for (column, coefficient) in row {
                    hash.update(u64::try_from(*column).unwrap().to_le_bytes());
                    hash.update(coefficient.to_bytes_le_vec());
                }
            }
        }
        hash.finalize().into()
    }

    #[test]
    fn boundary_recorded_unknown_shape_matches_known() {
        let (preprocessing, io, proof) = fixture();
        let shape = AkitaStage1BoundaryShape::new(&preprocessing, &io, &proof).unwrap();
        let input = shape.witness(&io, &proof).unwrap();
        let known = {
            let mut builder = R1csBuilder::new();
            let _ = shape.constrain(&mut builder, Some(&input)).unwrap();
            fingerprint(&builder.into_matrices())
        };
        let mut builder = R1csBuilder::new();
        let _ = shape.constrain(&mut builder, None).unwrap();
        assert_eq!(fingerprint(&builder.into_matrices()), known);
    }

    #[test]
    fn boundary_recorded_changed_io_frozen_proof_rejects() {
        let (preprocessing, io, proof) = fixture();
        let shape = AkitaStage1BoundaryShape::new(&preprocessing, &io, &proof).unwrap();
        let mut input = shape.witness(&io, &proof).unwrap();
        input.public.inputs[0] ^= 1;
        let mut builder = R1csBuilder::new();
        let _ = shape.constrain(&mut builder, Some(&input)).unwrap();
        let assignment = builder.witness().unwrap();
        assert!(builder.into_matrices().check_witness(&assignment).is_err());
    }

    #[test]
    fn boundary_recorded_canonical_commitment_and_profile_controls() {
        let (preprocessing, io, proof) = fixture();
        let shape = AkitaStage1BoundaryShape::new(&preprocessing, &io, &proof).unwrap();
        let input = shape.witness(&io, &proof).unwrap();
        let mut builder = R1csBuilder::new();
        let vars = shape
            .commitment
            .allocate(&mut builder, Some(&input.commitment))
            .unwrap();
        let mut assignment = builder.witness().unwrap();
        let bit = vars.bytes()[0].bit_expressions()[0].terms[0].0;
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&assignment).is_ok());
        assignment[bit.index()] = Fr::from_u64(1) - assignment[bit.index()];
        assert!(matrices.check_witness(&assignment).is_err());
        let mut malformed = input.commitment;
        malformed[..16].copy_from_slice(&MODULUS.to_le_bytes());
        assert!(shape
            .commitment
            .allocate(&mut R1csBuilder::new(), Some(&malformed))
            .is_err());
        let mut device = io.clone();
        device.outputs = vec![0];
        assert!(AkitaStage1BoundaryShape::new(&preprocessing, &device, &proof).is_err());
        let mut bad = serde_json::to_value(&proof.commitments).unwrap();
        bad["backend_coeff_len"] = serde_json::json!(9);
        let bad = serde_json::from_value(bad).unwrap();
        assert!(AkitaCommitmentShape::new(&preprocessing.pcs_setup, &bad).is_err());
    }
}
