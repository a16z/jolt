use std::{
    collections::BTreeMap,
    io::{Read, Write},
};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use num::FromPrimitive;
use strum::EnumCount;

use crate::zkvm::{config::OneHotParams, witness::AllCommittedPolynomials};
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{OpeningId, OpeningPoint, Openings, ReducedOpeningProof, SumcheckId},
    },
    subprotocols::sumcheck::{SumcheckInstanceProof, UniSkipFirstRoundProof},
    transcripts::Transcript,
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};

pub struct JoltProof<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> {
    pub opening_claims: Claims<F>,
    pub commitments: Vec<PCS::Commitment>,
    pub stage1_uni_skip_first_round_proof: UniSkipFirstRoundProof<F, FS>,
    pub stage1_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage2_uni_skip_first_round_proof: UniSkipFirstRoundProof<F, FS>,
    pub stage2_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage3_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage4_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage5_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage6_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    /// Trusted advice opening proof at point from RamValEvaluation
    pub trusted_advice_val_evaluation_proof: Option<PCS::Proof>,
    /// Trusted advice opening proof at point from RamValFinalEvaluation
    pub trusted_advice_val_final_proof: Option<PCS::Proof>,
    /// Untrusted advice opening proof at point from RamValEvaluation
    pub untrusted_advice_val_evaluation_proof: Option<PCS::Proof>,
    /// Untrusted advice opening proof at point from RamValFinalEvaluation
    pub untrusted_advice_val_final_proof: Option<PCS::Proof>,
    pub reduced_opening_proof: ReducedOpeningProof<F, PCS, FS>, // Stage 7
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub bytecode_K: usize,
    pub log_k_chunk: usize,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> CanonicalSerialize
    for JoltProof<F, PCS, FS>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // serialize ram_K, bytecode_K, log_chunk first
        self.ram_K.serialize_with_mode(&mut writer, compress)?;
        self.bytecode_K.serialize_with_mode(&mut writer, compress)?;
        self.log_k_chunk
            .serialize_with_mode(&mut writer, compress)?;
        let one_hot_params =
            OneHotParams::new_with_log_k_chunk(self.log_k_chunk, self.bytecode_K, self.ram_K);
        // ensure that all committed polys are set up before serializing proofs
        let _guard = AllCommittedPolynomials::initialize(&one_hot_params);
        self.opening_claims
            .serialize_with_mode(&mut writer, compress)?;
        self.commitments
            .serialize_with_mode(&mut writer, compress)?;
        self.untrusted_advice_commitment
            .serialize_with_mode(&mut writer, compress)?;
        self.stage1_uni_skip_first_round_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage1_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2_uni_skip_first_round_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage3_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage4_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage5_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage6_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.trusted_advice_val_evaluation_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.trusted_advice_val_final_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.untrusted_advice_val_evaluation_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.untrusted_advice_val_final_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.reduced_opening_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.trace_length
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        self.opening_claims.serialized_size(compress)
            + self.commitments.serialized_size(compress)
            + self.untrusted_advice_commitment.serialized_size(compress)
            + self
                .stage1_uni_skip_first_round_proof
                .serialized_size(compress)
            + self.stage1_sumcheck_proof.serialized_size(compress)
            + self
                .stage2_uni_skip_first_round_proof
                .serialized_size(compress)
            + self.stage2_sumcheck_proof.serialized_size(compress)
            + self.stage3_sumcheck_proof.serialized_size(compress)
            + self.stage4_sumcheck_proof.serialized_size(compress)
            + self.stage5_sumcheck_proof.serialized_size(compress)
            + self.stage6_sumcheck_proof.serialized_size(compress)
            + self
                .trusted_advice_val_evaluation_proof
                .serialized_size(compress)
            + self
                .trusted_advice_val_final_proof
                .serialized_size(compress)
            + self
                .untrusted_advice_val_evaluation_proof
                .serialized_size(compress)
            + self
                .untrusted_advice_val_final_proof
                .serialized_size(compress)
            + self.reduced_opening_proof.serialized_size(compress)
            + self.trace_length.serialized_size(compress)
            + self.ram_K.serialized_size(compress)
            + self.bytecode_K.serialized_size(compress)
            + self.log_k_chunk.serialized_size(compress)
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> Valid
    for JoltProof<F, PCS, FS>
{
    fn check(&self) -> Result<(), SerializationError> {
        self.opening_claims.check()?;
        self.commitments.check()?;
        self.untrusted_advice_commitment.check()?;
        self.stage1_uni_skip_first_round_proof.check()?;
        self.stage1_sumcheck_proof.check()?;
        self.stage2_uni_skip_first_round_proof.check()?;
        self.stage2_sumcheck_proof.check()?;
        self.stage3_sumcheck_proof.check()?;
        self.stage4_sumcheck_proof.check()?;
        self.stage5_sumcheck_proof.check()?;
        self.stage6_sumcheck_proof.check()?;
        self.trusted_advice_val_evaluation_proof.check()?;
        self.trusted_advice_val_final_proof.check()?;
        self.untrusted_advice_val_evaluation_proof.check()?;
        self.untrusted_advice_val_final_proof.check()?;
        self.reduced_opening_proof.check()?;
        self.trace_length.check()?;
        self.ram_K.check()?;
        self.bytecode_K.check()?;
        self.log_k_chunk.check()?;
        Ok(())
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> CanonicalDeserialize
    for JoltProof<F, PCS, FS>
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let ram_K = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let bytecode_K = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let log_chunk = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let one_hot_params = OneHotParams::new_with_log_k_chunk(log_chunk, bytecode_K, ram_K);
        // ensure that all committed polys are set up before deserializing proofs
        let _guard = AllCommittedPolynomials::initialize(&one_hot_params);
        let opening_claims = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let commitments = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let untrusted_advice_commitment =
            <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage1_uni_skip_first_round =
            <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage1_sumcheck = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage2_uni_skip_first_round =
            <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage2_sumcheck_proof = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage3_sumcheck_proof = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage4_sumcheck_proof = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage5_sumcheck_proof = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage6_sumcheck_proof = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let trusted_advice_val_evaluation_proof =
            <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let trusted_advice_val_final_proof =
            <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let untrusted_advice_val_evaluation_proof =
            <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let untrusted_advice_val_final_proof =
            <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let reduced_opening_proof = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        let trace_length = <_>::deserialize_with_mode(&mut reader, compress, validate)?;
        // drop(guard);

        Ok(Self {
            opening_claims,
            commitments,
            untrusted_advice_commitment,
            stage1_uni_skip_first_round_proof: stage1_uni_skip_first_round,
            stage1_sumcheck_proof: stage1_sumcheck,
            stage2_uni_skip_first_round_proof: stage2_uni_skip_first_round,
            stage2_sumcheck_proof,
            stage3_sumcheck_proof,
            stage4_sumcheck_proof,
            stage5_sumcheck_proof,
            stage6_sumcheck_proof,
            trusted_advice_val_evaluation_proof,
            trusted_advice_val_final_proof,
            untrusted_advice_val_evaluation_proof,
            untrusted_advice_val_final_proof,
            reduced_opening_proof,
            trace_length,
            ram_K,
            bytecode_K,
            log_k_chunk: log_chunk,
        })
    }
}

pub struct Claims<F: JoltField>(pub Openings<F>);

impl<F: JoltField> CanonicalSerialize for Claims<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.len().serialize_with_mode(&mut writer, compress)?;
        for (key, (_opening_point, claim)) in self.0.iter() {
            key.serialize_with_mode(&mut writer, compress)?;
            claim.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = self.0.len().serialized_size(compress);
        for (key, (_opening_point, claim)) in self.0.iter() {
            size += key.serialized_size(compress);
            size += claim.serialized_size(compress);
        }
        size
    }
}

impl<F: JoltField> Valid for Claims<F> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField> CanonicalDeserialize for Claims<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let size = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let mut claims = BTreeMap::new();
        for _ in 0..size {
            let key = OpeningId::deserialize_with_mode(&mut reader, compress, validate)?;
            let claim = F::deserialize_with_mode(&mut reader, compress, validate)?;
            claims.insert(key, (OpeningPoint::default(), claim));
        }

        Ok(Claims(claims))
    }
}

// Compact encoding for OpeningId:
// Each variant uses a fused byte = BASE + sumcheck_id (1 byte total for advice, 2 bytes for committed/virtual)
// - [0, NUM_SUMCHECKS) = UntrustedAdvice(sumcheck_id)
// - [NUM_SUMCHECKS, 2*NUM_SUMCHECKS) = TrustedAdvice(sumcheck_id)
// - [2*NUM_SUMCHECKS, 3*NUM_SUMCHECKS) + poly_index = Committed(poly, sumcheck_id)
// - [3*NUM_SUMCHECKS, 4*NUM_SUMCHECKS) + poly_index = Virtual(poly, sumcheck_id)
const OPENING_ID_UNTRUSTED_ADVICE_BASE: u8 = 0;
const OPENING_ID_TRUSTED_ADVICE_BASE: u8 =
    OPENING_ID_UNTRUSTED_ADVICE_BASE + SumcheckId::COUNT as u8;
const OPENING_ID_COMMITTED_BASE: u8 = OPENING_ID_TRUSTED_ADVICE_BASE + SumcheckId::COUNT as u8;
const OPENING_ID_VIRTUAL_BASE: u8 = OPENING_ID_COMMITTED_BASE + SumcheckId::COUNT as u8;

impl CanonicalSerialize for OpeningId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            OpeningId::UntrustedAdvice(sumcheck_id) => {
                let fused = OPENING_ID_UNTRUSTED_ADVICE_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::TrustedAdvice(sumcheck_id) => {
                let fused = OPENING_ID_TRUSTED_ADVICE_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Committed(committed_polynomial, sumcheck_id) => {
                let fused = OPENING_ID_COMMITTED_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)?;
                committed_polynomial.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Virtual(virtual_polynomial, sumcheck_id) => {
                let fused = OPENING_ID_VIRTUAL_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)?;
                virtual_polynomial.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            OpeningId::UntrustedAdvice(_) | OpeningId::TrustedAdvice(_) => 1,
            OpeningId::Committed(committed_polynomial, _) => {
                // 1 byte fused (variant + sumcheck_id) + poly index
                1 + committed_polynomial.serialized_size(compress)
            }
            OpeningId::Virtual(virtual_polynomial, _) => {
                // 1 byte fused (variant + sumcheck_id) + poly index
                1 + virtual_polynomial.serialized_size(compress)
            }
        }
    }
}

impl Valid for OpeningId {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for OpeningId {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let fused = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match fused {
            _ if fused < OPENING_ID_TRUSTED_ADVICE_BASE => {
                let sumcheck_id = fused - OPENING_ID_UNTRUSTED_ADVICE_BASE;
                Ok(OpeningId::UntrustedAdvice(
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ if fused < OPENING_ID_COMMITTED_BASE => {
                let sumcheck_id = fused - OPENING_ID_TRUSTED_ADVICE_BASE;
                Ok(OpeningId::TrustedAdvice(
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ if fused < OPENING_ID_VIRTUAL_BASE => {
                let sumcheck_id = fused - OPENING_ID_COMMITTED_BASE;
                let polynomial =
                    CommittedPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::Committed(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ => {
                let sumcheck_id = fused - OPENING_ID_VIRTUAL_BASE;
                let polynomial =
                    VirtualPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::Virtual(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
        }
    }
}

impl CanonicalSerialize for CommittedPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let index = self.to_index();
        debug_assert!(
            index <= u8::MAX as usize,
            "CommittedPolynomial index {index} exceeds u8::MAX"
        );
        (index as u8).serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        1 // u8
    }
}

impl Valid for CommittedPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for CommittedPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let index = u8::deserialize_with_mode(&mut reader, compress, validate)? as usize;
        Ok(CommittedPolynomial::from_index(index))
    }
}

impl CanonicalSerialize for VirtualPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let index = self.to_index();
        debug_assert!(
            index <= u8::MAX as usize,
            "VirtualPolynomial index {index} exceeds u8::MAX"
        );
        (index as u8).serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        1 // u8
    }
}

impl Valid for VirtualPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for VirtualPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let index = u8::deserialize_with_mode(&mut reader, compress, validate)? as usize;
        Ok(VirtualPolynomial::from_index(index))
    }
}

pub fn serialize_and_print_size(
    item_name: &str,
    file_name: &str,
    item: &impl CanonicalSerialize,
) -> Result<(), SerializationError> {
    use std::fs::File;
    let mut file = File::create(file_name)?;
    item.serialize_compressed(&mut file)?;
    let file_size_bytes = file.metadata()?.len();
    let file_size_kb = file_size_bytes as f64 / 1024.0;
    tracing::info!("{item_name} Written to {file_name}");
    tracing::info!("{item_name} size: {file_size_kb:.1} kB");
    Ok(())
}
