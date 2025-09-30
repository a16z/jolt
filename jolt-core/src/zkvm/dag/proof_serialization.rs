use std::{
    cell::RefCell,
    collections::BTreeMap,
    io::{Read, Write},
    rc::Rc,
};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use num::FromPrimitive;
use tracer::JoltDevice;

use crate::zkvm::witness::AllCommittedPolynomials;
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{
            OpeningId, OpeningPoint, Openings, ReducedOpeningProof, SumcheckId,
            VerifierOpeningAccumulator,
        },
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    zkvm::{
        dag::state_manager::{ProofData, ProofKeys, Proofs, StateManager, VerifierState},
        witness::{CommittedPolynomial, VirtualPolynomial},
        JoltVerifierPreprocessing,
    },
};

pub struct JoltProof<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> {
    opening_claims: Claims<F>,
    pub commitments: Vec<PCS::Commitment>,
    pub proofs: Proofs<F, PCS, FS>,
    pub trace_length: usize,
    ram_K: usize,
    bytecode_d: usize,
    twist_sumcheck_switch_index: usize,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> CanonicalSerialize
    for JoltProof<F, PCS, FS>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // serialize ram_K and bytecode_d first
        self.ram_K.serialize_with_mode(&mut writer, compress)?;
        self.bytecode_d.serialize_with_mode(&mut writer, compress)?;
        // ensure that all committed polys are set up before serializing proofs
        let _guard = AllCommittedPolynomials::initialize(self.ram_K, self.bytecode_d);
        self.opening_claims
            .serialize_with_mode(&mut writer, compress)?;
        self.commitments
            .serialize_with_mode(&mut writer, compress)?;
        self.proofs.serialize_with_mode(&mut writer, compress)?;
        self.trace_length
            .serialize_with_mode(&mut writer, compress)?;
        self.twist_sumcheck_switch_index
            .serialize_with_mode(&mut writer, compress)?;

        // drop(guard);
        Ok(())
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        self.opening_claims.serialized_size(compress)
            + self.commitments.serialized_size(compress)
            + self.proofs.serialized_size(compress)
            + self.trace_length.serialized_size(compress)
            + self.ram_K.serialized_size(compress)
            + self.bytecode_d.serialized_size(compress)
            + self.twist_sumcheck_switch_index.serialized_size(compress)
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> Valid
    for JoltProof<F, PCS, FS>
{
    fn check(&self) -> Result<(), SerializationError> {
        self.opening_claims.check()?;
        self.commitments.check()?;
        self.proofs.check()?;
        self.trace_length.check()?;
        self.ram_K.check()?;
        self.bytecode_d.check()?;
        self.twist_sumcheck_switch_index.check()?;
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
        let ram_K = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let bytecode_d = usize::deserialize_with_mode(&mut reader, compress, validate)?;

        // ensure that all committed polys are set up before deserializing proofs
        let _guard = AllCommittedPolynomials::initialize(ram_K, bytecode_d);
        let opening_claims = Claims::deserialize_with_mode(&mut reader, compress, validate)?;
        let commitments =
            Vec::<PCS::Commitment>::deserialize_with_mode(&mut reader, compress, validate)?;
        let proofs = Proofs::<F, PCS, FS>::deserialize_with_mode(&mut reader, compress, validate)?;
        let trace_length = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let twist_sumcheck_switch_index =
            usize::deserialize_with_mode(&mut reader, compress, validate)?;
        // drop(guard);

        Ok(Self {
            opening_claims,
            commitments,
            proofs,
            trace_length,
            ram_K,
            bytecode_d,
            twist_sumcheck_switch_index,
        })
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> JoltProof<F, PCS, FS> {
    pub fn from_prover_state_manager(mut state_manager: StateManager<'_, F, FS, PCS>) -> Self {
        let prover_state = state_manager.prover_state.as_mut().unwrap();
        let openings = std::mem::take(&mut prover_state.accumulator.borrow_mut().openings);
        let commitments = state_manager.commitments.take();
        let proofs = state_manager.proofs.take();
        let trace_length = prover_state.trace.len();
        let ram_K = state_manager.ram_K;
        let twist_sumcheck_switch_index = state_manager.twist_sumcheck_switch_index;

        Self {
            opening_claims: Claims(openings),
            commitments,
            proofs,
            trace_length,
            ram_K,
            bytecode_d: prover_state.preprocessing.shared.bytecode.d,
            twist_sumcheck_switch_index,
        }
    }

    pub fn to_verifier_state_manager<'a>(
        self,
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        program_io: JoltDevice,
    ) -> StateManager<'a, F, FS, PCS> {
        let mut opening_accumulator = VerifierOpeningAccumulator::<F>::new();
        // Populate claims in the verifier accumulator
        for (key, (_, claim)) in self.opening_claims.0.iter() {
            opening_accumulator
                .openings_mut()
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        let proofs = Rc::new(RefCell::new(self.proofs));
        let commitments = Rc::new(RefCell::new(self.commitments));
        let transcript = Rc::new(RefCell::new(FS::new(b"Jolt")));

        StateManager {
            transcript,
            proofs,
            commitments,
            program_io,
            ram_K: self.ram_K,
            twist_sumcheck_switch_index: self.twist_sumcheck_switch_index,
            prover_state: None,
            verifier_state: Some(VerifierState {
                preprocessing,
                trace_length: self.trace_length,
                accumulator: Rc::new(RefCell::new(opening_accumulator)),
            }),
        }
    }
}

pub struct Claims<F: JoltField>(Openings<F>);

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

impl CanonicalSerialize for OpeningId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            OpeningId::Committed(committed_polynomial, sumcheck_id) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                (*sumcheck_id as u8).serialize_with_mode(&mut writer, compress)?;
                committed_polynomial.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Virtual(virtual_polynomial, sumcheck_id) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                (*sumcheck_id as u8).serialize_with_mode(&mut writer, compress)?;
                virtual_polynomial.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            OpeningId::Committed(committed_polynomial, _) => {
                // +1 for OpeningIdVariant, +1 for sumcheck_id (which is a u8)
                committed_polynomial.serialized_size(compress) + 2
            }
            OpeningId::Virtual(virtual_polynomial, _) => {
                // +1 for OpeningIdVariant, +1 for sumcheck_id (which is a u8)
                virtual_polynomial.serialized_size(compress) + 2
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
        let opening_type = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let sumcheck_id = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match opening_type {
            0 => {
                let polynomial =
                    CommittedPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::Committed(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            1 => {
                let polynomial =
                    VirtualPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::Virtual(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl CanonicalSerialize for CommittedPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.to_index().serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.to_index().serialized_size(compress)
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
        let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(CommittedPolynomial::from_index(index))
    }
}

impl CanonicalSerialize for VirtualPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.to_index().serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.to_index().serialized_size(compress)
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
        let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(VirtualPolynomial::from_index(index))
    }
}

impl CanonicalSerialize for ProofKeys {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (*self as u8).serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        (*self as u8).serialized_size(compress)
    }
}

impl Valid for ProofKeys {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for ProofKeys {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let variant = u8::deserialize_with_mode(reader, compress, validate)?;
        ProofKeys::from_u8(variant).ok_or(SerializationError::InvalidData)
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> CanonicalSerialize
    for ProofData<F, PCS, FS>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            ProofData::SumcheckProof(proof) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(&mut writer, compress)
            }
            ProofData::ReducedOpeningProof(proof) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            ProofData::SumcheckProof(proof) => proof.serialized_size(compress),
            ProofData::ReducedOpeningProof(proof) => proof.serialized_size(compress),
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> Valid
    for ProofData<F, PCS, FS>
{
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> CanonicalDeserialize
    for ProofData<F, PCS, FS>
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match variant {
            0 => {
                let proof =
                    SumcheckInstanceProof::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(ProofData::SumcheckProof(proof))
            }
            1 => {
                let proof =
                    ReducedOpeningProof::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(ProofData::ReducedOpeningProof(proof))
            }
            _ => Err(SerializationError::InvalidData),
        }
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
