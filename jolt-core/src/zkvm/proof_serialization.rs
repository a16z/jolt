#[cfg(not(feature = "zk"))]
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use num::FromPrimitive;

#[cfg(not(feature = "zk"))]
use crate::poly::opening_proof::{OpeningPoint, Openings};
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::BlindFoldProof;
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, dory::DoryLayout},
        opening_proof::{OpeningId, PolynomialId, SumcheckId},
    },
};
use crate::{
    subprotocols::{
        sumcheck::SumcheckInstanceProof, univariate_skip::UniSkipFirstRoundProofVariant,
    },
    transcripts::Transcript,
    zkvm::{
        config::{OneHotConfig, ReadWriteConfig},
        instruction::{CircuitFlags, InstructionFlags},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

use crate::zkvm::transport;

const PROOF_MAGIC: &[u8; 7] = b"JOLTPRF";
const PROOF_VERSION: u8 = 1;

const MAX_PARAMS_LEN: u64 = 1024;
const MAX_SECTION_LEN: u64 = 128 * 1024;

pub struct JoltProof<F: JoltField, C: JoltCurve, PCS: CommitmentScheme<Field = F>, FS: Transcript> {
    pub commitments: Vec<PCS::Commitment>,
    pub stage1_uni_skip_first_round_proof: UniSkipFirstRoundProofVariant<F, C, FS>,
    pub stage1_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage2_uni_skip_first_round_proof: UniSkipFirstRoundProofVariant<F, C, FS>,
    pub stage2_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage3_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage4_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage5_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage6_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage7_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    #[cfg(feature = "zk")]
    pub blindfold_proof: BlindFoldProof<F, C>,
    pub joint_opening_proof: PCS::Proof,
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    #[cfg(not(feature = "zk"))]
    pub opening_claims: Claims<F>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: ReadWriteConfig,
    pub one_hot_config: OneHotConfig,
    pub dory_layout: DoryLayout,
}

#[inline]
fn io_err(e: std::io::Error) -> SerializationError {
    SerializationError::IoError(e)
}

macro_rules! write_section {
    ($w:expr, $c:expr, $($item:expr),+ $(,)?) => {{
        let len: u64 = 0 $(+ $item.serialized_size($c) as u64)+;
        transport::write_varint_u64($w, len).map_err(io_err)?;
        $($item.serialize_with_mode($w, $c)?;)+
    }};
}

macro_rules! section_size {
    ($c:expr, $($item:expr),+ $(,)?) => {{
        let payload: u64 = 0 $(+ $item.serialized_size($c) as u64)+;
        transport::varint_u64_len(payload) + payload as usize
    }};
}

fn check_trailing_bytes(limited: &std::io::Take<&mut impl Read>) -> Result<(), SerializationError> {
    if limited.limit() != 0 {
        return Err(io_err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} trailing bytes not consumed", limited.limit()),
        )));
    }
    Ok(())
}

impl<F: JoltField, C: JoltCurve, PCS: CommitmentScheme<Field = F>, FS: Transcript>
    JoltProof<F, C, PCS, FS>
{
    fn params_payload_len(&self, compress: Compress) -> u64 {
        (transport::varint_u64_len(self.trace_length as u64)
            + transport::varint_u64_len(self.ram_K as u64)) as u64
            + self.rw_config.serialized_size(compress) as u64
            + self.one_hot_config.serialized_size(compress) as u64
            + self.dory_layout.serialized_size(compress) as u64
    }

    fn commitments_payload_len(&self, compress: Compress) -> u64 {
        let count_len = transport::varint_u64_len(self.commitments.len() as u64) as u64;
        let items_len: u64 = self
            .commitments
            .iter()
            .map(|c| c.serialized_size(compress) as u64)
            .sum();
        let untrusted_len = self
            .untrusted_advice_commitment
            .as_ref()
            .map(|c| c.serialized_size(compress) as u64)
            .unwrap_or(0);
        count_len + items_len + 1 + untrusted_len
    }

    #[cfg(not(feature = "zk"))]
    fn claims_payload_len(&self, compress: Compress) -> u64 {
        let count_len = transport::varint_u64_len(self.opening_claims.0.len() as u64) as u64;
        let items_len: u64 = self
            .opening_claims
            .0
            .iter()
            .map(|(k, (_p, claim))| {
                (k.serialized_size(compress) + claim.serialized_size(compress)) as u64
            })
            .sum();
        count_len + items_len
    }
}

impl<F: JoltField, C: JoltCurve, PCS: CommitmentScheme<Field = F>, FS: Transcript>
    CanonicalSerialize for JoltProof<F, C, PCS, FS>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        transport::write_magic_version(&mut writer, PROOF_MAGIC, PROOF_VERSION).map_err(io_err)?;

        let params_len = self.params_payload_len(compress);
        transport::write_varint_u64(&mut writer, params_len).map_err(io_err)?;
        transport::write_varint_u64(&mut writer, self.trace_length as u64).map_err(io_err)?;
        transport::write_varint_u64(&mut writer, self.ram_K as u64).map_err(io_err)?;
        self.rw_config.serialize_with_mode(&mut writer, compress)?;
        self.one_hot_config
            .serialize_with_mode(&mut writer, compress)?;
        self.dory_layout
            .serialize_with_mode(&mut writer, compress)?;

        let commitments_len = self.commitments_payload_len(compress);
        transport::write_varint_u64(&mut writer, commitments_len).map_err(io_err)?;
        transport::write_varint_u64(&mut writer, self.commitments.len() as u64).map_err(io_err)?;
        for c in &self.commitments {
            c.serialize_with_mode(&mut writer, compress)?;
        }
        match &self.untrusted_advice_commitment {
            None => writer.write_all(&[0]).map_err(io_err)?,
            Some(c) => {
                writer.write_all(&[1]).map_err(io_err)?;
                c.serialize_with_mode(&mut writer, compress)?;
            }
        }

        #[cfg(not(feature = "zk"))]
        {
            let claims_len = self.claims_payload_len(compress);
            transport::write_varint_u64(&mut writer, claims_len).map_err(io_err)?;
            transport::write_varint_u64(&mut writer, self.opening_claims.0.len() as u64)
                .map_err(io_err)?;
            for (k, (_p, claim)) in self.opening_claims.0.iter() {
                k.serialize_with_mode(&mut writer, compress)?;
                claim.serialize_with_mode(&mut writer, compress)?;
            }
        }

        write_section!(
            &mut writer,
            compress,
            &self.stage1_uni_skip_first_round_proof,
            &self.stage1_sumcheck_proof
        );
        write_section!(
            &mut writer,
            compress,
            &self.stage2_uni_skip_first_round_proof,
            &self.stage2_sumcheck_proof
        );
        write_section!(&mut writer, compress, &self.stage3_sumcheck_proof);
        write_section!(&mut writer, compress, &self.stage4_sumcheck_proof);
        write_section!(&mut writer, compress, &self.stage5_sumcheck_proof);
        write_section!(&mut writer, compress, &self.stage6_sumcheck_proof);
        write_section!(&mut writer, compress, &self.stage7_sumcheck_proof);
        write_section!(&mut writer, compress, &self.joint_opening_proof);

        #[cfg(feature = "zk")]
        write_section!(&mut writer, compress, &self.blindfold_proof);

        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = PROOF_MAGIC.len() + 1; // magic + version byte

        let params_len = self.params_payload_len(compress);
        size += transport::varint_u64_len(params_len) + params_len as usize;

        let commitments_len = self.commitments_payload_len(compress);
        size += transport::varint_u64_len(commitments_len) + commitments_len as usize;

        #[cfg(not(feature = "zk"))]
        {
            let claims_len = self.claims_payload_len(compress);
            size += transport::varint_u64_len(claims_len) + claims_len as usize;
        }

        size += section_size!(
            compress,
            &self.stage1_uni_skip_first_round_proof,
            &self.stage1_sumcheck_proof
        );
        size += section_size!(
            compress,
            &self.stage2_uni_skip_first_round_proof,
            &self.stage2_sumcheck_proof
        );
        size += section_size!(compress, &self.stage3_sumcheck_proof);
        size += section_size!(compress, &self.stage4_sumcheck_proof);
        size += section_size!(compress, &self.stage5_sumcheck_proof);
        size += section_size!(compress, &self.stage6_sumcheck_proof);
        size += section_size!(compress, &self.stage7_sumcheck_proof);
        size += section_size!(compress, &self.joint_opening_proof);

        #[cfg(feature = "zk")]
        {
            size += section_size!(compress, &self.blindfold_proof);
        }

        size
    }
}

impl<F: JoltField, C: JoltCurve, PCS: CommitmentScheme<Field = F>, FS: Transcript> Valid
    for JoltProof<F, C, PCS, FS>
{
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField, C: JoltCurve, PCS: CommitmentScheme<Field = F>, FS: Transcript>
    CanonicalDeserialize for JoltProof<F, C, PCS, FS>
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let version = transport::read_magic_version(&mut reader, PROOF_MAGIC).map_err(io_err)?;
        if version != PROOF_VERSION {
            return Err(io_err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported proof version {version}, expected {PROOF_VERSION}"),
            )));
        }

        // Params
        let mut limited = transport::read_section(&mut reader, MAX_PARAMS_LEN).map_err(io_err)?;
        let t = transport::read_varint_u64(&mut limited).map_err(io_err)?;
        let r = transport::read_varint_u64(&mut limited).map_err(io_err)?;
        let trace_length = usize::try_from(t).map_err(|_| SerializationError::InvalidData)?;
        if trace_length == 0 {
            return Err(io_err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "trace_length must be nonzero",
            )));
        }
        let ram_K = usize::try_from(r).map_err(|_| SerializationError::InvalidData)?;
        let rw_config = ReadWriteConfig::deserialize_with_mode(&mut limited, compress, validate)?;
        let one_hot_config = OneHotConfig::deserialize_with_mode(&mut limited, compress, validate)?;
        let dory_layout = DoryLayout::deserialize_with_mode(&mut limited, compress, validate)?;
        check_trailing_bytes(&limited)?;

        // Commitments
        let mut limited = transport::read_section(&mut reader, MAX_SECTION_LEN).map_err(io_err)?;
        let n = transport::read_varint_u64(&mut limited).map_err(io_err)?;
        let n_usize = usize::try_from(n).map_err(|_| SerializationError::InvalidData)?;
        if n_usize > 10_000 {
            return Err(SerializationError::InvalidData);
        }
        let mut commitments = Vec::with_capacity(n_usize);
        for _ in 0..n_usize {
            commitments.push(PCS::Commitment::deserialize_with_mode(
                &mut limited,
                compress,
                validate,
            )?);
        }
        let presence = u8::deserialize_with_mode(&mut limited, compress, validate)?;
        let untrusted_advice_commitment = match presence {
            0 => None,
            1 => Some(PCS::Commitment::deserialize_with_mode(
                &mut limited,
                compress,
                validate,
            )?),
            _ => return Err(SerializationError::InvalidData),
        };
        check_trailing_bytes(&limited)?;

        // Opening claims (non-ZK only)
        #[cfg(not(feature = "zk"))]
        let opening_claims = {
            let mut limited =
                transport::read_section(&mut reader, MAX_SECTION_LEN).map_err(io_err)?;
            let n = transport::read_varint_u64(&mut limited).map_err(io_err)?;
            let n_usize = usize::try_from(n).map_err(|_| SerializationError::InvalidData)?;
            if n_usize > 10_000 {
                return Err(SerializationError::InvalidData);
            }
            let mut claims = BTreeMap::new();
            for _ in 0..n_usize {
                let key = OpeningId::deserialize_with_mode(&mut limited, compress, validate)?;
                let claim = F::deserialize_with_mode(&mut limited, compress, validate)?;
                if claims
                    .insert(key, (OpeningPoint::default(), claim))
                    .is_some()
                {
                    return Err(SerializationError::InvalidData);
                }
            }
            check_trailing_bytes(&limited)?;
            Claims(claims)
        };

        // Stage 1
        let mut limited = transport::read_section(&mut reader, MAX_SECTION_LEN).map_err(io_err)?;
        let stage1_uni_skip_first_round_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut limited, compress, validate)?;
        let stage1_sumcheck_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut limited, compress, validate)?;
        check_trailing_bytes(&limited)?;

        // Stage 2
        let mut limited = transport::read_section(&mut reader, MAX_SECTION_LEN).map_err(io_err)?;
        let stage2_uni_skip_first_round_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut limited, compress, validate)?;
        let stage2_sumcheck_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut limited, compress, validate)?;
        check_trailing_bytes(&limited)?;

        // Stages 3-7
        macro_rules! read_single_section {
            ($reader:expr) => {{
                let mut limited =
                    transport::read_section($reader, MAX_SECTION_LEN).map_err(io_err)?;
                let val =
                    CanonicalDeserialize::deserialize_with_mode(&mut limited, compress, validate)?;
                check_trailing_bytes(&limited)?;
                val
            }};
        }

        let stage3_sumcheck_proof = read_single_section!(&mut reader);
        let stage4_sumcheck_proof = read_single_section!(&mut reader);
        let stage5_sumcheck_proof = read_single_section!(&mut reader);
        let stage6_sumcheck_proof = read_single_section!(&mut reader);
        let stage7_sumcheck_proof = read_single_section!(&mut reader);
        let joint_opening_proof = read_single_section!(&mut reader);

        #[cfg(feature = "zk")]
        let blindfold_proof = read_single_section!(&mut reader);

        Ok(Self {
            commitments,
            stage1_uni_skip_first_round_proof,
            stage1_sumcheck_proof,
            stage2_uni_skip_first_round_proof,
            stage2_sumcheck_proof,
            stage3_sumcheck_proof,
            stage4_sumcheck_proof,
            stage5_sumcheck_proof,
            stage6_sumcheck_proof,
            stage7_sumcheck_proof,
            #[cfg(feature = "zk")]
            blindfold_proof,
            joint_opening_proof,
            untrusted_advice_commitment,
            #[cfg(not(feature = "zk"))]
            opening_claims,
            trace_length,
            ram_K,
            rw_config,
            one_hot_config,
            dory_layout,
        })
    }
}

#[cfg(not(feature = "zk"))]
pub struct Claims<F: JoltField>(pub Openings<F>);

#[cfg(not(feature = "zk"))]
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

#[cfg(not(feature = "zk"))]
impl<F: JoltField> Valid for Claims<F> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[cfg(not(feature = "zk"))]
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
            if claims
                .insert(key, (OpeningPoint::default(), claim))
                .is_some()
            {
                return Err(SerializationError::InvalidData);
            }
        }

        Ok(Claims(claims))
    }
}

impl CanonicalSerialize for DoryLayout {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        u8::from(*self).serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        u8::from(*self).serialized_size(compress)
    }
}

impl Valid for DoryLayout {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for DoryLayout {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u8::deserialize_with_mode(reader, compress, validate)?;
        if value > 1 {
            return Err(SerializationError::InvalidData);
        }
        Ok(DoryLayout::from(value))
    }
}

// OpeningId wire encoding (packed, self-describing):
//
// Header byte:
//   bits[7..6] = kind (0..=3)
//   bits[5..0] = sumcheck_id if < 63, otherwise 63 as an escape
//
// If bits[5..0] == 63:
//   next bytes = varint u64 sumcheck_id (must fit in u8 for current SumcheckId)
//
// If kind indicates a polynomial:
//   append polynomial id bytes (CommittedPolynomial / VirtualPolynomial)
const OPENING_ID_KIND_UNTRUSTED_ADVICE: u8 = 0;
const OPENING_ID_KIND_TRUSTED_ADVICE: u8 = 1;
const OPENING_ID_KIND_COMMITTED: u8 = 2;
const OPENING_ID_KIND_VIRTUAL: u8 = 3;
const OPENING_ID_SUMCHECK_ESCAPE: u8 = 63;

impl CanonicalSerialize for OpeningId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let (kind, sumcheck_u64, poly) = match *self {
            OpeningId::UntrustedAdvice(sumcheck_id) => {
                (OPENING_ID_KIND_UNTRUSTED_ADVICE, sumcheck_id as u64, None)
            }
            OpeningId::TrustedAdvice(sumcheck_id) => {
                (OPENING_ID_KIND_TRUSTED_ADVICE, sumcheck_id as u64, None)
            }
            OpeningId::Polynomial(PolynomialId::Committed(committed_polynomial), sumcheck_id) => (
                OPENING_ID_KIND_COMMITTED,
                sumcheck_id as u64,
                Some(PolynomialId::Committed(committed_polynomial)),
            ),
            OpeningId::Polynomial(PolynomialId::Virtual(virtual_polynomial), sumcheck_id) => (
                OPENING_ID_KIND_VIRTUAL,
                sumcheck_id as u64,
                Some(PolynomialId::Virtual(virtual_polynomial)),
            ),
        };

        let header = if sumcheck_u64 < OPENING_ID_SUMCHECK_ESCAPE as u64 {
            (kind << 6) | (sumcheck_u64 as u8)
        } else {
            (kind << 6) | OPENING_ID_SUMCHECK_ESCAPE
        };
        header.serialize_with_mode(&mut writer, compress)?;
        if (header & 0x3F) == OPENING_ID_SUMCHECK_ESCAPE {
            transport::write_varint_u64(&mut writer, sumcheck_u64).map_err(io_err)?;
        }
        if let Some(poly) = poly {
            match poly {
                PolynomialId::Committed(p) => p.serialize_with_mode(&mut writer, compress)?,
                PolynomialId::Virtual(p) => p.serialize_with_mode(&mut writer, compress)?,
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let sumcheck_u64 = match self {
            OpeningId::UntrustedAdvice(sumcheck_id)
            | OpeningId::TrustedAdvice(sumcheck_id)
            | OpeningId::Polynomial(_, sumcheck_id) => *sumcheck_id as u64,
        };
        let header_len = 1usize;
        let sumcheck_ext_len = if sumcheck_u64 < OPENING_ID_SUMCHECK_ESCAPE as u64 {
            0usize
        } else {
            transport::varint_u64_len(sumcheck_u64)
        };
        let poly_len = match self {
            OpeningId::UntrustedAdvice(_) | OpeningId::TrustedAdvice(_) => 0,
            OpeningId::Polynomial(PolynomialId::Committed(p), _) => p.serialized_size(compress),
            OpeningId::Polynomial(PolynomialId::Virtual(p), _) => p.serialized_size(compress),
        };
        header_len + sumcheck_ext_len + poly_len
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
        let header = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let kind = header >> 6;
        let small = header & 0x3F;

        let sumcheck_u64 = if small == OPENING_ID_SUMCHECK_ESCAPE {
            transport::read_varint_u64(&mut reader).map_err(io_err)?
        } else {
            small as u64
        };

        let sumcheck_u8 =
            u8::try_from(sumcheck_u64).map_err(|_| SerializationError::InvalidData)?;
        let sumcheck_id =
            SumcheckId::from_u8(sumcheck_u8).ok_or(SerializationError::InvalidData)?;

        match kind {
            OPENING_ID_KIND_UNTRUSTED_ADVICE => Ok(OpeningId::UntrustedAdvice(sumcheck_id)),
            OPENING_ID_KIND_TRUSTED_ADVICE => Ok(OpeningId::TrustedAdvice(sumcheck_id)),
            OPENING_ID_KIND_COMMITTED => {
                let polynomial =
                    CommittedPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::committed(polynomial, sumcheck_id))
            }
            OPENING_ID_KIND_VIRTUAL => {
                let polynomial =
                    VirtualPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::virt(polynomial, sumcheck_id))
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
        match self {
            Self::RdInc => 0u8.serialize_with_mode(writer, compress),
            Self::RamInc => 1u8.serialize_with_mode(writer, compress),
            Self::InstructionRa(i) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::BytecodeRa(i) => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::RamRa(i) => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::TrustedAdvice => 5u8.serialize_with_mode(writer, compress),
            Self::UntrustedAdvice => 6u8.serialize_with_mode(writer, compress),
        }
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        match self {
            Self::RdInc | Self::RamInc | Self::TrustedAdvice | Self::UntrustedAdvice => 1,
            Self::InstructionRa(_) | Self::BytecodeRa(_) | Self::RamRa(_) => 2,
        }
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
        Ok(
            match u8::deserialize_with_mode(&mut reader, compress, validate)? {
                0 => Self::RdInc,
                1 => Self::RamInc,
                2 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                3 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::BytecodeRa(i as usize)
                }
                4 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::RamRa(i as usize)
                }
                5 => Self::TrustedAdvice,
                6 => Self::UntrustedAdvice,
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

impl CanonicalSerialize for VirtualPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::PC => 0u8.serialize_with_mode(&mut writer, compress),
            Self::UnexpandedPC => 1u8.serialize_with_mode(&mut writer, compress),
            Self::NextPC => 2u8.serialize_with_mode(&mut writer, compress),
            Self::NextUnexpandedPC => 3u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsNoop => 4u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsVirtual => 5u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsFirstInSequence => 6u8.serialize_with_mode(&mut writer, compress),
            Self::LeftLookupOperand => 7u8.serialize_with_mode(&mut writer, compress),
            Self::RightLookupOperand => 8u8.serialize_with_mode(&mut writer, compress),
            Self::LeftInstructionInput => 9u8.serialize_with_mode(&mut writer, compress),
            Self::RightInstructionInput => 10u8.serialize_with_mode(&mut writer, compress),
            Self::Product => 11u8.serialize_with_mode(&mut writer, compress),
            Self::ShouldJump => 12u8.serialize_with_mode(&mut writer, compress),
            Self::ShouldBranch => 13u8.serialize_with_mode(&mut writer, compress),
            Self::WritePCtoRD => 14u8.serialize_with_mode(&mut writer, compress),
            Self::WriteLookupOutputToRD => 15u8.serialize_with_mode(&mut writer, compress),
            Self::Rd => 16u8.serialize_with_mode(&mut writer, compress),
            Self::Imm => 17u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Value => 18u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Value => 19u8.serialize_with_mode(&mut writer, compress),
            Self::RdWriteValue => 20u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Ra => 21u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Ra => 22u8.serialize_with_mode(&mut writer, compress),
            Self::RdWa => 23u8.serialize_with_mode(&mut writer, compress),
            Self::LookupOutput => 24u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRaf => 25u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRafFlag => 26u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRa(i) => {
                27u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::RegistersVal => 28u8.serialize_with_mode(&mut writer, compress),
            Self::RamAddress => 29u8.serialize_with_mode(&mut writer, compress),
            Self::RamRa => 30u8.serialize_with_mode(&mut writer, compress),
            Self::RamReadValue => 31u8.serialize_with_mode(&mut writer, compress),
            Self::RamWriteValue => 32u8.serialize_with_mode(&mut writer, compress),
            Self::RamVal => 33u8.serialize_with_mode(&mut writer, compress),
            Self::RamValInit => 34u8.serialize_with_mode(&mut writer, compress),
            Self::RamValFinal => 35u8.serialize_with_mode(&mut writer, compress),
            Self::RamHammingWeight => 36u8.serialize_with_mode(&mut writer, compress),
            Self::UnivariateSkip => 37u8.serialize_with_mode(&mut writer, compress),
            Self::OpFlags(flags) => {
                38u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::InstructionFlags(flags) => {
                39u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::LookupTableFlag(flag) => {
                40u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flag).unwrap()).serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        match self {
            Self::PC
            | Self::UnexpandedPC
            | Self::NextPC
            | Self::NextUnexpandedPC
            | Self::NextIsNoop
            | Self::NextIsVirtual
            | Self::NextIsFirstInSequence
            | Self::LeftLookupOperand
            | Self::RightLookupOperand
            | Self::LeftInstructionInput
            | Self::RightInstructionInput
            | Self::Product
            | Self::ShouldJump
            | Self::ShouldBranch
            | Self::WritePCtoRD
            | Self::WriteLookupOutputToRD
            | Self::Rd
            | Self::Imm
            | Self::Rs1Value
            | Self::Rs2Value
            | Self::RdWriteValue
            | Self::Rs1Ra
            | Self::Rs2Ra
            | Self::RdWa
            | Self::LookupOutput
            | Self::InstructionRaf
            | Self::InstructionRafFlag
            | Self::RegistersVal
            | Self::RamAddress
            | Self::RamRa
            | Self::RamReadValue
            | Self::RamWriteValue
            | Self::RamVal
            | Self::RamValInit
            | Self::RamValFinal
            | Self::RamHammingWeight
            | Self::UnivariateSkip => 1,
            Self::InstructionRa(_)
            | Self::OpFlags(_)
            | Self::InstructionFlags(_)
            | Self::LookupTableFlag(_) => 2,
        }
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
        Ok(
            match u8::deserialize_with_mode(&mut reader, compress, validate)? {
                0 => Self::PC,
                1 => Self::UnexpandedPC,
                2 => Self::NextPC,
                3 => Self::NextUnexpandedPC,
                4 => Self::NextIsNoop,
                5 => Self::NextIsVirtual,
                6 => Self::NextIsFirstInSequence,
                7 => Self::LeftLookupOperand,
                8 => Self::RightLookupOperand,
                9 => Self::LeftInstructionInput,
                10 => Self::RightInstructionInput,
                11 => Self::Product,
                12 => Self::ShouldJump,
                13 => Self::ShouldBranch,
                14 => Self::WritePCtoRD,
                15 => Self::WriteLookupOutputToRD,
                16 => Self::Rd,
                17 => Self::Imm,
                18 => Self::Rs1Value,
                19 => Self::Rs2Value,
                20 => Self::RdWriteValue,
                21 => Self::Rs1Ra,
                22 => Self::Rs2Ra,
                23 => Self::RdWa,
                24 => Self::LookupOutput,
                25 => Self::InstructionRaf,
                26 => Self::InstructionRafFlag,
                27 => {
                    let i = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                28 => Self::RegistersVal,
                29 => Self::RamAddress,
                30 => Self::RamRa,
                31 => Self::RamReadValue,
                32 => Self::RamWriteValue,
                33 => Self::RamVal,
                34 => Self::RamValInit,
                35 => Self::RamValFinal,
                36 => Self::RamHammingWeight,
                37 => Self::UnivariateSkip,
                38 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = CircuitFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::OpFlags(flags)
                }
                39 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = InstructionFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::InstructionFlags(flags)
                }
                40 => {
                    let flag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::LookupTableFlag(flag as usize)
                }
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

pub fn serialize_and_print_size(
    item_name: &str,
    file_name: &str,
    item: &impl CanonicalSerialize,
) -> Result<(), SerializationError> {
    let mut file = File::create(file_name)?;
    item.serialize_compressed(&mut file)?;
    let file_size_bytes = file.metadata()?.len();
    let file_size_kb = file_size_bytes as f64 / 1024.0;
    tracing::info!("{item_name} Written to {file_name}");
    tracing::info!("{item_name} size: {file_size_kb:.1} kB");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::opening_proof::{OpeningId, SumcheckId};
    use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
    use crate::zkvm::RV64IMACProof;

    #[test]
    fn opening_id_header_is_packed_common_case() {
        let id = OpeningId::UntrustedAdvice(SumcheckId::SpartanOuter);
        let mut bytes = Vec::new();
        id.serialize_compressed(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 1);
        let expected = (OPENING_ID_KIND_UNTRUSTED_ADVICE << 6) | (SumcheckId::SpartanOuter as u8);
        assert_eq!(bytes[0], expected);

        let id = OpeningId::committed(CommittedPolynomial::RdInc, SumcheckId::SpartanOuter);
        let mut bytes = Vec::new();
        id.serialize_compressed(&mut bytes).unwrap();
        assert!(bytes.len() >= 2);
        assert_eq!(bytes[0] >> 6, OPENING_ID_KIND_COMMITTED);
        assert_eq!(bytes[0] & 0x3F, SumcheckId::SpartanOuter as u8);

        let id = OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let mut bytes = Vec::new();
        id.serialize_compressed(&mut bytes).unwrap();
        assert!(bytes.len() >= 2);
        assert_eq!(bytes[0] >> 6, OPENING_ID_KIND_VIRTUAL);
        assert_eq!(bytes[0] & 0x3F, SumcheckId::SpartanOuter as u8);
    }

    #[test]
    fn opening_id_roundtrips() {
        let cases = [
            OpeningId::UntrustedAdvice(SumcheckId::SpartanOuter),
            OpeningId::TrustedAdvice(SumcheckId::SpartanOuter),
            OpeningId::committed(CommittedPolynomial::RdInc, SumcheckId::SpartanOuter),
            OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter),
        ];
        for id in cases {
            let mut bytes = Vec::new();
            id.serialize_compressed(&mut bytes).unwrap();
            let decoded = OpeningId::deserialize_compressed(bytes.as_slice()).unwrap();
            assert_eq!(decoded, id);
        }
    }

    #[test]
    fn proof_version_byte() {
        assert_eq!(PROOF_VERSION, 1);
    }

    #[test]
    fn proof_magic_required() {
        let mut just_magic = Vec::new();
        transport::write_magic_version(&mut just_magic, PROOF_MAGIC, PROOF_VERSION).unwrap();
        let res = RV64IMACProof::deserialize_with_mode(
            std::io::Cursor::new(&just_magic),
            Compress::Yes,
            Validate::Yes,
        )
        .map(|_| ());
        assert!(res.is_err());
    }

    #[test]
    fn wrong_version_rejected() {
        let mut buf = Vec::new();
        transport::write_magic_version(&mut buf, PROOF_MAGIC, 99).unwrap();
        // Append enough zeros to avoid EOF on the version read
        buf.extend_from_slice(&[0u8; 100]);
        let res = RV64IMACProof::deserialize_with_mode(
            std::io::Cursor::new(&buf),
            Compress::Yes,
            Validate::Yes,
        )
        .map(|_| ());
        match res {
            Err(SerializationError::IoError(e)) => {
                assert!(
                    e.to_string().contains("unsupported proof version"),
                    "unexpected error: {e}"
                );
            }
            other => panic!("expected IoError with version message, got {other:?}"),
        }
    }

    #[test]
    fn wrong_magic_rejected() {
        let mut buf = Vec::new();
        transport::write_magic_version(&mut buf, b"BADMAGC", 1).unwrap();
        buf.extend_from_slice(&[0u8; 100]);
        let res = RV64IMACProof::deserialize_with_mode(
            std::io::Cursor::new(&buf),
            Compress::Yes,
            Validate::Yes,
        )
        .map(|_| ());
        match res {
            Err(SerializationError::IoError(e)) => {
                assert!(
                    e.to_string().contains("invalid proof magic"),
                    "unexpected error: {e}"
                );
            }
            other => panic!("expected IoError with magic message, got {other:?}"),
        }
    }
}
