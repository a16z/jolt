#[cfg(not(feature = "zk"))]
use std::collections::BTreeMap;
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

const PROOF_MAGIC: &[u8; 4] = b"JOLT";
const PROOF_VERSION: u8 = 1;
const PROOF_FLAGS_RESERVED_MASK: u8 = 0xFE;
const PROOF_FLAG_ZK: u8 = 0x01;

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
        debug_assert!(len <= MAX_SECTION_LEN, "section size {len} exceeds MAX_SECTION_LEN");
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

fn check_trailing_bytes<R: Read>(
    limited: &std::io::Take<&mut R>,
) -> Result<(), SerializationError> {
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
        let flags: u8 = if cfg!(feature = "zk") {
            PROOF_FLAG_ZK
        } else {
            0
        };
        writer.write_all(&[flags]).map_err(io_err)?;

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
        write_section!(&mut writer, compress, &self.opening_claims);

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
        let mut size = PROOF_MAGIC.len() + 1 + 1; // magic + version + flags

        let params_len = self.params_payload_len(compress);
        size += transport::varint_u64_len(params_len) + params_len as usize;

        let commitments_len = self.commitments_payload_len(compress);
        size += transport::varint_u64_len(commitments_len) + commitments_len as usize;

        #[cfg(not(feature = "zk"))]
        {
            size += section_size!(compress, &self.opening_claims);
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

        let mut flags_buf = [0u8; 1];
        reader.read_exact(&mut flags_buf).map_err(io_err)?;
        let flags = flags_buf[0];
        if flags & PROOF_FLAGS_RESERVED_MASK != 0 {
            return Err(io_err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown proof flags bits set: {flags:#04x}"),
            )));
        }
        let proof_is_zk = flags & PROOF_FLAG_ZK != 0;
        let compiled_for_zk = cfg!(feature = "zk");
        if proof_is_zk != compiled_for_zk {
            let mode = if proof_is_zk { "ZK" } else { "standard" };
            let expected = if compiled_for_zk { "ZK" } else { "standard" };
            return Err(io_err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("proof was serialized in {mode} mode but deserializer expects {expected}"),
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

        #[cfg(not(feature = "zk"))]
        let opening_claims: Claims<F> = read_single_section!(&mut reader);

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

        let stage3_sumcheck_proof = read_single_section!(&mut reader);
        let stage4_sumcheck_proof = read_single_section!(&mut reader);
        let stage5_sumcheck_proof = read_single_section!(&mut reader);
        let stage6_sumcheck_proof = read_single_section!(&mut reader);
        let stage7_sumcheck_proof = read_single_section!(&mut reader);
        let joint_opening_proof = read_single_section!(&mut reader);

        #[cfg(feature = "zk")]
        let blindfold_proof = read_single_section!(&mut reader);

        let mut eof_check = [0u8; 1];
        match reader.read(&mut eof_check) {
            Ok(0) => {}
            Ok(_) => {
                return Err(io_err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unexpected trailing bytes after proof",
                )));
            }
            Err(e) => return Err(io_err(e)),
        }

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
const MAX_CLAIMS_COUNT: u64 = 10_000;

#[cfg(not(feature = "zk"))]
impl<F: JoltField> CanonicalSerialize for Claims<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        transport::write_varint_u64(&mut writer, self.0.len() as u64).map_err(io_err)?;
        for (key, (_opening_point, claim)) in self.0.iter() {
            key.serialize_with_mode(&mut writer, compress)?;
            claim.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = transport::varint_u64_len(self.0.len() as u64);
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
        let n = transport::read_varint_u64(&mut reader).map_err(io_err)?;
        let n_usize = usize::try_from(n).map_err(|_| SerializationError::InvalidData)?;
        if n > MAX_CLAIMS_COUNT {
            return Err(SerializationError::InvalidData);
        }
        let mut claims = BTreeMap::new();
        for _ in 0..n_usize {
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
            Self::Rd => 14u8.serialize_with_mode(&mut writer, compress),
            Self::Imm => 15u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Value => 16u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Value => 17u8.serialize_with_mode(&mut writer, compress),
            Self::RdWriteValue => 18u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Ra => 19u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Ra => 20u8.serialize_with_mode(&mut writer, compress),
            Self::RdWa => 21u8.serialize_with_mode(&mut writer, compress),
            Self::LookupOutput => 22u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRaf => 23u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRafFlag => 24u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRa(i) => {
                25u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::RegistersVal => 26u8.serialize_with_mode(&mut writer, compress),
            Self::RamAddress => 27u8.serialize_with_mode(&mut writer, compress),
            Self::RamRa => 28u8.serialize_with_mode(&mut writer, compress),
            Self::RamReadValue => 29u8.serialize_with_mode(&mut writer, compress),
            Self::RamWriteValue => 30u8.serialize_with_mode(&mut writer, compress),
            Self::RamVal => 31u8.serialize_with_mode(&mut writer, compress),
            Self::RamValInit => 32u8.serialize_with_mode(&mut writer, compress),
            Self::RamValFinal => 33u8.serialize_with_mode(&mut writer, compress),
            Self::RamHammingWeight => 34u8.serialize_with_mode(&mut writer, compress),
            Self::UnivariateSkip => 35u8.serialize_with_mode(&mut writer, compress),
            Self::OpFlags(flags) => {
                36u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::InstructionFlags(flags) => {
                37u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::LookupTableFlag(flag) => {
                38u8.serialize_with_mode(&mut writer, compress)?;
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
                14 => Self::Rd,
                15 => Self::Imm,
                16 => Self::Rs1Value,
                17 => Self::Rs2Value,
                18 => Self::RdWriteValue,
                19 => Self::Rs1Ra,
                20 => Self::Rs2Ra,
                21 => Self::RdWa,
                22 => Self::LookupOutput,
                23 => Self::InstructionRaf,
                24 => Self::InstructionRafFlag,
                25 => {
                    let i = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                26 => Self::RegistersVal,
                27 => Self::RamAddress,
                28 => Self::RamRa,
                29 => Self::RamReadValue,
                30 => Self::RamWriteValue,
                31 => Self::RamVal,
                32 => Self::RamValInit,
                33 => Self::RamValFinal,
                34 => Self::RamHammingWeight,
                35 => Self::UnivariateSkip,
                36 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = CircuitFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::OpFlags(flags)
                }
                37 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = InstructionFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::InstructionFlags(flags)
                }
                38 => {
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
    use std::fs::File;
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
        use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};

        let sumcheck_ids = [
            SumcheckId::SpartanOuter,
            SumcheckId::RamReadWriteChecking,
            SumcheckId::HammingWeightClaimReduction,
        ];

        let committed_polys = [
            CommittedPolynomial::RdInc,
            CommittedPolynomial::RamInc,
            CommittedPolynomial::InstructionRa(0),
            CommittedPolynomial::InstructionRa(7),
            CommittedPolynomial::BytecodeRa(0),
            CommittedPolynomial::RamRa(0),
            CommittedPolynomial::TrustedAdvice,
            CommittedPolynomial::UntrustedAdvice,
        ];

        let virtual_polys = [
            VirtualPolynomial::PC,
            VirtualPolynomial::NextPC,
            VirtualPolynomial::UnivariateSkip,
            VirtualPolynomial::InstructionRa(0),
            VirtualPolynomial::InstructionRa(5),
            VirtualPolynomial::OpFlags(CircuitFlags::AddOperands),
            VirtualPolynomial::OpFlags(CircuitFlags::IsLastInSequence),
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            VirtualPolynomial::LookupTableFlag(0),
            VirtualPolynomial::LookupTableFlag(3),
        ];

        for &sc in &sumcheck_ids {
            let id = OpeningId::UntrustedAdvice(sc);
            let mut bytes = Vec::new();
            id.serialize_compressed(&mut bytes).unwrap();
            assert_eq!(
                OpeningId::deserialize_compressed(bytes.as_slice()).unwrap(),
                id
            );

            let id = OpeningId::TrustedAdvice(sc);
            let mut bytes = Vec::new();
            id.serialize_compressed(&mut bytes).unwrap();
            assert_eq!(
                OpeningId::deserialize_compressed(bytes.as_slice()).unwrap(),
                id
            );

            for &cp in &committed_polys {
                let id = OpeningId::committed(cp, sc);
                let mut bytes = Vec::new();
                id.serialize_compressed(&mut bytes).unwrap();
                assert_eq!(
                    OpeningId::deserialize_compressed(bytes.as_slice()).unwrap(),
                    id
                );
            }

            for &vp in &virtual_polys {
                let id = OpeningId::virt(vp, sc);
                let mut bytes = Vec::new();
                id.serialize_compressed(&mut bytes).unwrap();
                assert_eq!(
                    OpeningId::deserialize_compressed(bytes.as_slice()).unwrap(),
                    id
                );
            }
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
        transport::write_magic_version(&mut buf, b"BAAD", 1).unwrap();
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
