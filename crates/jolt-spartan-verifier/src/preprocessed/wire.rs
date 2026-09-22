//! Fixed-shape wire v1 for the clear v2 protocol. See specs/preprocessed-spartan-wire.md.
//! Counts come only from an authenticated, bounded key; only bounded round-width tags vary within fixed-size records.
use super::sparse::{ProductLayerProof, ProductNetworkProof, SlabOpening, SparseMatrixProof};
use super::{
    ComputationKey, MatrixApplicationIds, MatrixError, MatrixShape, PreprocessedProof,
    PublicColumnProof,
};
use jolt_crypto::Bn254G1;
use jolt_field::{CanonicalBytes, CanonicalEncoding, Fr, Zero};
use jolt_hyperkzg::{HyperKZGProof, HyperKZGVerifierSetup};
use jolt_poly::CompressedPoly;
use jolt_sumcheck::CompressedSumcheckProof;
use thiserror::Error;

const MAGIC: &[u8; 16] = b"JOLT-SPARK-PROOF";
const VERSION: u32 = 1;
const HEADER: usize = 52;
const KEY_BYTES: usize = 476;
const MAX_DEPTH: usize = 32;
const MAX_PUBLIC_COLUMNS: usize = 1024;
const MAX_PROOF_BYTES: usize = 1 << 20;

#[derive(Debug, Error)]
pub enum WireError {
    #[error("wire length, truncation, or trailing bytes")]
    Length,
    #[error("unsupported wire version or noncanonical key header")]
    Version,
    #[error("wire geometry exceeds the v1 resource profile")]
    Limit,
    #[error("noncanonical field encoding")]
    Field,
    #[error("invalid or noncanonical compressed subgroup point")]
    Group,
    #[error("typed proof shape differs from authenticated key")]
    Shape,
    #[error("compressed round width is outside its degree bound")]
    RoundEncoding,
    #[error("nonzero padding in a compressed round record")]
    Padding,
    #[error(transparent)]
    Protocol(#[from] MatrixError),
}

/// Decodes the existing canonical v2 key, authenticates it, then verifies wire v1.
/// `expected_id` and `setup` must be authenticated by the application, not the sender.
/// Public inputs are exactly p-1 canonical little-endian Fr values, without a count.
/// This bounds parsing; it does not establish setup provenance, ZK or ROM extraction.
pub fn verify_bytes(
    expected_id: &[u8; 32],
    setup: &HyperKZGVerifierSetup,
    key_bytes: &[u8],
    public_input_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<(), WireError> {
    let key = ComputationKey::decode_wire(key_bytes, expected_id, setup)?;
    let count = key.shape.public_columns - 1;
    if public_input_bytes.len() != count * 32 {
        return Err(WireError::Length);
    }
    let inputs = Reader::new(public_input_bytes).fields(count)?;
    let proof = key.decode_proof(proof_bytes)?;
    key.verify(expected_id, &inputs, &proof, setup)?;
    Ok(())
}

impl ComputationKey {
    /// Parses exactly the existing 476-byte canonical key; checks policy before allocation.
    pub fn decode_wire(
        bytes: &[u8],
        expected_id: &[u8; 32],
        setup: &HyperKZGVerifierSetup,
    ) -> Result<Self, WireError> {
        if bytes.len() != KEY_BYTES {
            return Err(WireError::Length);
        }
        if Self::digest(bytes) != *expected_id {
            return Err(MatrixError::Identity.into());
        }
        let mut input = Reader::new(bytes);
        // Canonical re-encoding below checks magic, IDs and modulus as one owned format.
        let _ = input.take(16 + 28 + 32)?;
        let shape = MatrixShape {
            rows: input.count()?,
            columns: input.count()?,
            public_columns: input.count()?,
            padded_rows: input.count()?,
            padded_private: input.count()?,
            operations: input.count()?,
            memory: input.count()?,
            public_slabs: input.count()?,
        };
        let _ = Geometry::new(shape)?;
        let num_powers = input.u64()?;
        let max_public_degree = input.u64()?;
        let application = MatrixApplicationIds {
            circuit: input.array()?,
            profile: input.array()?,
            public_schema: input.array()?,
            table: input.array()?,
        };
        let matrix_digest = input.array()?;
        let setup_id = input.array()?;
        let setup_digest = input.array()?;
        let commitments = [input.group()?, input.group()?, input.group()?];
        input.finish()?;
        let key = Self::new(shape, application, matrix_digest, commitments, setup)?;
        if key.num_powers != num_powers
            || key.max_public_degree != max_public_degree
            || key.setup_id != setup_id
            || key.setup_digest != setup_digest
        {
            return Err(MatrixError::Identity.into());
        }
        if key.canonical_bytes() != bytes {
            return Err(WireError::Version);
        }
        key.authenticate(expected_id, setup)?;
        Ok(key)
    }

    /// Encodes the exact typed transcript profile; zero padding is not transcript data.
    pub fn encode_proof(&self, proof: &PreprocessedProof) -> Result<Vec<u8>, WireError> {
        let g = Geometry::new(self.shape)?;
        let mut out = Writer(Vec::with_capacity(g.bytes));
        out.0.extend(MAGIC);
        out.0.extend(VERSION.to_le_bytes());
        out.0.extend(self.id());
        out.group(&proof.witness_commitment);
        out.sumcheck(&proof.outer, g.r, 3)?;
        out.fields(&proof.outer_evaluations, 3)?;
        out.fields(&proof.public.evaluations, self.shape.public_columns * 3)?;
        out.pcs(&proof.public.opening, g.public)?;
        out.sumcheck(&proof.inner, g.w, 2)?;
        out.fields(&proof.private_values, 3)?;
        out.group(&proof.sparse.dereference_commitment);
        out.fields(&proof.sparse.roots, 16)?;
        out.fields(&proof.sparse.dot_halves, 6)?;
        out.network(&proof.sparse.operations, g.n, 12, true)?;
        out.network(&proof.sparse.memory, g.l, 4, false)?;
        out.slab(&proof.sparse.dereferences, 6, g.n + 3)?;
        out.slab(&proof.sparse.operation_values, 16, g.n + 4)?;
        out.slab(&proof.sparse.audit_values, 2, g.l + 1)?;
        out.fields(&[proof.witness_evaluation], 1)?;
        out.pcs(&proof.witness_opening, g.w)?;
        if out.0.len() != g.bytes {
            return Err(WireError::Shape);
        }
        Ok(out.0)
    }

    /// Decodes a proof using this already-authenticated key's fixed geometry.
    /// Applications should use `verify_bytes` to authenticate the key and verify algebra.
    pub fn decode_proof(&self, bytes: &[u8]) -> Result<PreprocessedProof, WireError> {
        let g = Geometry::new(self.shape)?;
        if bytes.len() != g.bytes {
            return Err(WireError::Length);
        }
        let mut input = Reader::new(bytes);
        if input.take(16)? != MAGIC || input.array::<4>()? != VERSION.to_le_bytes() {
            return Err(WireError::Version);
        }
        if input.array::<32>()? != self.id() {
            return Err(MatrixError::Identity.into());
        }
        let witness_commitment = input.group()?;
        let outer = input.sumcheck(g.r, 3)?;
        let outer_evaluations = input.field_array()?;
        let public = PublicColumnProof {
            evaluations: input.fields(self.shape.public_columns * 3)?,
            opening: input.pcs(g.public)?,
        };
        let inner = input.sumcheck(g.w, 2)?;
        let private_values = input.field_array()?;
        let sparse = SparseMatrixProof {
            dereference_commitment: input.group()?,
            roots: input.field_array()?,
            dot_halves: input.field_array()?,
            operations: input.network(g.n, 12, true)?,
            memory: input.network(g.l, 4, false)?,
            dereferences: input.slab(6, g.n + 3)?,
            operation_values: input.slab(16, g.n + 4)?,
            audit_values: input.slab(2, g.l + 1)?,
        };
        let witness_evaluation = input.field()?;
        let witness_opening = input.pcs(g.w)?;
        input.finish()?;
        Ok(PreprocessedProof {
            witness_commitment,
            outer,
            outer_evaluations,
            public,
            inner,
            private_values,
            sparse,
            witness_evaluation,
            witness_opening,
        })
    }
}

struct Geometry {
    r: usize,
    w: usize,
    n: usize,
    l: usize,
    public: usize,
    bytes: usize,
}
impl Geometry {
    fn new(shape: MatrixShape) -> Result<Self, WireError> {
        shape.validate()?;
        let [r, w, n, l, t] = [
            shape.padded_rows,
            shape.padded_private,
            shape.operations,
            shape.memory,
            shape.public_slabs,
        ]
        .map(|x| x.trailing_zeros() as usize);
        if [r, w, n, l, t].into_iter().any(|d| d > MAX_DEPTH)
            || shape.public_columns > MAX_PUBLIC_COLUMNS
        {
            return Err(WireError::Limit);
        }
        // Depth <=32 and p<=1024 bound every subsequent product/sum on 32-bit usize too.
        let public = r + t;
        let arities = [public, n + 3, n + 4, l + 1, w];
        let scalars = 3 * r
            + 3
            + 3 * shape.public_columns
            + 2 * w
            + 3
            + 16
            + 6
            + 3 * n * (n - 1) / 2
            + 24 * n
            + 18
            + 3 * l * (l - 1) / 2
            + 8 * l
            + 6
            + 16
            + 2
            + 1
            + 3 * arities.iter().sum::<usize>();
        let groups = 2 + arities.iter().map(|n| n + 2).sum::<usize>();
        let round_tags = r + w + n * (n - 1) / 2 + l * (l - 1) / 2;
        let bytes = HEADER + round_tags + 32 * (scalars + groups);
        if bytes > MAX_PROOF_BYTES {
            return Err(WireError::Limit);
        }
        Ok(Self {
            r,
            w,
            n,
            l,
            public,
            bytes,
        })
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        let (head, tail) = self.bytes.split_at_checked(n).ok_or(WireError::Length)?;
        self.bytes = tail;
        Ok(head)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], WireError> {
        self.take(N)?.try_into().map_err(|_| WireError::Length)
    }
    fn u64(&mut self) -> Result<u64, WireError> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn count(&mut self) -> Result<usize, WireError> {
        usize::try_from(self.u64()?).map_err(|_| WireError::Limit)
    }
    fn field(&mut self) -> Result<Fr, WireError> {
        Fr::from_bytes_le_checked(self.take(32)?).ok_or(WireError::Field)
    }
    fn fields(&mut self, n: usize) -> Result<Vec<Fr>, WireError> {
        (0..n).map(|_| self.field()).collect()
    }
    fn field_array<const N: usize>(&mut self) -> Result<[Fr; N], WireError> {
        self.fields(N)?.try_into().map_err(|_| WireError::Shape)
    }
    fn group(&mut self) -> Result<Bn254G1, WireError> {
        Bn254G1::from_compressed_bytes(self.take(32)?).ok_or(WireError::Group)
    }
    fn pcs(&mut self, n: usize) -> Result<HyperKZGProof, WireError> {
        let com = (0..n - 1).map(|_| self.group()).collect::<Result<_, _>>()?;
        let v = [self.fields(n)?, self.fields(n)?, self.fields(n)?];
        let w = [self.group()?, self.group()?, self.group()?];
        Ok(HyperKZGProof { com, v, w })
    }
    fn sumcheck(
        &mut self,
        rounds: usize,
        width: usize,
    ) -> Result<CompressedSumcheckProof<Fr>, WireError> {
        let mut round_polynomials = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let [count] = self.array()?;
            let count = usize::from(count);
            if count == 0 || count > width {
                return Err(WireError::RoundEncoding);
            }
            let mut coefficients = self.fields(width)?;
            if coefficients
                .iter()
                .skip(count)
                .any(|value| *value != Fr::zero())
            {
                return Err(WireError::Padding);
            }
            coefficients.truncate(count);
            round_polynomials.push(CompressedPoly::new(coefficients));
        }
        Ok(CompressedSumcheckProof { round_polynomials })
    }

    fn network(
        &mut self,
        depth: usize,
        width: usize,
        dots: bool,
    ) -> Result<ProductNetworkProof, WireError> {
        let mut layers = Vec::with_capacity(depth);
        for j in 0..depth {
            layers.push(ProductLayerProof {
                sumcheck: self.sumcheck(j, 3)?,
                ends: (0..width)
                    .map(|_| self.field_array())
                    .collect::<Result<_, _>>()?,
                dot_ends: (0..if dots && j + 1 == depth { 6 } else { 0 })
                    .map(|_| self.field_array())
                    .collect::<Result<_, _>>()?,
            });
        }
        Ok(ProductNetworkProof { layers })
    }
    fn slab(&mut self, count: usize, n: usize) -> Result<SlabOpening, WireError> {
        Ok(SlabOpening {
            evaluations: self.fields(count)?,
            opening: self.pcs(n)?,
        })
    }
    fn finish(self) -> Result<(), WireError> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(WireError::Length)
        }
    }
}
struct Writer(Vec<u8>);
impl Writer {
    fn group(&mut self, value: &Bn254G1) {
        self.0.extend(value.compressed_bytes());
    }
    fn fields(&mut self, values: &[Fr], n: usize) -> Result<(), WireError> {
        if values.len() != n {
            return Err(WireError::Shape);
        }
        for value in values {
            self.0.extend(value.to_bytes_le_vec());
        }
        Ok(())
    }
    fn pcs(&mut self, proof: &HyperKZGProof, n: usize) -> Result<(), WireError> {
        if proof.com.len() != n - 1 {
            return Err(WireError::Shape);
        }
        for c in &proof.com {
            self.group(c);
        }
        for row in &proof.v {
            self.fields(row, n)?;
        }
        for w in &proof.w {
            self.group(w);
        }
        Ok(())
    }
    fn sumcheck(
        &mut self,
        proof: &CompressedSumcheckProof<Fr>,
        rounds: usize,
        width: usize,
    ) -> Result<(), WireError> {
        if proof.round_polynomials.len() != rounds {
            return Err(WireError::Shape);
        }
        for round in &proof.round_polynomials {
            let coeffs = round.coeffs_except_linear_term();
            if coeffs.is_empty() || coeffs.len() > width {
                return Err(WireError::RoundEncoding);
            }
            self.0
                .push(u8::try_from(coeffs.len()).map_err(|_| WireError::RoundEncoding)?);
            self.fields(coeffs, coeffs.len())?;
            for _ in coeffs.len()..width {
                self.fields(&[Fr::zero()], 1)?;
            }
        }
        Ok(())
    }
    fn network(
        &mut self,
        proof: &ProductNetworkProof,
        depth: usize,
        width: usize,
        dots: bool,
    ) -> Result<(), WireError> {
        if proof.layers.len() != depth {
            return Err(WireError::Shape);
        }
        for (j, layer) in proof.layers.iter().enumerate() {
            self.sumcheck(&layer.sumcheck, j, 3)?;
            if layer.ends.len() != width
                || layer.dot_ends.len() != if dots && j + 1 == depth { 6 } else { 0 }
            {
                return Err(WireError::Shape);
            }
            for end in &layer.ends {
                self.fields(end, 2)?;
            }
            for end in &layer.dot_ends {
                self.fields(end, 3)?;
            }
        }
        Ok(())
    }
    fn slab(&mut self, proof: &SlabOpening, count: usize, n: usize) -> Result<(), WireError> {
        self.fields(&proof.evaluations, count)?;
        self.pcs(&proof.opening, n)
    }
}
