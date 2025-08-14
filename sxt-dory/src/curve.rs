#![allow(missing_docs)]
use crate::poly::Polynomial;
use crate::{arithmetic::*, compute_polynomial_commitment, multilinear_lagrange_vec, ProverSetup};
use ark_bn254::{Bn254, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{
    bn::{G1Prepared as BnG1Prepared, G2Prepared as BnG2Prepared},
    pairing::{MillerLoopOutput, Pairing as ArkPairing},
    AffineRepr, CurveGroup,
};
use ark_ff::{Field as ArkField, One, PrimeField, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError};
use ark_serialize::{Read, Valid, Validate, Write};
use ark_std::rand::{rngs::StdRng, RngCore, SeedableRng};
use rayon::prelude::*;

use jolt_optimizations::{
    dory_g1::precompute_g1_generators_windowed2_signed, vector_add_scalar_mul_g1_windowed2_signed,
    vector_add_scalar_mul_g2_windowed2_signed, PrecomputedShamir4Data, Windowed2Signed2Data,
    Windowed2Signed4Data,
};

/// Create a fixed RNG for deterministic tests
pub fn test_rng() -> StdRng {
    let seed = [
        1, 0, 0, 30, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    StdRng::from_seed(seed)
}

/* --------- Field trait for Fr --------------------------------------- */
impl Field for Fr {
    fn zero() -> Self {
        Zero::zero()
    }
    fn one() -> Self {
        One::one()
    }
    fn is_zero(&self) -> bool {
        Zero::is_zero(self)
    }

    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
    fn sub(&self, rhs: &Self) -> Self {
        *self - *rhs
    }
    fn mul(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
    fn inv(&self) -> Option<Self> {
        if Zero::is_zero(self) {
            None
        } else {
            Some(self.inverse().unwrap())
        }
    }
    fn random<R: RngCore>(_rng: &mut R) -> Self {
        // We use our own fixed RNG for testing
        let mut rng = test_rng();
        Fr::rand(&mut rng)
    }

    fn from_u64(val: u64) -> Self {
        Fr::from(val)
    }

    fn from_i64(val: i64) -> Self {
        if val >= 0 {
            Fr::from(val as u64)
        } else {
            -Fr::from((-val) as u64)
        }
    }
}

/* --------- Group trait for G1Affine -------------------------------- */
impl Group for G1Affine {
    type Scalar = Fr;

    fn identity() -> Self {
        G1Affine::identity()
    }

    fn add(&self, rhs: &Self) -> Self {
        (self.into_group() + rhs.into_group()).into_affine()
    }

    fn neg(&self) -> Self {
        (-self.into_group()).into_affine()
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        self.mul_bigint((*k).into_bigint()).into_affine()
    }

    fn random<R: RngCore>(_rng: &mut R) -> Self {
        let mut rng = test_rng();
        G1Projective::rand(&mut rng).into_affine()
    }
}

/// G1Affine and G2Affine are the same up to alias from arkworks.
/// Hence, we have to use newType idiom here to avoid compiler conflicts
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct G2AffineWrapper(G2Affine);

// Implement operator traits for G2AffineWrapper
impl std::ops::Add for G2AffineWrapper {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        G2AffineWrapper((self.0.into_group() + rhs.0.into_group()).into_affine())
    }
}

impl std::ops::Add<&G2AffineWrapper> for G2AffineWrapper {
    type Output = Self;

    fn add(self, rhs: &G2AffineWrapper) -> Self::Output {
        G2AffineWrapper((self.0.into_group() + rhs.0.into_group()).into_affine())
    }
}

impl std::ops::Neg for G2AffineWrapper {
    type Output = Self;

    fn neg(self) -> Self::Output {
        G2AffineWrapper((-self.0.into_group()).into_affine())
    }
}

impl From<G2Affine> for G2AffineWrapper {
    fn from(value: G2Affine) -> Self {
        G2AffineWrapper(value)
    }
}

impl From<G2AffineWrapper> for G2Affine {
    fn from(value: G2AffineWrapper) -> Self {
        value.0
    }
}

// Implementations for ark-serialize
impl CanonicalSerialize for G2AffineWrapper {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }

    fn serialize_compressed<W: std::io::Write>(
        &self,
        writer: W,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.serialize_with_mode(writer, ark_serialize::Compress::Yes)
    }

    fn compressed_size(&self) -> usize {
        self.serialized_size(ark_serialize::Compress::Yes)
    }

    fn serialize_uncompressed<W: std::io::Write>(
        &self,
        writer: W,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.serialize_with_mode(writer, ark_serialize::Compress::No)
    }

    fn uncompressed_size(&self) -> usize {
        self.serialized_size(ark_serialize::Compress::No)
    }
}
impl CanonicalDeserialize for G2AffineWrapper {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        G2Affine::deserialize_with_mode(reader, compress, validate).map(G2AffineWrapper)
    }

    fn deserialize_compressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::Yes, Validate::Yes)
    }

    fn deserialize_compressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::Yes, Validate::No)
    }

    fn deserialize_uncompressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::Yes)
    }

    fn deserialize_uncompressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::No)
    }
}
impl Valid for G2AffineWrapper {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

/* --------- Group trait for G2AffineWrapper ------------------------ */
impl Group for G2AffineWrapper {
    type Scalar = Fr;

    fn identity() -> Self {
        G2AffineWrapper(G2Affine::identity())
    }

    fn add(&self, rhs: &Self) -> Self {
        G2AffineWrapper((self.0.into_group() + rhs.0.into_group()).into_affine())
    }

    fn neg(&self) -> Self {
        G2AffineWrapper((-self.0.into_group()).into_affine())
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        G2AffineWrapper(self.0.mul_bigint((*k).into_bigint()).into_affine())
    }

    fn random<R: RngCore>(_rng: &mut R) -> Self {
        // We use our own fixed RNG for testing
        let mut rng = test_rng();
        G2AffineWrapper(G2Projective::rand(&mut rng).into_affine())
    }
}

/* --------- Group trait for Fq12 (GT) ------------------------------- */
impl Group for Fq12 {
    type Scalar = Fr;

    fn identity() -> Self {
        Self::one()
    }

    fn add(&self, rhs: &Self) -> Self {
        *self * *rhs // Multiplicative group
    }

    fn neg(&self) -> Self {
        if Zero::is_zero(self) {
            *self
        } else {
            self.inverse().unwrap()
        }
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        // We convert to BigInt representation suitable for powering
        let repr = (*k).into_bigint();
        self.pow(repr)
    }

    fn random<R: RngCore>(_rng: &mut R) -> Self {
        // We use our own fixed RNG for testing
        let mut rng = test_rng();
        Self::rand(&mut rng)
    }
}

/* --------- lightweight Pairing wrapper ----------------------------- */
#[derive(Clone, Debug)]
pub struct ArkBn254Pairing;

impl Pairing for ArkBn254Pairing {
    type G1 = G1Affine;
    type G2 = G2AffineWrapper;
    type GT = Fq12;

    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT {
        Bn254::pairing(*p, q.0).0
    }

    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        // Convert to the cached version format for consistency
        Self::multi_pair_cached(Some(ps), None, None, Some(qs), None, None)
    }

    fn multi_pair_cached(
        g1_points: Option<&[Self::G1]>,
        g1_count: Option<usize>,
        g1_cache: Option<&G1Cache>,
        g2_points: Option<&[Self::G2]>,
        g2_count: Option<usize>,
        g2_cache: Option<&G2Cache>,
    ) -> Self::GT {
        match (g1_points, g1_count, g1_cache, g2_points, g2_count, g2_cache) {
            // Case 1: Both G1 and G2 use cached prepared values (fully optimized)
            (None, Some(g1_c), Some(g1_cache), None, Some(g2_c), Some(g2_cache)) => {
                assert_eq!(g1_c, g2_c, "G1 and G2 counts must be equal");
                if g1_c == 0 {
                    return Fq12::one();
                }

                // Extract prepared values as iterators - avoids Vec allocation and cloning
                let g1_prepared = (0..g1_c).map(|i| {
                    g1_cache
                        .get_prepared(i)
                        .expect("Index out of bounds in G1 cache")
                });

                let g2_prepared = (0..g2_c).map(|i| {
                    g2_cache
                        .get_prepared(i)
                        .expect("Index out of bounds in G2 cache")
                });

                let ml_result = Bn254::multi_miller_loop(g1_prepared, g2_prepared).0;

                let pairing_result = Bn254::final_exponentiation(MillerLoopOutput(ml_result))
                    .expect("Final exponentiation should not fail");

                pairing_result.0
            }

            // Case 2: G1 cached, G2 fresh points (partial optimization)
            (None, Some(g1_c), Some(g1_cache), Some(g2_points), _, _) => {
                assert_eq!(
                    g1_c,
                    g2_points.len(),
                    "G1 count must equal G2 points length"
                );
                if g1_c == 0 {
                    return Fq12::one();
                }

                // G1 from cache as iterator, G2 fresh preparation
                let g1_prepared = (0..g1_c).map(|i| {
                    g1_cache
                        .get_prepared(i)
                        .expect("Index out of bounds in G1 cache")
                });

                let g2_prepared = g2_points
                    .par_iter()
                    .map(|q| BnG2Prepared::from(q.0))
                    .collect::<Vec<_>>();

                let ml_result = Bn254::multi_miller_loop(g1_prepared, &g2_prepared).0;

                let pairing_result = Bn254::final_exponentiation(MillerLoopOutput(ml_result))
                    .expect("Final exponentiation should not fail");

                pairing_result.0
            }

            // Case 3: G1 fresh points, G2 cached (partial optimization)
            (Some(g1_points), _, _, None, Some(g2_c), Some(g2_cache)) => {
                assert_eq!(
                    g1_points.len(),
                    g2_c,
                    "G1 points length must equal G2 count"
                );
                if g2_c == 0 {
                    return Fq12::one();
                }

                // G1 fresh preparation, G2 from cache as iterator
                let g1_prepared = g1_points
                    .par_iter()
                    .map(|&g| BnG1Prepared::from(g))
                    .collect::<Vec<_>>();

                let g2_prepared = (0..g2_c).map(|i| {
                    g2_cache
                        .get_prepared(i)
                        .expect("Index out of bounds in G2 cache")
                });

                let ml_result = Bn254::multi_miller_loop(&g1_prepared, g2_prepared).0;

                let pairing_result = Bn254::final_exponentiation(MillerLoopOutput(ml_result))
                    .expect("Final exponentiation should not fail");

                pairing_result.0
            }

            // Case 4: Both fresh points (no caching benefit)
            (Some(g1_points), _, _, Some(g2_points), _, _) => {
                assert_eq!(
                    g1_points.len(),
                    g2_points.len(),
                    "G1 and G2 vectors must have equal length"
                );
                if g1_points.is_empty() {
                    return Fq12::one();
                }

                let left = g1_points
                    .par_iter()
                    .map(|&g| BnG1Prepared::from(g))
                    .collect::<Vec<_>>();

                let right = g2_points
                    .par_iter()
                    .map(|q| BnG2Prepared::from(q.0))
                    .collect::<Vec<_>>();

                let ml_result = Bn254::multi_miller_loop(&left, &right).0;

                let pairing_result = Bn254::final_exponentiation(MillerLoopOutput(ml_result))
                    .expect("Final exponentiation should not fail");

                pairing_result.0
            }

            _ => panic!("Invalid combination of parameters provided to multi_pair_cached"),
        }
    }
}

/* --------- Cache structures for optimized MSM operations ----------- */

/// Cache entry for a single G1 point containing precomputed values
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G1CacheEntry {
    /// Original affine point
    pub affine: G1Affine,
    /// Projective version for faster group operations
    pub projective: G1Projective,
    /// Prepared version for pairing operations
    pub prepared: BnG1Prepared<ark_bn254::Config>,
}

/// Cache for multiple G1 points
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G1Cache {
    /// Cached entries indexed by position
    pub entries: Vec<G1CacheEntry>,
    /// Full precomputed windowed2 signed data for efficient scalar multiplication
    pub precomputed_data: Option<Windowed2Signed2Data>,
}

impl G1Cache {
    /// Initialize cache from a vector of G1 affine points
    pub fn new(generators: &[G1Affine]) -> Self {
        // First convert all generators to projective form
        let generators_proj: Vec<G1Projective> =
            generators.iter().map(|g| g.into_group()).collect();

        // Create precomputed windowed2 signed data for all generators
        let precomputed_data = precompute_g1_generators_windowed2_signed(&generators_proj);

        let entries: Vec<G1CacheEntry> = generators
            .par_iter()
            .map(|&g| {
                let projective = g.into_group();
                let prepared = BnG1Prepared::from(g);

                G1CacheEntry {
                    affine: g,
                    projective,
                    prepared,
                }
            })
            .collect();

        Self {
            entries,
            precomputed_data: Some(precomputed_data),
        }
    }

    /// Save cache to file with parallel serialization
    pub fn save_to_file(&self, path: &str) -> Result<(), SerializationError> {
        use std::io::Write as StdWrite;

        // Serialize entries in parallel
        let serialized_entries: Vec<Vec<u8>> = self
            .entries
            .par_iter()
            .map(|entry| {
                let mut bytes = Vec::new();
                entry.serialize_compressed(&mut bytes)?;
                Ok(bytes)
            })
            .collect::<Result<Vec<_>, SerializationError>>()?;

        // Now write to file sequentially
        let mut file = std::fs::File::create(path).map_err(|e| SerializationError::IoError(e))?;

        // Write number of entries
        (self.entries.len() as u64).serialize_compressed(&mut file)?;

        // Write each entry with its size
        for entry_bytes in serialized_entries {
            (entry_bytes.len() as u64).serialize_compressed(&mut file)?;
            file.write_all(&entry_bytes)
                .map_err(|e| SerializationError::IoError(e))?;
        }

        // Write precomputed_data
        self.precomputed_data.serialize_compressed(&mut file)?;

        file.flush().map_err(|e| SerializationError::IoError(e))?;
        Ok(())
    }

    /// Load cache from file with parallel deserialization
    pub fn load_from_file(path: &str) -> Result<Self, SerializationError> {
        use std::io::Read as StdRead;

        let mut file = std::fs::File::open(path).map_err(|e| SerializationError::IoError(e))?;

        // Read number of entries
        let num_entries = u64::deserialize_compressed(&mut file)? as usize;

        // Read all entry data first
        let mut entry_data = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            let size = u64::deserialize_compressed(&mut file)? as usize;
            let mut bytes = vec![0u8; size];
            file.read_exact(&mut bytes)
                .map_err(|e| SerializationError::IoError(e))?;
            entry_data.push(bytes);
        }

        // Deserialize entries in parallel
        let entries: Vec<G1CacheEntry> = entry_data
            .into_par_iter()
            .map(|bytes| G1CacheEntry::deserialize_compressed(&bytes[..]))
            .collect::<Result<Vec<_>, SerializationError>>()?;

        // Read precomputed_data
        let precomputed_data = Option::<Windowed2Signed2Data>::deserialize_compressed(&mut file)?;

        Ok(Self {
            entries,
            precomputed_data,
        })
    }

    /// Get a cache entry by index
    pub fn get_entry(&self, index: usize) -> Option<&G1CacheEntry> {
        self.entries.get(index)
    }

    /// Get the projective version of a point by index
    pub fn get_projective(&self, index: usize) -> Option<&G1Projective> {
        self.entries.get(index).map(|e| &e.projective)
    }

    /// Get the prepared version of a point by index
    pub fn get_prepared(&self, index: usize) -> Option<&BnG1Prepared<ark_bn254::Config>> {
        self.entries.get(index).map(|e| &e.prepared)
    }

    /// Number of cached entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get precomputed windowed2 signed data
    /// Returns a reference to the full windowed data structure
    pub fn get_windowed_data(&self) -> Option<&Windowed2Signed2Data> {
        self.precomputed_data.as_ref()
    }

    /// Calculate and print memory usage statistics for the cache
    pub fn print_memory_stats(&self) {
        use std::mem;

        println!("=== G1 Cache Memory Usage ===");
        println!("Number of entries: {}", self.entries.len());

        // Calculate entries vector memory
        let entries_capacity = self.entries.capacity();
        let entry_size = mem::size_of::<G1CacheEntry>();
        let entries_allocated = entries_capacity * entry_size;
        let entries_used = self.entries.len() * entry_size;

        println!("\nEntries Vector:");
        println!("  Entry struct size: {} bytes", entry_size);
        println!("  Capacity: {} entries", entries_capacity);
        println!("  Used: {} entries", self.entries.len());
        println!(
            "  Allocated memory: {} bytes ({:.2} MB)",
            entries_allocated,
            entries_allocated as f64 / 1_048_576.0
        );
        println!(
            "  Used memory: {} bytes ({:.2} MB)",
            entries_used,
            entries_used as f64 / 1_048_576.0
        );

        // Get actual sizes of types
        println!("\nType sizes:");
        println!("  G1Affine: {} bytes", mem::size_of::<G1Affine>());
        println!("  G1Projective: {} bytes", mem::size_of::<G1Projective>());
        println!(
            "  BnG1Prepared: {} bytes",
            mem::size_of::<BnG1Prepared<ark_bn254::Config>>()
        );

        // Calculate windowed data memory
        let mut windowed_memory = 0;
        if let Some(windowed_data) = &self.precomputed_data {
            // Size of the Option wrapper
            windowed_memory += mem::size_of::<Option<Windowed2Signed2Data>>();

            // Get the actual size of tables
            let tables_len = windowed_data.windowed2_tables.len();
            let tables_capacity = windowed_data.windowed2_tables.capacity();

            // We can't easily get the size of Windowed2Signed2Table without access to the type
            // But we can estimate based on the fact it contains signed multiples
            // Each table typically contains 12 G1Projective points for 2-bit windowed method
            let estimated_table_size = 12 * mem::size_of::<G1Projective>();

            let windowed_allocated = tables_capacity * estimated_table_size;
            let windowed_used = tables_len * estimated_table_size;

            println!("\nWindowed2Signed2Data:");
            println!("  Tables count: {}", tables_len);
            println!("  Tables capacity: {}", tables_capacity);
            println!("  Estimated per table: {} bytes", estimated_table_size);
            println!(
                "  Allocated: {} bytes ({:.2} MB)",
                windowed_allocated,
                windowed_allocated as f64 / 1_048_576.0
            );
            println!(
                "  Used: {} bytes ({:.2} MB)",
                windowed_used,
                windowed_used as f64 / 1_048_576.0
            );

            windowed_memory += windowed_allocated;
        } else {
            println!("\nNo windowed precomputed data");
        }

        // Total memory
        let total_allocated = entries_allocated + windowed_memory + mem::size_of::<Self>();
        println!(
            "\n>>> G1 Cache Total Allocated: {} bytes ({:.2} MB)",
            total_allocated,
            total_allocated as f64 / 1_048_576.0
        );
        println!("=====================================\n");
    }
}

/// Cache entry for a single G2 point containing precomputed values
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G2CacheEntry {
    /// Original affine point
    pub affine: G2Affine,
    /// line evaluations for multi pairing
    pub prepared: BnG2Prepared<ark_bn254::Config>,
}

/// Cache for multiple G2 points
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G2Cache {
    /// Cached entries indexed by position
    pub entries: Vec<G2CacheEntry>,
    /// Full precomputed windowed2 signed data for efficient scalar multiplication
    pub precomputed_data: Option<Windowed2Signed4Data>,
    /// Precomputed GLV tables for g_fin point (if provided)
    pub g_fin_glv_tables: Option<PrecomputedShamir4Data>,
}

impl G2Cache {
    /// Initialize cache from a vector of G2 affine points
    /// At the moment the cache is more or less just the prepared values for the multi pairing
    /// @TODO(markosg04) simplify the cache
    pub fn new(generators: &[G2Affine], _g_fin: Option<&G2Affine>) -> Self {
        let entries: Vec<G2CacheEntry> = generators
            .par_iter()
            .map(|&g| {
                let prepared = BnG2Prepared::from(g);

                G2CacheEntry {
                    affine: g,
                    prepared,
                }
            })
            .collect();

        Self {
            entries,
            precomputed_data: None,
            g_fin_glv_tables: None,
        }
    }

    /// Initialize cache from a vector of G2AffineWrapper points
    pub fn new_from_wrappers(
        generators: &[G2AffineWrapper],
        g_fin: Option<&G2AffineWrapper>,
    ) -> Self {
        let native_generators: Vec<G2Affine> = generators.iter().map(|w| w.0).collect();
        let native_g_fin = g_fin.map(|w| &w.0);
        Self::new(&native_generators, native_g_fin)
    }

    /// Save cache to file with parallel serialization
    pub fn save_to_file(&self, path: &str) -> Result<(), SerializationError> {
        use std::io::Write as StdWrite;

        // Serialize entries in parallel
        let serialized_entries: Vec<Vec<u8>> = self
            .entries
            .par_iter()
            .map(|entry| {
                let mut bytes = Vec::new();
                entry.serialize_compressed(&mut bytes)?;
                Ok(bytes)
            })
            .collect::<Result<Vec<_>, SerializationError>>()?;

        // Now write to file sequentially
        let mut file = std::fs::File::create(path).map_err(|e| SerializationError::IoError(e))?;

        // Write number of entries
        (self.entries.len() as u64).serialize_compressed(&mut file)?;

        // Write each entry with its size
        for entry_bytes in serialized_entries {
            (entry_bytes.len() as u64).serialize_compressed(&mut file)?;
            file.write_all(&entry_bytes)
                .map_err(|e| SerializationError::IoError(e))?;
        }

        // Write precomputed_data
        self.precomputed_data.serialize_compressed(&mut file)?;

        // Write g_fin_glv_tables
        self.g_fin_glv_tables.serialize_compressed(&mut file)?;

        file.flush().map_err(|e| SerializationError::IoError(e))?;
        Ok(())
    }

    /// Load cache from file with parallel deserialization
    pub fn load_from_file(path: &str) -> Result<Self, SerializationError> {
        use std::io::Read as StdRead;

        let mut file = std::fs::File::open(path).map_err(|e| SerializationError::IoError(e))?;

        // Read number of entries
        let num_entries = u64::deserialize_compressed(&mut file)? as usize;

        // Read all entry data first
        let mut entry_data = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            let size = u64::deserialize_compressed(&mut file)? as usize;
            let mut bytes = vec![0u8; size];
            file.read_exact(&mut bytes)
                .map_err(|e| SerializationError::IoError(e))?;
            entry_data.push(bytes);
        }

        // Deserialize entries in parallel
        let entries: Vec<G2CacheEntry> = entry_data
            .into_par_iter()
            .map(|bytes| G2CacheEntry::deserialize_compressed(&bytes[..]))
            .collect::<Result<Vec<_>, SerializationError>>()?;

        // Read precomputed_data
        let precomputed_data = Option::<Windowed2Signed4Data>::deserialize_compressed(&mut file)?;

        // Read g_fin_glv_tables
        let g_fin_glv_tables = Option::<PrecomputedShamir4Data>::deserialize_compressed(&mut file)?;

        Ok(Self {
            entries,
            precomputed_data,
            g_fin_glv_tables,
        })
    }

    /// Get a cache entry by index
    pub fn get_entry(&self, index: usize) -> Option<&G2CacheEntry> {
        self.entries.get(index)
    }

    /// Get the prepared version of a point by index
    pub fn get_prepared(&self, index: usize) -> Option<&BnG2Prepared<ark_bn254::Config>> {
        self.entries.get(index).map(|e| &e.prepared)
    }

    /// Number of cached entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get precomputed windowed2 signed data
    /// Returns a reference to the full windowed data structure
    pub fn get_windowed_data(&self) -> Option<&Windowed2Signed4Data> {
        self.precomputed_data.as_ref()
    }

    /// Get precomputed GLV tables for g_fin if available
    /// Returns a reference to avoid clones
    pub fn get_g_fin_glv_tables(&self) -> Option<&PrecomputedShamir4Data> {
        self.g_fin_glv_tables.as_ref()
    }

    /// Calculate and print memory usage statistics for the cache
    pub fn print_memory_stats(&self) {
        use std::mem;

        println!("=== G2 Cache Memory Usage ===");
        println!("Number of entries: {}", self.entries.len());

        // Calculate entries vector memory
        let entries_capacity = self.entries.capacity();
        let entry_size = mem::size_of::<G2CacheEntry>();
        let entries_allocated = entries_capacity * entry_size;
        let entries_used = self.entries.len() * entry_size;

        println!("\nEntries Vector:");
        println!("  Entry struct size: {} bytes", entry_size);
        println!("  Capacity: {} entries", entries_capacity);
        println!("  Used: {} entries", self.entries.len());
        println!(
            "  Allocated memory: {} bytes ({:.2} MB)",
            entries_allocated,
            entries_allocated as f64 / 1_048_576.0
        );
        println!(
            "  Used memory: {} bytes ({:.2} MB)",
            entries_used,
            entries_used as f64 / 1_048_576.0
        );

        // Get actual sizes of types
        println!("\nType sizes:");
        println!("  G2Affine: {} bytes", mem::size_of::<G2Affine>());
        println!("  G2Projective: {} bytes", mem::size_of::<G2Projective>());
        println!(
            "  BnG2Prepared: {} bytes",
            mem::size_of::<BnG2Prepared<ark_bn254::Config>>()
        );

        // Calculate windowed data memory
        let mut windowed_memory = 0;
        if let Some(windowed_data) = &self.precomputed_data {
            windowed_memory += mem::size_of::<Option<Windowed2Signed4Data>>();

            let tables_len = windowed_data.windowed2_tables.len();
            let tables_capacity = windowed_data.windowed2_tables.capacity();

            // For G2, each windowed table contains 24 G2Projective points for 2-bit windowed method
            // (4 bases with ±P, ±2P, ±3P each)
            let estimated_table_size = 24 * mem::size_of::<G2Projective>();

            let windowed_allocated = tables_capacity * estimated_table_size;
            let windowed_used = tables_len * estimated_table_size;

            println!("\nWindowed2Signed4Data:");
            println!("  Tables count: {}", tables_len);
            println!("  Tables capacity: {}", tables_capacity);
            println!("  Estimated per table: {} bytes", estimated_table_size);
            println!(
                "  Allocated: {} bytes ({:.2} MB)",
                windowed_allocated,
                windowed_allocated as f64 / 1_048_576.0
            );
            println!(
                "  Used: {} bytes ({:.2} MB)",
                windowed_used,
                windowed_used as f64 / 1_048_576.0
            );

            windowed_memory += windowed_allocated;
        } else {
            println!("\nNo windowed precomputed data");
        }

        // Calculate g_fin GLV tables memory
        let mut glv_memory = 0;
        if let Some(glv_tables) = &self.g_fin_glv_tables {
            glv_memory += mem::size_of::<Option<PrecomputedShamir4Data>>();

            // PrecomputedShamir4Data contains tables for GLV precomputation
            // Each table has 15 entries (2^4 - 1) of G2Projective points
            let glv_tables_len = glv_tables.shamir_tables.len();
            let glv_tables_capacity = glv_tables.shamir_tables.capacity();
            let shamir_table_size = 15 * mem::size_of::<G2Projective>();

            let glv_allocated = glv_tables_capacity * shamir_table_size;
            let glv_used = glv_tables_len * shamir_table_size;

            println!("\nG_fin GLV Tables (PrecomputedShamir4Data):");
            println!("  Tables count: {}", glv_tables_len);
            println!("  Tables capacity: {}", glv_tables_capacity);
            println!("  Per table: {} bytes (15 G2Projective)", shamir_table_size);
            println!(
                "  Allocated: {} bytes ({:.2} MB)",
                glv_allocated,
                glv_allocated as f64 / 1_048_576.0
            );
            println!(
                "  Used: {} bytes ({:.2} MB)",
                glv_used,
                glv_used as f64 / 1_048_576.0
            );

            glv_memory += glv_allocated;
        } else {
            println!("\nNo g_fin GLV tables");
        }

        // Total memory
        let total_allocated =
            entries_allocated + windowed_memory + glv_memory + mem::size_of::<Self>();
        println!(
            "\n>>> G2 Cache Total Allocated: {} bytes ({:.2} MB)",
            total_allocated,
            total_allocated as f64 / 1_048_576.0
        );
        println!("=====================================\n");
    }
}

// Create type aliases for cleaner code
pub type G1PrecomputedData = Windowed2Signed2Data;
pub type G2PrecomputedData = Windowed2Signed4Data;

// Optimized MSM implementation using ark-ec's VariableBaseMSM for G1
pub struct OptimizedMsmG1;

impl MultiScalarMul<G1Affine> for OptimizedMsmG1 {
    fn msm(bases: &[G1Affine], scalars: &[Fr]) -> G1Affine {
        if bases.is_empty() {
            return G1Affine::identity();
        }

        use ark_ec::VariableBaseMSM;

        G1Projective::msm(bases, scalars)
            .unwrap_or_else(|_| G1Projective::zero())
            .into_affine()
    }

    fn fixed_base_vector_msm(
        base: &G1Affine,
        scalars: &[Fr],
        g1_cache: Option<&crate::curve::G1Cache>,
        g2_cache: Option<&crate::curve::G2Cache>,
    ) -> Vec<G1Affine> {
        // Caches not used for G1 fixed base vector MSM
        let _ = (g1_cache, g2_cache);

        if scalars.is_empty() {
            return vec![];
        }

        // Convert base to projective for the jolt-optimizations function
        let base_proj = base.into_group();

        // Use jolt-optimizations fixed_base_vector_msm_g1
        let results_proj = jolt_optimizations::fixed_base_vector_msm_g1(&base_proj, scalars);

        // Convert results back to affine
        G1Projective::normalize_batch(&results_proj)
    }

    fn fixed_scalar_variable_with_add(bases: &[G1Affine], vs: &mut [G1Affine], scalar: &Fr) {
        assert_eq!(
            bases.len(),
            vs.len(),
            "bases and vs must have the same length"
        );

        // Convert to projective for computation
        let mut vs_proj: Vec<G1Projective> = vs.iter().map(|v| v.into_group()).collect();
        let bases_proj: Vec<G1Projective> = bases.iter().map(|b| b.into_group()).collect();

        // Use jolt-optimizations function: v[i] = v[i] + scalar * generators[i]
        // Note: We always use online version here. To use precomputed, call the cached version directly
        jolt_optimizations::vector_add_scalar_mul_g1_online(&mut vs_proj, &bases_proj, *scalar);

        // Convert back to affine
        let affines = G1Projective::normalize_batch(&vs_proj);
        for (i, affine) in affines.into_iter().enumerate() {
            vs[i] = affine;
        }
    }

    fn fixed_scalar_variable_with_add_cached(
        bases_count: usize,
        g1_cache: Option<&crate::curve::G1Cache>,
        _g2_cache: Option<&crate::curve::G2Cache>,
        vs: &mut [G1Affine],
        scalar: &Fr,
    ) {
        assert_eq!(bases_count, vs.len(), "bases_count must equal vs length");

        if let Some(cache) = g1_cache {
            if let Some(windowed_data) = cache.get_windowed_data() {
                // Check if we have enough precomputed data
                assert!(
                    bases_count <= windowed_data.windowed2_tables.len(),
                    "Requested bases_count {} exceeds precomputed data size {}",
                    bases_count,
                    windowed_data.windowed2_tables.len()
                );

                // Convert to projective for computation
                let mut vs_proj: Vec<G1Projective> = vs.iter().map(|v| v.into_group()).collect();

                // Create a subset of the windowed data for the required count
                let subset_data = jolt_optimizations::Windowed2Signed2Data {
                    windowed2_tables: windowed_data.windowed2_tables[..bases_count].to_vec(),
                };

                // Use jolt-optimizations function with windowed2 signed data
                vector_add_scalar_mul_g1_windowed2_signed(&mut vs_proj, *scalar, &subset_data);

                // Convert back to affine
                let affines = G1Projective::normalize_batch(&vs_proj);
                for (i, affine) in affines.into_iter().enumerate() {
                    vs[i] = affine;
                }
            } else {
                panic!("Windowed data not available in G1 cache");
            }
        } else {
            // Fall back to extracting bases from cache and using online version
            // This should not happen in practice as we check for cache availability
            panic!("G1 cache not available for cached operation");
        }
    }

    fn fixed_scalar_scale_with_add(vs: &mut [G1Affine], addends: &[G1Affine], scalar: &Fr) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have the same length"
        );

        // Convert to projective for computation
        let mut vs_proj: Vec<G1Projective> = vs.iter().map(|v| v.into_group()).collect();
        let addends_proj: Vec<G1Projective> = addends.iter().map(|a| a.into_group()).collect();

        // Use jolt-optimizations function: v[i] = scalar * v[i] + gamma[i]
        jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
            &mut vs_proj,
            *scalar,
            &addends_proj,
        );

        // Convert back to affine
        let affines = G1Projective::normalize_batch(&vs_proj);
        for (i, affine) in affines.into_iter().enumerate() {
            vs[i] = affine;
        }
    }
}

// Optimized MSM implementation using ark-ec's VariableBaseMSM for G2
pub struct OptimizedMsmG2;

impl MultiScalarMul<G2AffineWrapper> for OptimizedMsmG2 {
    fn msm(bases: &[G2AffineWrapper], scalars: &[Fr]) -> G2AffineWrapper {
        if bases.is_empty() {
            return G2AffineWrapper::identity();
        }

        // Convert wrappers to native G2Affine
        let native_bases: Vec<G2Affine> = bases.iter().map(|b| b.0).collect();

        // Use ark-ec's optimized MSM
        use ark_ec::VariableBaseMSM;

        let result = G2Projective::msm(&native_bases, scalars)
            .unwrap_or_else(|_| G2Projective::zero())
            .into_affine();

        G2AffineWrapper(result)
    }

    fn fixed_base_vector_msm(
        base: &G2AffineWrapper,
        scalars: &[Fr],
        _g1_cache: Option<&crate::curve::G1Cache>,
        g2_cache: Option<&crate::curve::G2Cache>,
    ) -> Vec<G2AffineWrapper> {
        if scalars.is_empty() {
            return vec![];
        }

        // Check if we have cached GLV tables for g_fin
        if let Some(glv_tables) = g2_cache.and_then(|cache| cache.get_g_fin_glv_tables()) {
            // println!("USING PRECOMPUTED GLV TABLES FOR G_FIN!");
            // Use precomputed GLV tables
            let results_proj: Vec<G2Projective> = scalars
                .par_iter()
                .map(|&scalar| {
                    // glv_four_scalar_mul returns a vector, we need the first element
                    jolt_optimizations::glv_four_scalar_mul(glv_tables, scalar)[0]
                })
                .collect();

            // Batch convert to affine
            let affines = G2Projective::normalize_batch(&results_proj);
            affines
                .into_iter()
                .map(|affine| G2AffineWrapper(affine))
                .collect()
        } else {
            // Fall back to online computation
            let base_proj = base.0.into_group();

            // Compute scalar multiplication for each scalar with the fixed base
            let results_proj: Vec<G2Projective> = scalars
                .par_iter()
                .map(|&scalar| {
                    jolt_optimizations::glv_four_scalar_mul_online(scalar, &[base_proj])[0]
                })
                .collect();

            // Batch convert to affine
            let affines = G2Projective::normalize_batch(&results_proj);
            affines
                .into_iter()
                .map(|affine| G2AffineWrapper(affine))
                .collect()
        }
    }

    fn fixed_scalar_variable_with_add(
        bases: &[G2AffineWrapper],
        vs: &mut [G2AffineWrapper],
        scalar: &Fr,
    ) {
        assert_eq!(
            bases.len(),
            vs.len(),
            "bases and vs must have the same length"
        );

        // Convert to projective for computation
        let mut vs_proj: Vec<G2Projective> = vs.iter().map(|v| v.0.into_group()).collect();
        let bases_proj: Vec<G2Projective> = bases.iter().map(|b| b.0.into_group()).collect();

        // Use jolt-optimizations function: v[i] = v[i] + scalar * generators[i]
        jolt_optimizations::vector_add_scalar_mul_g2_online(&mut vs_proj, &bases_proj, *scalar);

        // Convert back to affine wrapper
        let affines = G2Projective::normalize_batch(&vs_proj);
        for (i, affine) in affines.into_iter().enumerate() {
            vs[i] = G2AffineWrapper(affine);
        }
    }

    fn fixed_scalar_variable_with_add_cached(
        bases_count: usize,
        _g1_cache: Option<&crate::curve::G1Cache>,
        g2_cache: Option<&crate::curve::G2Cache>,
        vs: &mut [G2AffineWrapper],
        scalar: &Fr,
    ) {
        assert_eq!(bases_count, vs.len(), "bases_count must equal vs length");

        if let Some(cache) = g2_cache {
            if let Some(windowed_data) = cache.get_windowed_data() {
                // Check if we have enough precomputed data
                assert!(
                    bases_count <= windowed_data.windowed2_tables.len(),
                    "Requested bases_count {} exceeds precomputed data size {}",
                    bases_count,
                    windowed_data.windowed2_tables.len()
                );

                // Convert to projective for computation
                let mut vs_proj: Vec<G2Projective> = vs.iter().map(|v| v.0.into_group()).collect();

                // Create a subset of the windowed data for the required count
                let subset_data = jolt_optimizations::Windowed2Signed4Data {
                    windowed2_tables: windowed_data.windowed2_tables[..bases_count].to_vec(),
                };

                // Use jolt-optimizations function with windowed2 signed data
                vector_add_scalar_mul_g2_windowed2_signed(&mut vs_proj, *scalar, &subset_data);

                // Convert back to affine wrapper
                let affines = G2Projective::normalize_batch(&vs_proj);
                for (i, affine) in affines.into_iter().enumerate() {
                    vs[i] = G2AffineWrapper(affine);
                }
            } else {
                panic!("Windowed data not available in G2 cache");
            }
        } else {
            // Fall back to extracting bases from cache and using online version
            // This should not happen in practice as we check for cache availability
            panic!("G2 cache not available for cached operation");
        }
    }

    fn fixed_scalar_scale_with_add(
        vs: &mut [G2AffineWrapper],
        addends: &[G2AffineWrapper],
        scalar: &Fr,
    ) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have the same length"
        );

        // Convert to projective for computation
        let mut vs_proj: Vec<G2Projective> = vs.iter().map(|v| v.0.into_group()).collect();
        let addends_proj: Vec<G2Projective> = addends.iter().map(|a| a.0.into_group()).collect();

        // Use jolt-optimizations function: v[i] = scalar * v[i] + gamma[i]
        jolt_optimizations::vector_scalar_mul_add_gamma_g2_online(
            &mut vs_proj,
            *scalar,
            &addends_proj,
        );

        // Convert back to affine wrapper
        let affines = G2Projective::normalize_batch(&vs_proj);
        for (i, affine) in affines.into_iter().enumerate() {
            vs[i] = G2AffineWrapper(affine);
        }
    }
}

// Implementation of MultiScalarMul for GT (Fq12) - fallback to dummy since no native MSM
pub struct DummyMsm<G: Group> {
    _phantom: std::marker::PhantomData<G>,
}

impl<G: Group> MultiScalarMul<G> for DummyMsm<G> {
    fn msm(bases: &[G], scalars: &[G::Scalar]) -> G {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "msm requires equal length inputs"
        );
        if bases.is_empty() {
            return G::identity();
        }

        bases
            .iter()
            .zip(scalars.iter())
            .fold(G::identity(), |acc, (base, scalar)| {
                acc.add(&base.scale(scalar))
            })
    }

    fn fixed_base_vector_msm(
        base: &G,
        scalars: &[G::Scalar],
        _g1_cache: Option<&crate::curve::G1Cache>,
        _g2_cache: Option<&crate::curve::G2Cache>,
    ) -> Vec<G> {
        // Naive implementation: compute base * scalar for each scalar
        // Caches are ignored
        scalars.iter().map(|scalar| base.scale(scalar)).collect()
    }

    fn fixed_scalar_variable_with_add(bases: &[G], vs: &mut [G], scalar: &G::Scalar) {
        assert_eq!(
            bases.len(),
            vs.len(),
            "bases and vs must have the same length"
        );

        // Naive implementation: v[i] = v[i] + scalar * bases[i]
        for (i, base) in bases.iter().enumerate() {
            vs[i] = vs[i].add(&base.scale(scalar));
        }
    }

    fn fixed_scalar_scale_with_add(vs: &mut [G], addends: &[G], scalar: &G::Scalar) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have the same length"
        );

        // Naive implementation: v[i] = scalar * v[i] + addends[i]
        for i in 0..vs.len() {
            vs[i] = vs[i].scale(scalar).add(&addends[i]);
        }
    }

    fn fixed_scalar_variable_with_add_cached(
        _bases_count: usize,
        _g1_cache: Option<&crate::curve::G1Cache>,
        _g2_cache: Option<&crate::curve::G2Cache>,
        _vs: &mut [G],
        _scalar: &G::Scalar,
    ) {
        panic!("DummyMsm does not support cached operations");
    }
}

/// Standard polynomial with field elements for testing
#[derive(Clone, Debug, PartialEq)]
pub struct StandardPolynomial<F: Field> {
    pub coeffs: Vec<F>,
}

impl<F: Field> StandardPolynomial<F> {
    pub fn new(coeffs: &[F]) -> Self {
        Self {
            coeffs: coeffs.to_vec(),
        }
    }

    fn get(&self, index: usize) -> F {
        if index < self.coeffs.len() {
            self.coeffs[index]
        } else {
            F::zero()
        }
    }

    /// Evaluates the polynomial at a given point
    pub fn evaluate(&self, point: &[F]) -> F {
        let len = self.coeffs.len();
        let mut eval_vec: Vec<F> = vec![F::zero(); len];

        let expected_size = 1 << point.len();
        assert!(
            len <= expected_size,
            "Too many coefficients: got {}, max for {} variables is {}",
            len,
            point.len(),
            expected_size
        );

        multilinear_lagrange_vec(&mut eval_vec, point);

        // Compute inner product <coeffs, eval_vec>
        let mut result = F::zero();
        for (i, eval) in eval_vec.iter().enumerate() {
            let coeff = self.get(i);
            result = result.add(&coeff.mul(eval));
        }
        result
    }
}

// Implement Polynomial trait for StandardPolynomial
impl<F: Field, G1: Group<Scalar = F>> Polynomial<F, G1> for StandardPolynomial<F> {
    fn len(&self) -> usize {
        self.coeffs.len()
    }

    /// Commits to rows of the polynomial when viewed as a matrix
    fn commit_rows<M1: MultiScalarMul<G1>>(&self, g1_generators: &[G1], row_len: usize) -> Vec<G1> {
        let mut commitments = Vec::new();
        let len = self.coeffs.len();

        let num_rows = (len + row_len - 1) / row_len;
        for row in 0..num_rows {
            let row_start = row * row_len;
            let row_end = (row_start + row_len).min(len);
            let actual_row_len = row_end - row_start;

            if actual_row_len > 0 {
                let mut row_coeffs = vec![F::zero(); actual_row_len];
                for i in 0..actual_row_len {
                    row_coeffs[i] = self.get(row_start + i);
                }
                let commitment = M1::msm(&g1_generators[..actual_row_len], &row_coeffs);
                commitments.push(commitment);
            }
        }

        commitments
    }

    /// Computes the vector-matrix product v = L^T * M where M is the polynomial as a matrix
    ///
    /// # Arguments
    /// * `left_vec` - The L vector (row evaluation weights)
    /// * `sigma` - log₂(columns) - matrix width
    /// * `nu` - log₂(rows) - matrix height
    ///
    /// # Returns
    /// Result vector v where v[j] = sum_i L[i] * M[i,j]
    #[tracing::instrument(skip_all)]
    fn vector_matrix_product(&self, left_vec: &[F], sigma: usize, nu: usize) -> Vec<F>
    where
        Self: Sync,
    {
        use rayon::prelude::*;

        let cols_per_row = 1 << sigma;
        let len = self.coeffs.len();
        let num_rows = (1 << nu).min(left_vec.len());

        if num_rows == 0 {
            return vec![F::zero(); cols_per_row];
        }

        let effective_rows: Vec<(usize, &F)> = (0..num_rows)
            .filter_map(|row_idx| {
                let weight = &left_vec[row_idx];
                if !weight.is_zero() {
                    Some((row_idx, weight))
                } else {
                    None
                }
            })
            .collect();

        if effective_rows.is_empty() {
            return vec![F::zero(); cols_per_row];
        }

        (0..cols_per_row)
            .into_par_iter()
            .map(|col_idx| {
                let mut col_sum = F::zero();

                // Process all contributing rows for this column
                for &(row_idx, l_weight) in &effective_rows {
                    let coeff_idx = row_idx * cols_per_row + col_idx;
                    if coeff_idx < len {
                        let coeff = self.get(coeff_idx);
                        let product = l_weight.mul(&coeff);
                        col_sum = col_sum.add(&product);
                    }
                }

                col_sum
            })
            .collect()
    }
}

/// Create commitment batch, batching factors, and evaluations for verification
/// This provides the values needed for verify_evaluation_proof
pub fn commit_and_evaluate_batch<
    E: Pairing<G1 = G1>,
    M1: MultiScalarMul<G1>,
    F: Field,
    G1: Group<Scalar = F>,
>(
    poly: &StandardPolynomial<F>,
    point: &[F],
    offset: usize,
    sigma: usize,
    prover_setup: &ProverSetup<E>,
) -> (
    Vec<E::GT>, // commitment_batch
    Vec<F>,     // batching_factors
    Vec<F>,     // evaluations
)
where
    F: Field + Clone,
{
    // Compute the commitment to the polynomial
    let (commitment, _) =
        compute_polynomial_commitment::<E, M1, _, F, G1>(poly, offset, sigma, prover_setup);

    // Compute the evaluation of the polynomial at the point
    let evaluation = poly.evaluate(point);

    // For a single polynomial, we use a single batching factor of 1
    let commitment_batch = vec![commitment];

    // @TODO(markosg04): support batching
    let batching_factors = vec![F::one()];
    let evaluations = vec![evaluation]; // for now just one evaluation

    (commitment_batch, batching_factors, evaluations)
}
