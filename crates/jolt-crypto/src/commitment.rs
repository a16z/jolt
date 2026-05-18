use std::{
    error::Error,
    fmt::{self, Debug},
};

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use jolt_transcript::AppendToTranscript;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[cfg(feature = "parallel")]
const PAR_THRESHOLD: usize = 1024;

/// Base commitment abstraction: defines only the output type.
///
/// This is the root of the commitment trait hierarchy, shared by both
/// vector commitments ([`VectorCommitment`]) and
/// polynomial commitment schemes (`jolt_openings::CommitmentScheme`).
/// The `Output` associated type is the single piece of connective tissue
/// between these different levels of abstraction.
pub trait Commitment:
    Clone + Debug + Eq + Send + Sync + 'static + Serialize + DeserializeOwned
{
    /// The commitment value (e.g., a group element, a Merkle root, a lattice vector).
    type Output: Clone + Debug + Eq + Send + Sync + 'static + Serialize + DeserializeOwned;
}

/// Backend-agnostic vector commitment.
///
/// Extends [`Commitment`] with the ability to commit to a vector of field
/// elements with a blinding factor. Uses `Self::Output` from the supertrait
/// as the commitment value type.
pub trait VectorCommitment:
    Commitment<Output: Copy + AppendToTranscript + Serialize + DeserializeOwned>
{
    type Field: Field;

    /// Transparent setup parameters (generators, public parameters, etc.).
    type Setup: Clone + Send + Sync;

    /// Maximum number of values this setup can commit to.
    #[must_use]
    fn capacity(setup: &Self::Setup) -> usize;

    /// Commits to `values` with the given `blinding` factor.
    ///
    /// # Panics
    ///
    /// May panic if `values.len()` exceeds [`Self::capacity()`].
    #[must_use]
    fn commit(setup: &Self::Setup, values: &[Self::Field], blinding: &Self::Field) -> Self::Output;

    /// Returns `true` if `commitment` opens to `(values, blinding)`.
    #[must_use]
    fn verify(
        setup: &Self::Setup,
        commitment: &Self::Output,
        values: &[Self::Field],
        blinding: &Self::Field,
    ) -> bool;

    /// Opens a row-major matrix of committed rows at `(row_point, entry_point)`.
    ///
    /// Missing entries at the end of `flattened_rows` are treated as zero.
    fn open_committed_rows(
        flattened_rows: &[Self::Field],
        row_blindings: &[Self::Field],
        row_len: usize,
        row_point: &[Self::Field],
        entry_point: &[Self::Field],
    ) -> Result<(VectorCommitmentOpening<Self::Field>, Self::Field), VectorOpeningError> {
        let row_count = point_len_to_basis_len(row_point.len())?;
        validate_row_len(row_len, entry_point.len())?;
        let max_len = row_count
            .checked_mul(row_len)
            .ok_or(VectorOpeningError::DimensionOverflow)?;
        if flattened_rows.len() > max_len {
            return Err(VectorOpeningError::FlattenedRowsTooLong {
                max: max_len,
                got: flattened_rows.len(),
            });
        }
        if row_blindings.len() != row_count {
            return Err(VectorOpeningError::RowBlindingsLengthMismatch {
                expected: row_count,
                got: row_blindings.len(),
            });
        }

        let row_weights = EqPolynomial::new(row_point.to_vec()).evaluations();
        let entry_weights = EqPolynomial::new(entry_point.to_vec()).evaluations();
        let combined_vector = combine_rows(flattened_rows, row_len, &row_weights, max_len);
        let combined_blinding = inner_product(row_blindings, &row_weights);

        let evaluation = inner_product(&combined_vector, &entry_weights);
        Ok((
            VectorCommitmentOpening {
                combined_vector,
                combined_blinding,
            },
            evaluation,
        ))
    }

    /// Verifies a row-combined opening and returns the evaluation at `entry_point`.
    ///
    fn verify_committed_rows(
        setup: &Self::Setup,
        row_commitments: &[Self::Output],
        row_point: &[Self::Field],
        entry_point: &[Self::Field],
        opening: &VectorCommitmentOpening<Self::Field>,
    ) -> Result<Self::Field, VectorOpeningError>
    where
        Self::Output: HomomorphicCommitment<Self::Field>,
    {
        let row_count = point_len_to_basis_len(row_point.len())?;
        if row_commitments.len() != row_count {
            return Err(VectorOpeningError::RowCommitmentsLengthMismatch {
                expected: row_count,
                got: row_commitments.len(),
            });
        }

        let row_len = opening.combined_vector.len();
        validate_row_len(row_len, entry_point.len())?;
        let capacity = Self::capacity(setup);
        if row_len > capacity {
            return Err(VectorOpeningError::CommitmentCapacityExceeded { capacity, row_len });
        }

        let row_weights = EqPolynomial::new(row_point.to_vec()).evaluations();
        let combined_commitment = combine_commitments(row_commitments, &row_weights);
        if !Self::verify(
            setup,
            &combined_commitment,
            &opening.combined_vector,
            &opening.combined_blinding,
        ) {
            return Err(VectorOpeningError::CommitmentMismatch);
        }

        let entry_weights = EqPolynomial::new(entry_point.to_vec()).evaluations();
        Ok(inner_product(&opening.combined_vector, &entry_weights))
    }
}

/// Opening data for a linear combination of committed rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorCommitmentOpening<F> {
    pub combined_vector: Vec<F>,
    pub combined_blinding: F,
}

/// Errors returned by committed-row opening helpers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VectorOpeningError {
    RowLenZero,
    RowLenNotPowerOfTwo { row_len: usize },
    PointTooLarge { point_len: usize },
    EntryPointLengthMismatch { expected: usize, got: usize },
    RowBlindingsLengthMismatch { expected: usize, got: usize },
    RowCommitmentsLengthMismatch { expected: usize, got: usize },
    FlattenedRowsTooLong { max: usize, got: usize },
    CommitmentCapacityExceeded { capacity: usize, row_len: usize },
    DimensionOverflow,
    CommitmentMismatch,
}

impl fmt::Display for VectorOpeningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowLenZero => write!(f, "row length must be non-zero"),
            Self::RowLenNotPowerOfTwo { row_len } => {
                write!(f, "row length {row_len} is not a power of two")
            }
            Self::PointTooLarge { point_len } => {
                write!(
                    f,
                    "point length {point_len} does not fit in usize dimensions"
                )
            }
            Self::EntryPointLengthMismatch { expected, got } => write!(
                f,
                "entry point length mismatch: expected {expected}, got {got}"
            ),
            Self::RowBlindingsLengthMismatch { expected, got } => write!(
                f,
                "row blinding count mismatch: expected {expected}, got {got}"
            ),
            Self::RowCommitmentsLengthMismatch { expected, got } => write!(
                f,
                "row commitment count mismatch: expected {expected}, got {got}"
            ),
            Self::FlattenedRowsTooLong { max, got } => {
                write!(f, "flattened row data has length {got}, maximum is {max}")
            }
            Self::CommitmentCapacityExceeded { capacity, row_len } => write!(
                f,
                "row length {row_len} exceeds commitment capacity {capacity}"
            ),
            Self::DimensionOverflow => write!(f, "vector opening dimensions overflow usize"),
            Self::CommitmentMismatch => write!(f, "combined vector commitment does not match rows"),
        }
    }
}

impl Error for VectorOpeningError {}

/// Additive homomorphism on commitment values over a scalar field `F`.
///
/// Captures the ability to linearly combine two commitments without
/// knowing the committed values:
/// ```text
/// linear_combine(c1, c2, s) = c1 ⊕ s ⊗ c2
/// ```
///
/// Required by Nova folding for instance-level commitment operations.
/// Not all commitment schemes have this property (e.g., hash-based schemes
/// do not). Pedersen and lattice-based schemes do.
///
/// Blanket-implemented for [`JoltGroup`](crate::JoltGroup) over any field
/// (via `scalar_mul` + addition). Non-group commitment types (e.g., lattice
/// vectors) can implement this trait directly for their native scalar field.
pub trait HomomorphicCommitment<F: Field>: Clone {
    /// Returns the additive identity commitment.
    #[must_use]
    fn identity() -> Self;

    /// Computes `c1 + scalar * c2`.
    #[must_use]
    fn linear_combine(c1: &Self, c2: &Self, scalar: &F) -> Self;
}

/// Derives a commitment setup from a source setup (e.g., PCS SRS → Pedersen generators).
///
/// This is the bridge between a polynomial commitment scheme's structured
/// reference string and a vector commitment's setup parameters. Each PCS
/// implements this for the vector commitment setups it can derive.
///
/// Backend-agnostic: works for EC (Pedersen from Dory/KZG SRS), lattice
/// (matrix columns from lattice SRS), or hash-based (Merkle params) schemes.
pub trait DeriveSetup<Source> {
    fn derive(source: &Source, capacity: usize) -> Self;
}

fn validate_row_len(row_len: usize, entry_point_len: usize) -> Result<(), VectorOpeningError> {
    if row_len == 0 {
        return Err(VectorOpeningError::RowLenZero);
    }
    if !row_len.is_power_of_two() {
        return Err(VectorOpeningError::RowLenNotPowerOfTwo { row_len });
    }
    let expected = row_len.trailing_zeros() as usize;
    if entry_point_len != expected {
        return Err(VectorOpeningError::EntryPointLengthMismatch {
            expected,
            got: entry_point_len,
        });
    }
    Ok(())
}

fn point_len_to_basis_len(point_len: usize) -> Result<usize, VectorOpeningError> {
    if point_len >= usize::BITS as usize {
        return Err(VectorOpeningError::PointTooLarge { point_len });
    }
    Ok(1usize << point_len)
}

#[cfg(feature = "parallel")]
fn combine_rows<F: Field>(
    flattened_rows: &[F],
    row_len: usize,
    row_weights: &[F],
    max_len: usize,
) -> Vec<F> {
    let mut combined_vector = vec![F::zero(); row_len];

    if max_len >= PAR_THRESHOLD {
        use rayon::prelude::*;

        combined_vector
            .par_iter_mut()
            .enumerate()
            .for_each(|(entry_index, combined_entry)| {
                let mut acc = F::zero();
                for (row_index, row_weight) in row_weights.iter().copied().enumerate() {
                    if let Some(value) = flattened_rows.get(row_index * row_len + entry_index) {
                        acc += row_weight * *value;
                    }
                }
                *combined_entry = acc;
            });
    } else {
        for (entry_index, combined_entry) in combined_vector.iter_mut().enumerate() {
            let mut acc = F::zero();
            for (row_index, row_weight) in row_weights.iter().copied().enumerate() {
                if let Some(value) = flattened_rows.get(row_index * row_len + entry_index) {
                    acc += row_weight * *value;
                }
            }
            *combined_entry = acc;
        }
    }

    combined_vector
}

#[cfg(not(feature = "parallel"))]
fn combine_rows<F: Field>(
    flattened_rows: &[F],
    row_len: usize,
    row_weights: &[F],
    _max_len: usize,
) -> Vec<F> {
    let mut combined_vector = vec![F::zero(); row_len];

    for (entry_index, combined_entry) in combined_vector.iter_mut().enumerate() {
        let mut acc = F::zero();
        for (row_index, row_weight) in row_weights.iter().copied().enumerate() {
            if let Some(value) = flattened_rows.get(row_index * row_len + entry_index) {
                acc += row_weight * *value;
            }
        }
        *combined_entry = acc;
    }

    combined_vector
}

fn inner_product<F: Field>(lhs: &[F], rhs: &[F]) -> F {
    #[cfg(feature = "parallel")]
    {
        if lhs.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;

            return lhs
                .par_iter()
                .zip(rhs.par_iter())
                .map(|(left, right)| *left * *right)
                .sum();
        }
    }

    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| *left * *right)
        .sum()
}

fn combine_commitments<F, C>(commitments: &[C], weights: &[F]) -> C
where
    F: Field,
    C: HomomorphicCommitment<F> + Copy + Send + Sync,
{
    #[cfg(feature = "parallel")]
    {
        if commitments.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;

            return commitments
                .par_iter()
                .zip(weights.par_iter())
                .map(|(commitment, weight)| C::linear_combine(&C::identity(), commitment, weight))
                .reduce(C::identity, |left, right| {
                    C::linear_combine(&left, &right, &F::one())
                });
        }
    }

    commitments
        .iter()
        .zip(weights.iter())
        .fold(C::identity(), |acc, (commitment, weight)| {
            C::linear_combine(&acc, commitment, weight)
        })
}

impl<G: crate::JoltGroup, F: Field> HomomorphicCommitment<F> for G {
    #[inline]
    fn identity() -> G {
        G::identity()
    }

    #[inline]
    fn linear_combine(c1: &G, c2: &G, scalar: &F) -> G {
        *c1 + c2.scalar_mul(scalar)
    }
}
