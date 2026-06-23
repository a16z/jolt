use core::{
    fmt::{self, Debug},
    hash::{Hash, Hasher},
};

/// Label identifying which protocol a witness item belongs to; appears in
/// every [`crate::WitnessError`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NamespaceId {
    pub name: &'static str,
}

impl NamespaceId {
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }
}

/// Type-level identity of one protocol's witness taxonomy.
///
/// Implemented by uninhabited enums (no values exist; all dispatch is
/// compile-time). The associated types bind the protocol's polynomial,
/// opening, public-input, and challenge identifier spaces, so oracles from
/// different protocols cannot be mixed up.
pub trait WitnessNamespace {
    type CommittedId: Copy + Debug + Eq + Hash;
    type VirtualId: Copy + Debug + Eq + Hash;
    type OpeningId: Copy + Debug + Eq + Hash;
    type PublicId: Copy + Debug + Eq + Hash;
    type ChallengeId: Copy + Debug + Eq + Hash;

    const ID: NamespaceId;
}

/// Typed reference to a single polynomial within namespace `N`.
///
/// Committed polynomials go through the commitment scheme; virtual
/// polynomials are derived during proving and never committed.
pub enum OracleRef<N: WitnessNamespace> {
    Committed(N::CommittedId),
    Virtual(N::VirtualId),
}

// Manual impls: deriving would incorrectly require traits on `N` even
// though `N` only appears through its associated types.
impl<N: WitnessNamespace> Debug for OracleRef<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Committed(id) => f.debug_tuple("Committed").field(id).finish(),
            Self::Virtual(id) => f.debug_tuple("Virtual").field(id).finish(),
        }
    }
}

impl<N: WitnessNamespace> Clone for OracleRef<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N: WitnessNamespace> Copy for OracleRef<N> {}

impl<N: WitnessNamespace> PartialEq for OracleRef<N> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Committed(lhs), Self::Committed(rhs)) => lhs == rhs,
            (Self::Virtual(lhs), Self::Virtual(rhs)) => lhs == rhs,
            _ => false,
        }
    }
}

impl<N: WitnessNamespace> Eq for OracleRef<N> {}

impl<N: WitnessNamespace> Hash for OracleRef<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Committed(id) => {
                0_u8.hash(state);
                id.hash(state);
            }
            Self::Virtual(id) => {
                1_u8.hash(state);
                id.hash(state);
            }
        }
    }
}

impl<N: WitnessNamespace> OracleRef<N> {
    pub const fn committed(id: N::CommittedId) -> Self {
        Self::Committed(id)
    }

    pub const fn virtual_polynomial(id: N::VirtualId) -> Self {
        Self::Virtual(id)
    }
}
