use core::{fmt::Debug, hash::Hash};

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

/// Committed polynomials go through the commitment scheme; virtual
/// polynomials are derived during proving and never committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OracleKind<C, V> {
    Committed(C),
    Virtual(V),
}

/// Typed reference to a single polynomial within namespace `N`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct OracleRef<N: WitnessNamespace> {
    pub kind: OracleKind<N::CommittedId, N::VirtualId>,
}

// Manual impls: deriving would incorrectly require `N: Clone + Copy` even
// though `N` only appears through its associated types.
impl<N: WitnessNamespace> Clone for OracleRef<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N: WitnessNamespace> Copy for OracleRef<N> {}

impl<N: WitnessNamespace> OracleRef<N> {
    pub const fn committed(id: N::CommittedId) -> Self {
        Self {
            kind: OracleKind::Committed(id),
        }
    }

    pub const fn virtual_polynomial(id: N::VirtualId) -> Self {
        Self {
            kind: OracleKind::Virtual(id),
        }
    }
}
