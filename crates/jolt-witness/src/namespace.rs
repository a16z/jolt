use core::{fmt::Debug, hash::Hash};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NamespaceId {
    pub name: &'static str,
}

impl NamespaceId {
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }
}

pub trait WitnessNamespace {
    type CommittedId: Copy + Debug + Eq + Hash;
    type VirtualId: Copy + Debug + Eq + Hash;
    type OpeningId: Copy + Debug + Eq + Hash;
    type PublicId: Copy + Debug + Eq + Hash;
    type ChallengeId: Copy + Debug + Eq + Hash;

    const ID: NamespaceId;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OracleKind<C, V> {
    Committed(C),
    Virtual(V),
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct OracleRef<N: WitnessNamespace> {
    pub kind: OracleKind<N::CommittedId, N::VirtualId>,
}

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
