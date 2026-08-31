pub const OUTER_REMAINDER_OPENINGS: usize = 35;
pub(super) const OUTER_REMAINDER_PRODUCT_ENDPOINTS: usize = 2;
pub(super) const OUTER_REMAINDER_MAX_OUTPUTS: usize =
    OUTER_REMAINDER_OPENINGS + OUTER_REMAINDER_PRODUCT_ENDPOINTS;
pub(super) const OUTER_REMAINDER_STREAM_ROWS: usize = 10;
pub(super) const OUTER_REMAINDER_COLLAPSED_A_FIELDS: usize = 96;
pub(super) const OUTER_REMAINDER_FIRST_B_FIELDS: usize = 13;
pub(super) const OUTER_REMAINDER_SECOND_B_FIELDS: usize = 15;
pub(super) const OUTER_REMAINDER_A_LOOKUP_FIELDS: usize = OUTER_REMAINDER_STREAM_ROWS
    + 2 * OUTER_REMAINDER_COLLAPSED_A_FIELDS
    + OUTER_REMAINDER_FIRST_B_FIELDS
    + OUTER_REMAINDER_SECOND_B_FIELDS;
pub(super) const DEVICE_BUFFERS: usize = 9;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OuterRemainderSequenceConfig {
    pub materialize_threads_per_threadgroup: Option<usize>,
    pub stream_bind_threads_per_threadgroup: Option<usize>,
    pub transition_threads_per_threadgroup: Option<usize>,
    pub opening_threads_per_threadgroup: Option<usize>,
    pub max_threadgroups: usize,
    pub cpu_tail_elements: usize,
    pub storage_initialization: OuterRemainderStorageInitialization,
    pub product_uniskip_carrier: bool,
    pub registers_claim_carrier: bool,
}

impl Default for OuterRemainderSequenceConfig {
    fn default() -> Self {
        Self {
            materialize_threads_per_threadgroup: Some(256),
            stream_bind_threads_per_threadgroup: Some(128),
            transition_threads_per_threadgroup: Some(128),
            opening_threads_per_threadgroup: Some(256),
            max_threadgroups: 8192,
            cpu_tail_elements: 1 << 18,
            storage_initialization: OuterRemainderStorageInitialization::Full,
            product_uniskip_carrier: false,
            registers_claim_carrier: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OuterRemainderStorageInitialization {
    Lazy,
    Full,
}

impl OuterRemainderStorageInitialization {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lazy => "lazy",
            Self::Full => "full",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OuterRemainderPhase {
    BeforeMaterialize,
    BOnly,
    Interleaved,
    Exported,
    OpeningsComplete,
    Poisoned,
}

impl OuterRemainderPhase {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::BeforeMaterialize => "before materialization",
            Self::BOnly => "B-only",
            Self::Interleaved => "interleaved",
            Self::Exported => "CPU tail exported",
            Self::OpeningsComplete => "openings complete",
            Self::Poisoned => "poisoned",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OuterRemainderStorageStats {
    pub owned_bytes: u64,
    pub buffer_identities: [usize; DEVICE_BUFFERS],
    pub compact_row_identity: usize,
    pub residual_row_identity: usize,
    pub cold_row_identity: Option<usize>,
    pub row_device_registry_id: u64,
}
