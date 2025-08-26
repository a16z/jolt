#![allow(non_snake_case)]
#![allow(incomplete_features)]
#![allow(long_running_const_eval)]
#![allow(type_alias_bounds)]
#![allow(clippy::assertions_on_result_states)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::from_over_into)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::too_long_first_doc_paragraph)]

#[cfg(feature = "host")]
pub mod benches;

#[cfg(feature = "host")]
pub mod host;

pub mod field;
pub mod guest;
pub mod msm;
pub mod poly;
pub mod subprotocols;
pub mod transcripts;
pub mod utils;
pub mod zkvm;
pub use ark_bn254;
