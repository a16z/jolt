#![allow(non_snake_case)]
#![allow(clippy::assertions_on_result_states)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::from_over_into)]
#![feature(extend_one)]
#![feature(associated_type_defaults)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(iter_next_chunk)]
#![allow(long_running_const_eval)]
#![allow(clippy::len_without_is_empty)]
#![allow(type_alias_bounds)]
#![allow(clippy::too_long_first_doc_paragraph)]

#[cfg(feature = "host")]
pub mod benches;

#[cfg(feature = "host")]
pub mod host;

pub mod field;
pub mod jolt;
pub mod lasso;
pub mod msm;
pub mod poly;
pub mod r1cs;
pub mod subprotocols;
pub mod utils;
