//! jolt-akita's integration tests, linked as one binary so the Akita prover
//! and its dependencies are compiled and linked once rather than per file.

mod support;

mod e2e;
mod external_catalog;
mod fold_paths;
mod grouped_capacity;
mod native_batching;
mod one_hot_groups;
mod pathologies;
mod prefix_packing;
mod runtime_chunking;
mod schedules;
mod scheme_surface;
