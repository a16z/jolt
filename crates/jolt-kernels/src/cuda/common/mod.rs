pub mod context;
pub mod device;
pub mod error;
#[cfg(test)]
pub(crate) mod half_fold;
pub(crate) mod lt_poly;
#[cfg(test)]
pub(crate) mod pack;
pub mod primitives;
#[cfg(test)]
pub(crate) mod ra_poly;
#[cfg(test)]
pub(crate) mod split_eq;
#[cfg(test)]
pub(crate) mod sum_of_products;
#[cfg(test)]
pub(crate) mod testing;
pub mod xfer_stats;
