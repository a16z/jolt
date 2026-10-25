pub(crate) mod address_first;
pub(crate) mod address_major_matrix;
pub(crate) mod address_rounds;
pub mod context;
pub(crate) mod dense_product;
pub mod device;
pub(crate) mod device_columns;
pub(crate) mod devices;
pub mod error;
pub(crate) mod half_fold;
pub(crate) mod lt_poly;
pub(crate) mod msm;
pub(crate) mod one_hot_fold;
#[cfg(test)]
pub(crate) mod one_hot_witness;
pub(crate) mod pack;
#[cfg(test)]
pub(crate) mod pairing;
pub(crate) mod precommitted_reduction;
pub(crate) mod prefix_suffix;
pub mod primitives;
pub(crate) mod ra_poly;
pub(crate) mod read_write_matrix;
pub(crate) mod split_eq;
pub(crate) mod sum_of_products;
#[cfg(test)]
pub(crate) mod testing;
pub(crate) mod unreduced;
pub mod xfer_stats;
