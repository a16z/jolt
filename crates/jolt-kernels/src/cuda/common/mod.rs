#[cfg(test)]
pub(crate) mod address_first;
#[cfg(test)]
pub(crate) mod address_major_matrix;
pub mod context;
#[cfg(test)]
pub(crate) mod dense_product;
pub mod device;
pub(crate) mod device_columns;
pub(crate) mod devices;
pub mod error;
pub(crate) mod half_fold;
pub(crate) mod lt_poly;
#[cfg(test)]
pub(crate) mod msm;
#[cfg(test)]
pub(crate) mod one_hot_fold;
#[cfg(test)]
pub(crate) mod pack;
#[cfg(test)]
pub(crate) mod pairing;
pub(crate) mod precommitted_reduction;
pub(crate) mod prefix_suffix;
pub mod primitives;
#[cfg(test)]
pub(crate) mod ra_poly;
#[cfg(test)]
pub(crate) mod read_write_matrix;
pub(crate) mod split_eq;
pub(crate) mod sum_of_products;
#[cfg(test)]
pub(crate) mod testing;
#[cfg(test)]
pub(crate) mod unreduced;
pub mod xfer_stats;
