use crate::circuit::utils;
use maybe_rayon::{
    prelude::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    error::Error,
    fmt::Debug,
    iter::Iterator,
    ops::{Add, Deref, DerefMut, Div, Mul, Neg, Range, Sub},
};
use thiserror::Error;
use tract_onnx::prelude::tract_itertools::Itertools;

/// Implementations of common operations on tensors.
pub mod ops;

/// A generic multi-dimensional array representation of a Tensor.
/// The `inner` attribute contains a vector of values whereas `dims` corresponds to
/// the dimensionality of the array and as such determines how we index, query for
/// values, or slice a Tensor.
#[derive(Clone, Debug, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Tensor<T: TensorType> {
    inner: Vec<T>,
    dims: Vec<usize>,
    scale: Option<crate::Scale>,
}

/// A wrapper for tensor related errors.
#[derive(Debug, Error)]
pub enum TensorError {
    /// Shape mismatch in a operation
    #[error("dimension mismatch in tensor op: {0}")]
    DimMismatch(String),
    /// Shape when instantiating
    #[error("dimensionality error when manipulating a tensor: {0}")]
    DimError(String),
    /// wrong method was called on a tensor-like struct
    #[error("wrong method called")]
    WrongMethod,
    /// Significant bit truncation when instantiating
    #[error("Significant bit truncation when instantiating, try lowering the scale")]
    SigBitTruncationError,
    /// Failed to convert to field element tensor
    #[error("Failed to convert to field element tensor")]
    FeltError,
    /// Table lookup error
    #[error("Table lookup error")]
    TableLookupError,
    /// Unsupported operation
    #[error("Unsupported operation on a tensor type")]
    Unsupported,
}

/// The (inner) type of tensor elements.
pub trait TensorType: Clone + Debug + 'static {
    /// Returns the zero value.
    fn zero() -> Option<Self> {
        None
    }
    /// Returns the unit value.
    fn one() -> Option<Self> {
        None
    }
    /// Max operator for ordering values.
    fn tmax(&self, _: &Self) -> Option<Self> {
        None
    }
}

macro_rules! tensor_type {
    ($rust_type:ty, $tensor_type:ident, $zero:expr, $one:expr) => {
        impl TensorType for $rust_type {
            fn zero() -> Option<Self> {
                Some($zero)
            }
            fn one() -> Option<Self> {
                Some($one)
            }

            fn tmax(&self, other: &Self) -> Option<Self> {
                Some(max(*self, *other))
            }
        }
    };
}

impl TensorType for f32 {
    fn zero() -> Option<Self> {
        Some(0.0)
    }

    // f32 doesnt impl Ord so we cant just use max like we can for i32, usize.
    // A comparison between f32s needs to handle NAN values.
    fn tmax(&self, other: &Self) -> Option<Self> {
        match (self.is_nan(), other.is_nan()) {
            (true, true) => Some(f32::NAN),
            (true, false) => Some(*other),
            (false, true) => Some(*self),
            (false, false) => {
                if self >= other {
                    Some(*self)
                } else {
                    Some(*other)
                }
            }
        }
    }
}

impl TensorType for f64 {
    fn zero() -> Option<Self> {
        Some(0.0)
    }

    // f32 doesnt impl Ord so we cant just use max like we can for i32, usize.
    // A comparison between f32s needs to handle NAN values.
    fn tmax(&self, other: &Self) -> Option<Self> {
        match (self.is_nan(), other.is_nan()) {
            (true, true) => Some(f64::NAN),
            (true, false) => Some(*other),
            (false, true) => Some(*self),
            (false, false) => {
                if self >= other {
                    Some(*self)
                } else {
                    Some(*other)
                }
            }
        }
    }
}

tensor_type!(bool, Bool, false, true);
tensor_type!(i128, Int128, 0, 1);
tensor_type!(u128, UInt128, 0, 1);
tensor_type!(i32, Int32, 0, 1);
tensor_type!(usize, USize, 0, 1);
tensor_type!((), Empty, (), ());
tensor_type!(utils::F32, F32, utils::F32(0.0), utils::F32(1.0));

impl<T: TensorType> TensorType for Tensor<T> {
    fn zero() -> Option<Self> {
        Some(Tensor::new(Some(&[T::zero().unwrap()]), &[1]).unwrap())
    }
    fn one() -> Option<Self> {
        Some(Tensor::new(Some(&[T::one().unwrap()]), &[1]).unwrap())
    }
}

// specific types
impl TensorType for halo2curves::pasta::Fp {
    fn zero() -> Option<Self> {
        Some(halo2curves::pasta::Fp::zero())
    }

    fn one() -> Option<Self> {
        Some(halo2curves::pasta::Fp::one())
    }

    fn tmax(&self, other: &Self) -> Option<Self> {
        Some((*self).max(*other))
    }
}

impl TensorType for halo2curves::bn256::Fr {
    fn zero() -> Option<Self> {
        Some(halo2curves::bn256::Fr::zero())
    }

    fn one() -> Option<Self> {
        Some(halo2curves::bn256::Fr::one())
    }

    fn tmax(&self, other: &Self) -> Option<Self> {
        Some((*self).max(*other))
    }
}

impl<T: TensorType> IntoIterator for Tensor<T> {
    type Item = T;
    type IntoIter = ::std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<T: TensorType> Deref for Tensor<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.inner.deref()
    }
}

impl<T: TensorType> DerefMut for Tensor<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.inner.deref_mut()
    }
}

impl<T: PartialEq + TensorType> PartialEq for Tensor<T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        self.dims == other.dims && self.deref() == other.deref()
    }
}

impl<I: Iterator> From<I> for Tensor<I::Item>
where
    I::Item: TensorType + Clone,
    Vec<I::Item>: FromIterator<I::Item>,
{
    fn from(value: I) -> Tensor<I::Item> {
        let data: Vec<I::Item> = value.collect::<Vec<I::Item>>();
        Tensor::new(Some(&data), &[data.len()]).unwrap()
    }
}

impl<T> FromIterator<T> for Tensor<T>
where
    T: TensorType + Clone,
    Vec<T>: FromIterator<T>,
{
    fn from_iter<I: IntoIterator<Item = T>>(value: I) -> Tensor<T> {
        let data: Vec<I::Item> = value.into_iter().collect::<Vec<I::Item>>();
        Tensor::new(Some(&data), &[data.len()]).unwrap()
    }
}

impl<T: Clone + TensorType + std::marker::Send + std::marker::Sync>
    maybe_rayon::iter::IntoParallelIterator for Tensor<T>
{
    type Iter = maybe_rayon::vec::IntoIter<T>;
    type Item = T;
    fn into_par_iter(self) -> Self::Iter {
        self.inner.into_par_iter()
    }
}

impl<'data, T: Clone + TensorType + std::marker::Send + std::marker::Sync>
    maybe_rayon::iter::IntoParallelRefMutIterator<'data> for Tensor<T>
{
    type Iter = maybe_rayon::slice::IterMut<'data, T>;
    type Item = &'data mut T;
    fn par_iter_mut(&'data mut self) -> Self::Iter {
        self.inner.par_iter_mut()
    }
}

impl<T: Clone + TensorType> Tensor<T> {
    /// Sets (copies) the tensor values to the provided ones.
    pub fn new(values: Option<&[T]>, dims: &[usize]) -> Result<Self, TensorError> {
        let total_dims: usize = if !dims.is_empty() {
            dims.iter().product()
        } else if values.is_some() {
            1
        } else {
            0
        };
        match values {
            Some(v) => {
                if total_dims != v.len() {
                    return Err(TensorError::DimError(format!(
                        "Cannot create tensor of length {} with dims {:?}",
                        v.len(),
                        dims
                    )));
                }
                Ok(Tensor {
                    inner: Vec::from(v),
                    dims: Vec::from(dims),
                    scale: None,
                })
            }
            None => Ok(Tensor {
                inner: vec![T::zero().unwrap(); total_dims],
                dims: Vec::from(dims),
                scale: None,
            }),
        }
    }

    /// set the tensor's (optional) scale parameter
    pub fn set_scale(&mut self, scale: crate::Scale) {
        self.scale = Some(scale)
    }

    /// getter for scale
    pub fn scale(&self) -> Option<crate::Scale> {
        self.scale
    }

    /// Returns the number of elements in the tensor.
    pub fn len(&self) -> usize {
        self.dims().iter().product::<usize>()
    }

    /// Checks if the number of elements in tensor is 0.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Checks if the number of elements in tensor is 1 but with an empty dimension
    /// (this is for onnx compatibility).
    pub fn is_singleton(&self) -> bool {
        self.dims().is_empty() && self.len() == 1
    }

    /// Set one single value on the tensor.
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(None, &[3, 3, 3]).unwrap();
    ///
    /// a.set(&[0, 0, 1], 10);
    /// assert_eq!(a[0 + 0 + 1], 10);
    ///
    /// a.set(&[2, 2, 0], 9);
    /// assert_eq!(a[2*9 + 2*3 + 0], 9);
    /// ```
    pub fn set(&mut self, indices: &[usize], value: T) {
        let index = self.get_index(indices);
        self[index] = value;
    }

    /// Get a single value from the Tensor.
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(None, &[2, 3, 5]).unwrap();
    ///
    /// a[1*15 + 1*5 + 1] = 5;
    /// assert_eq!(a.get(&[1, 1, 1]), 5);
    /// ```
    pub fn get(&self, indices: &[usize]) -> T {
        let index = self.get_index(indices);
        self[index].clone()
    }

    /// Get a mutable array index from rows / columns indices.
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(None, &[2, 3, 5]).unwrap();
    ///
    /// a[1*15 + 1*5 + 1] = 5;
    /// assert_eq!(a.get(&[1, 1, 1]), 5);
    /// ```
    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        assert_eq!(self.dims.len(), indices.len());
        let mut index = 0;
        let mut d = 1;
        for i in (0..indices.len()).rev() {
            assert!(self.dims[i] > indices[i]);
            index += indices[i] * d;
            d *= self.dims[i];
        }
        &mut self[index]
    }

    /// Pad to a length that is divisible by n
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1,2,3,4,5,6]), &[2, 3]).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6, 0, 0]), &[8]).unwrap();     /// assert_eq!(a.pad_to_zero_rem(4).unwrap(), expected);
    ///
    /// let expected = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6, 0, 0, 0]), &[9]).unwrap();     /// assert_eq!(a.pad_to_zero_rem(9).unwrap(), expected);
    /// ```
    pub fn pad_to_zero_rem(&self, n: usize) -> Result<Tensor<T>, TensorError> {
        let mut inner = self.inner.clone();
        let remainder = self.len() % n;
        if remainder != 0 {
            inner.resize(self.len() + n - remainder, T::zero().unwrap());
        }
        Tensor::new(Some(&inner), &[inner.len()])
    }

    /// Get a single value from the Tensor.
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(None, &[2, 3, 5]).unwrap();
    ///
    /// let flat_index = 1*15 + 1*5 + 1;
    /// a[1*15 + 1*5 + 1] = 5;
    /// assert_eq!(a.get_flat_index(flat_index), 5);
    /// ```
    pub fn get_flat_index(&self, index: usize) -> T {
        self[index].clone()
    }

    /// Display a tensor
    pub fn show(&self) -> String {
        if self.len() > 12 {
            let start = self[..12].to_vec();
            // print the two split by ... in the middle
            format!(
                "[{} ...]",
                start.iter().map(|x| format!("{x:?}")).join(", "),
            )
        } else {
            format!("[{:?}]", self.iter().map(|x| format!("{x:?}")).join(", "))
        }
    }

    /// Get a slice from the Tensor.
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3]).unwrap();
    /// let mut b = Tensor::<i32>::new(Some(&[1, 2]), &[2]).unwrap();
    ///
    /// assert_eq!(a.get_slice(&[0..2]).unwrap(), b);
    /// ```
    pub fn get_slice(&self, indices: &[Range<usize>]) -> Result<Tensor<T>, TensorError>
    where
        T: Send + Sync,
    {
        if indices.is_empty() {
            return Ok(self.clone());
        }
        if self.dims.len() < indices.len() {
            return Err(TensorError::DimError(format!(
                "The dimensionality of the slice {:?} is greater than the tensor's
  {:?}",
                indices, self.dims
            )));
        } else if indices.iter().map(|x| x.end - x.start).collect::<Vec<_>>() == self.dims {
            // else if slice is the same as dims, return self
            return Ok(self.clone());
        }

        // if indices weren't specified we fill them in as required
        let mut full_indices = indices.to_vec();

        for i in 0..(self.dims.len() - indices.len()) {
            full_indices.push(0..self.dims()[indices.len() + i])
        }

        let cartesian_coord: Vec<Vec<usize>> = full_indices
            .iter()
            .cloned()
            .multi_cartesian_product()
            .collect();

        let res: Vec<T> = cartesian_coord
            .par_iter()
            .map(|e| {
                let index = self.get_index(e);
                self[index].clone()
            })
            .collect();

        let dims: Vec<usize> = full_indices.iter().map(|e| e.end - e.start).collect();

        Tensor::new(Some(&res), &dims)
    }

    /// Get the array index from rows / columns indices.
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    ///
    /// assert_eq!(a.get_index(&[2, 2, 2]), 26);
    /// assert_eq!(a.get_index(&[1, 2, 2]), 17);
    /// assert_eq!(a.get_index(&[1, 2, 0]), 15);
    /// assert_eq!(a.get_index(&[1, 0, 1]), 10);
    /// ```
    pub fn get_index(&self, indices: &[usize]) -> usize {
        assert_eq!(self.dims.len(), indices.len());
        let mut index = 0;
        let mut d = 1;
        for i in (0..indices.len()).rev() {
            assert!(self.dims[i] > indices[i]);
            index += indices[i] * d;
            d *= self.dims[i];
        }
        index
    }

    /// Duplicates every nth element
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[6]).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[1, 2, 3, 3, 4, 5, 5, 6]), &[8]).unwrap();     /// assert_eq!(a.duplicate_every_n(3, 1, 0).unwrap(), expected);     /// assert_eq!(a.duplicate_every_n(7, 1, 0).unwrap(), a);
    ///
    /// let expected = Tensor::<i32>::new(Some(&[1, 1, 2, 3, 3, 4, 5, 5, 6]), &[9]).unwrap();     /// assert_eq!(a.duplicate_every_n(3, 1, 2).unwrap(),expected);     ///
    /// ```
    pub fn duplicate_every_n(
        &self,
        n: usize,
        num_repeats: usize,
        initial_offset: usize,
    ) -> Result<Tensor<T>, TensorError> {
        let mut inner: Vec<T> = vec![];
        let mut offset = initial_offset;
        for (i, elem) in self.inner.clone().into_iter().enumerate() {
            if (i + offset + 1) % n == 0 {
                inner.extend(vec![elem; 1 + num_repeats]);
                offset += num_repeats;
            } else {
                inner.push(elem.clone());
            }
        }
        Tensor::new(Some(&inner), &[inner.len()])
    }

    /// Removes every nth element
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let a = Tensor::<i32>::new(Some(&[1, 2, 3, 3, 4, 5, 6, 6]), &[8]).unwrap();     /// let expected = Tensor::<i32>::new(Some(&[1, 2, 3, 3, 5, 6, 6]), &[7]).unwrap();     /// assert_eq!(a.remove_every_n(4, 1, 0).unwrap(), expected);     ///
    pub fn remove_every_n(
        &self,
        n: usize,
        num_repeats: usize,
        initial_offset: usize,
    ) -> Result<Tensor<T>, TensorError> {
        let mut inner: Vec<T> = vec![];
        let mut indices_to_remove = std::collections::HashSet::new();
        for i in 0..self.inner.len() {
            if (i + initial_offset + 1) % n == 0 {
                for j in 1..(1 + num_repeats) {
                    indices_to_remove.insert(i + j);
                }
            }
        }

        let old_inner = self.inner.clone();
        for (i, elem) in old_inner.into_iter().enumerate() {
            if !indices_to_remove.contains(&i) {
                inner.push(elem.clone());
            }
        }

        Tensor::new(Some(&inner), &[inner.len()])
    }

    /// Remove indices
    /// WARN: assumes indices are in ascending order for speed
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[6]).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[1, 2, 3, 6]), &[4]).unwrap();
    /// let mut indices = vec![3, 4];
    /// assert_eq!(a.remove_indices(&mut indices, true).unwrap(), expected);
    ///
    ///
    /// let a = Tensor::<i32>::new(Some(&[52, -245, 153, 13, -4, -56, -163, 249,
    /// -128, -172, 396, 143, 2, -96, 504, -44, -158, -393, 61, 95, 191, 74, 64, -219,
    ///  553, 104, 235, 222, 44, -216, 63, -251, 40, -140, 112, -355, 60, 123, 26, -116,
    /// -89, -200, -109, 168, 135, -34, -99, -54, 5, -81, 322, 87, 4, -139, 420, 92, -295,
    /// -12, 262, -1, 26, -48, 231, 1, -335, 244, 188, -4, 5, -362, 57, -198, -184, -117,
    /// 40, 305, 49, 30, -59, -26, -37, 96]), &[82]).unwrap();     
    /// let b = Tensor::<i32>::new(Some(&[52, -245, 153, 13, -4, -56, -163, 249, -128, -172, 396,
    /// 143, 2, -96, 504, -44, -158, -393, 61, 95, 191, 74, 64, -219, 553, 104, 235, 222,
    /// 44, -216, 63, -251, 40, -140, 112, -355, 60, 123, 26, -116, -89, -200, -109, 168,
    /// 135, -34, -99, -54, 5, -81, 322, 87, 4, -139, 420, 92, -295, -12, 262, -1, 26,
    /// -48, 231, -335, 244, 188, 5, -362, 57, -198, -184, -117, 40, 305, 49, 30, -59, -26, -37, 96]), &[80]).unwrap();     
    /// let mut indices = vec![63, 67];
    /// assert_eq!(a.remove_indices(&mut indices, true).unwrap(), b);
    /// ```
    pub fn remove_indices(
        &self,
        indices: &mut [usize],
        is_sorted: bool,
    ) -> Result<Tensor<T>, TensorError> {
        let mut inner: Vec<T> = self.inner.clone();
        // time it
        if !is_sorted {
            indices.par_sort_unstable();
        }
        // remove indices
        for elem in indices.iter().rev() {
            inner.remove(*elem);
        }

        Tensor::new(Some(&inner), &[inner.len()])
    }

    /// Returns the tensor's dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    ///Reshape the tensor
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    /// a.reshape(&[9, 3]);
    /// assert_eq!(a.dims(), &[9, 3]);
    /// ```
    pub fn reshape(&mut self, new_dims: &[usize]) -> Result<(), TensorError> {
        // in onnx parlance this corresponds to converting a tensor to a single element
        if new_dims.is_empty() {
            if !(self.len() == 1 || self.is_empty()) {
                return Err(TensorError::DimError(
                    "Cannot reshape to empty tensor".to_string(),
                ));
            }
            self.dims = vec![];
        } else {
            let product = if new_dims != [0] {
                new_dims.iter().product::<usize>()
            } else {
                0
            };
            if self.len() != product {
                return Err(TensorError::DimError(format!(
                    "Cannot reshape tensor of length {} to {:?}",
                    self.len(),
                    new_dims
                )));
            }
            self.dims = Vec::from(new_dims);
        }
        Ok(())
    }

    /// Move axis of the tensor
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    /// let b = a.move_axis(0, 2).unwrap();
    /// assert_eq!(b.dims(), &[3, 3, 3]);
    ///
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 1, 2]).unwrap();     
    /// let mut expected = Tensor::<i32>::new(Some(&[1, 3, 5, 2, 4, 6]), &[1, 2, 3]).unwrap();    
    /// let b = a.move_axis(0, 2).unwrap();
    /// assert_eq!(b, expected);
    ///
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), &[2, 3, 2]).unwrap();     
    ///  let mut expected =
    /// Tensor::<i32>::new(Some(&[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]), &[2, 2, 3]).unwrap();    
    /// let b = a.move_axis(1, 2).unwrap();
    /// assert_eq!(b, expected);
    ///
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), &[2, 3, 2]).unwrap();     
    ///  let mut expected = Tensor::<i32>::new(Some(&[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]), &[2, 2, 3]).unwrap();     
    /// let b = a.move_axis(2, 1).unwrap();
    /// assert_eq!(b, expected);
    /// ```
    pub fn move_axis(&mut self, source: usize, destination: usize) -> Result<Self, TensorError> {
        assert!(source < self.dims.len());
        assert!(destination < self.dims.len());
        let mut new_dims = self.dims.clone();
        new_dims.remove(source);
        new_dims.insert(destination, self.dims[source]);

        // now reconfigure the elements appropriately in the new array
        //  eg. if we have a 3x3x3 array and we want to move the 0th axis to the 2nd position
        // //  we need to move the elements at 0, 1, 2, 3, 4, 5, 6, 7, 8 to 0, 3, 6, 1, 4, 7,
        // 2, 5, 8         //  so we need to move the elements at 0, 1, 2 to 0, 3, 6
        // //  and the elements at 3, 4, 5 to 1, 4, 7  and the elements at 6, 7, 8 to 2,
        // 5, 8
        let cartesian_coords = new_dims
            .iter()
            .map(|d| 0..*d)
            .multi_cartesian_product()
            .collect::<Vec<Vec<usize>>>();

        let mut output = Tensor::new(None, &new_dims)?;

        for coord in cartesian_coords {
            let mut old_coord = vec![0; self.dims.len()];

            // now fetch the old index
            for (i, c) in coord.iter().enumerate() {
                if i == destination {
                    old_coord[source] = *c;
                } else if i == source && source < destination {
                    old_coord[source + 1] = *c;
                } else if i == source && source > destination {
                    old_coord[source - 1] = *c;
                } else if (i < source && source < destination)
                    || (i < destination && source > destination)
                {
                    old_coord[i] = *c;
                } else if i > source && source < destination {
                    old_coord[i + 1] = *c;
                } else if i > destination && source > destination {
                    old_coord[i - 1] = *c;
                } else {
                    return Err(TensorError::DimError(
                        "Unknown condition for moving the axis".to_string(),
                    ));
                }
            }
            output.set(&coord, self.get(&old_coord));
        }

        Ok(output)
    }

    /// Swap axes of the tensor
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    /// let b = a.swap_axes(0, 2).unwrap();
    /// assert_eq!(b.dims(), &[3, 3, 3]);
    ///
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 1, 2]).unwrap();     
    /// let mut expected = Tensor::<i32>::new(Some(&[1, 3, 5, 2, 4, 6]), &[2, 1, 3]).unwrap();     
    /// let b = a.swap_axes(0, 2).unwrap();
    /// assert_eq!(b, expected);
    ///
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), &[2, 3, 2]).unwrap();     
    /// let mut expected =
    /// Tensor::<i32>::new(Some(&[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]), &[2, 2, 3]).unwrap();     
    /// let b = a.swap_axes(1, 2).unwrap();
    /// assert_eq!(b, expected);
    ///
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), &[2, 3, 2]).unwrap();     
    /// /// let mut expected =
    /// Tensor::<i32>::new(Some(&[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]), &[2, 2, 3]).unwrap();     
    /// let b = a.swap_axes(2, 1).unwrap();
    /// assert_eq!(b, expected);
    /// ```
    pub fn swap_axes(&mut self, source: usize, destination: usize) -> Result<Self, TensorError> {
        assert!(source < self.dims.len());
        assert!(destination < self.dims.len());
        let mut new_dims = self.dims.clone();
        new_dims[source] = self.dims[destination];
        new_dims[destination] = self.dims[source];

        // now reconfigure the elements appropriately in the new array
        //  eg. if we have a 3x3x3 array and we want to move the 0th axis to the
        // 2nd position
        //  we need to move the elements at 0, 1, 2, 3, 4, 5, 6, 7, 8
        // to 0, 3, 6, 1, 4, 7, 2, 5, 8
        //  so we need to move the elements at 0, 1,
        // 2 to 0, 3, 6
        //  and the elements at 3, 4, 5 to 1, 4, 7
        //  and the elements at 6, 7, 8 to 2, 5, 8
        let cartesian_coords = new_dims
            .iter()
            .map(|d| 0..*d)
            .multi_cartesian_product()
            .collect::<Vec<Vec<usize>>>();

        let mut output = Tensor::new(None, &new_dims)?;

        for coord in cartesian_coords {
            let mut old_coord = vec![0; self.dims.len()];

            // now fetch the old index
            for (i, c) in coord.iter().enumerate() {
                if i == destination {
                    old_coord[source] = *c;
                } else if i == source {
                    old_coord[destination] = *c;
                } else {
                    old_coord[i] = *c;
                }
            }
            output.set(&coord, self.get(&old_coord));
        }

        Ok(output)
    }

    /// Broadcasts the tensor to a given shape
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3, 1]).unwrap();
    ///
    /// let mut expected = Tensor::<i32>::new(Some(&[1, 1, 1, 2, 2, 2, 3, 3, 3]), &[3, 3]).unwrap();     /// assert_eq!(a.expand(&[3, 3]).unwrap(), expected);
    /// ```
    pub fn expand(&self, shape: &[usize]) -> Result<Self, TensorError> {
        if self.dims().len() > shape.len() {
            return Err(TensorError::DimError(format!(
                "Cannot expand {:?} to the smaller shape {:?}",
                self.dims(),
                shape
            )));
        }

        if shape == self.dims() {
            return Ok(self.clone());
        }

        for d in self.dims() {
            if !(shape.contains(d) || *d == 1) {
                return Err(TensorError::DimError(format!(
                    "The current dimension {d} must be contained in the new shape
  {shape:?} or be 1",
                )));
            }
        }

        let cartesian_coords = shape
            .iter()
            .map(|d| 0..*d)
            .multi_cartesian_product()
            .collect::<Vec<Vec<usize>>>();

        let mut output = Tensor::new(None, shape)?;

        for coord in cartesian_coords {
            let mut new_coord = Vec::with_capacity(self.dims().len());
            for (i, c) in coord.iter().enumerate() {
                if i < self.dims().len() && self.dims()[i] == 1 {
                    new_coord.push(0);
                } else if i >= self.dims().len() {
                    // do nothing at this point does not exist in the original tensor
                } else {
                    new_coord.push(*c);
                }
            }
            output.set(&coord, self.get(&new_coord));
        }

        Ok(output)
    }

    ///Flatten the tensor shape
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    /// a.flatten();
    /// assert_eq!(a.dims(), &[27]);
    /// ```
    pub fn flatten(&mut self) {
        if !self.dims().is_empty() && (self.dims() != [0]) {
            self.dims = Vec::from([self.dims.iter().product::<usize>()]);
        }
    }

    /// Maps a function to tensors
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.map(|x| i32::pow(x,2));
    /// assert_eq!(c, Tensor::from([1, 16].into_iter()))
    /// ```
    pub fn map<F: FnMut(T) -> G, G: TensorType>(&self, mut f: F) -> Tensor<G> {
        let mut t = Tensor::from(self.inner.iter().map(|e| f(e.clone())));
        // safe to unwrap as we know the dims are correct
        t.reshape(self.dims()).unwrap();
        t
    }

    /// Maps a function to tensors and enumerates
    /// ```
    /// use ezkl::tensor::{Tensor, TensorError};
    /// let mut a = Tensor::<i32>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.enum_map::<_,_,TensorError>(|i, x| Ok(i32::pow(x + i as i32, 2))).unwrap();     /// assert_eq!(c, Tensor::from([1, 25].into_iter()));
    /// ```
    pub fn enum_map<F: FnMut(usize, T) -> Result<G, E>, G: TensorType, E: Error>(
        &self,
        mut f: F,
    ) -> Result<Tensor<G>, E> {
        let vec: Result<Vec<G>, E> = self
            .inner
            .iter()
            .enumerate()
            .map(|(i, e)| f(i, e.clone()))
            .collect();
        let mut t: Tensor<G> = Tensor::from(vec?.iter().cloned());
        // safe to unwrap as we know the dims are correct
        t.reshape(self.dims()).unwrap();
        Ok(t)
    }

    /// Maps a function to tensors and enumerates in parallel
    /// ```
    /// use ezkl::tensor::{Tensor, TensorError};
    /// let mut a = Tensor::<i32>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.par_enum_map::<_,_,TensorError>(|i, x| Ok(i32::pow(x + i as i32, 2))).unwrap();     /// assert_eq!(c, Tensor::from([1, 25].into_iter()));
    /// ```
    pub fn par_enum_map<
        F: Fn(usize, T) -> Result<G, E> + std::marker::Send + std::marker::Sync,
        G: TensorType + std::marker::Send + std::marker::Sync,
        E: Error + std::marker::Send + std::marker::Sync,
    >(
        &self,
        f: F,
    ) -> Result<Tensor<G>, E>
    where
        T: std::marker::Send + std::marker::Sync,
    {
        let vec: Result<Vec<G>, E> = self
            .inner
            .par_iter()
            .enumerate()
            .map(move |(i, e)| f(i, e.clone()))
            .collect();
        let mut t: Tensor<G> = Tensor::from(vec?.iter().cloned());
        // safe to unwrap as we know the dims are correct
        t.reshape(self.dims()).unwrap();
        Ok(t)
    }

    /// Maps a function to tensors and enumerates in parallel
    /// ```
    /// use ezkl::tensor::{Tensor, TensorError};
    /// let mut a = Tensor::<i32>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.par_enum_map::<_,_,TensorError>(|i, x| Ok(i32::pow(x + i as i32, 2))).unwrap();     /// assert_eq!(c, Tensor::from([1, 25].into_iter()));
    /// ```
    pub fn par_enum_map_mut_filtered<
        F: Fn(usize) -> Result<T, E> + std::marker::Send + std::marker::Sync,
        E: Error + std::marker::Send + std::marker::Sync,
    >(
        &mut self,
        filter_indices: &std::collections::HashSet<&usize>,
        f: F,
    ) -> Result<(), E>
    where
        T: std::marker::Send + std::marker::Sync,
    {
        self.inner
            .par_iter_mut()
            .enumerate()
            .filter(|(i, _)| filter_indices.contains(i))
            .for_each(move |(i, e)| *e = f(i).unwrap());
        Ok(())
    }
}

impl<T: Clone + TensorType> Tensor<Tensor<T>> {
    /// Flattens a tensor of tensors
    /// ```
    /// use ezkl::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap();     
    /// let mut b = Tensor::<i32>::new(Some(&[1, 4]), &[2, 1]).unwrap();     
    /// let mut c = Tensor::new(Some(&[a,b]), &[2]).unwrap();
    /// let mut d = c.combine().unwrap();
    /// assert_eq!(d.dims(), &[8]);
    /// ```
    pub fn combine(&self) -> Result<Tensor<T>, TensorError> {
        let mut dims = 0;
        let mut inner = Vec::new();
        for t in self.inner.clone().into_iter() {
            dims += t.len();
            inner.extend(t.inner);
        }
        Tensor::new(Some(&inner), &[dims])
    }
}

impl<T: TensorType + Add<Output = T> + std::marker::Send + std::marker::Sync> Add for Tensor<T> {
    type Output = Result<Tensor<T>, TensorError>;
    /// Adds tensors.
    /// # Arguments
    ///
    /// * `self` - Tensor
    /// * `rhs` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use std::ops::Add;
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2, 3, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.add(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[4, 4, 4, 2, 2, 2]), &[2, 3]).unwrap();     /// assert_eq!(result, expected);
    ///
    /// // Now test 1D casting
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2]),
    ///     &[1]).unwrap();
    /// let result = x.add(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[4, 3, 4, 3, 3, 3]), &[2, 3]).unwrap();     /// assert_eq!(result, expected);
    ///
    ///
    /// // Now test 2D casting
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2, 3]),
    ///     &[2]).unwrap();
    /// let result = x.add(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[4, 3, 4, 4, 4, 4]), &[2, 3]).unwrap();     /// assert_eq!(result, expected);
    /// ```
    fn add(self, rhs: Self) -> Self::Output {
        let broadcasted_shape = get_broadcasted_shape(self.dims(), rhs.dims()).unwrap();
        let mut lhs = self.expand(&broadcasted_shape).unwrap();
        let rhs = rhs.expand(&broadcasted_shape).unwrap();

        lhs.par_iter_mut().zip(rhs).for_each(|(o, r)| {
            *o = o.clone() + r;
        });

        Ok(lhs)
    }
}

impl<T: TensorType + Neg<Output = T> + std::marker::Send + std::marker::Sync> Neg for Tensor<T> {
    type Output = Tensor<T>;
    /// Negates a tensor.
    /// # Arguments
    /// * `self` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use std::ops::Neg;
    /// let x = Tensor::<i32>::new(
    ///    Some(&[2, 1, 2, 1, 1, 1]),
    ///   &[2, 3],
    /// ).unwrap();
    /// let result = x.neg();
    /// let expected = Tensor::<i32>::new(Some(&[-2, -1, -2, -1, -1, -1]), &[2, 3]).unwrap();     /// assert_eq!(result, expected);
    /// ```
    fn neg(self) -> Self {
        let mut output = self;
        output.par_iter_mut().for_each(|x| {
            *x = x.clone().neg();
        });
        output
    }
}

impl<T: TensorType + Sub<Output = T> + std::marker::Send + std::marker::Sync> Sub for Tensor<T> {
    type Output = Result<Tensor<T>, TensorError>;
    /// Subtracts tensors.
    /// # Arguments
    ///
    /// * `self` - Tensor
    /// * `rhs` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use std::ops::Sub;
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2, 3, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.sub(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[0, -2, 0, 0, 0, 0]), &[2, 3]).unwrap();     /// assert_eq!(result, expected);
    ///
    /// // Now test 1D sub
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2]),
    ///     &[1],
    /// ).unwrap();
    /// let result = x.sub(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[0, -1, 0, -1, -1, -1]), &[2, 3]).unwrap();     /// assert_eq!(result, expected);
    ///
    /// // Now test 2D sub
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2, 3]),
    ///     &[2],
    /// ).unwrap();
    /// let result = x.sub(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[0, -1, 0, -2, -2, -2]), &[2, 3]).unwrap();     /// assert_eq!(result, expected);
    /// ```
    fn sub(self, rhs: Self) -> Self::Output {
        let broadcasted_shape = get_broadcasted_shape(self.dims(), rhs.dims()).unwrap();
        let mut lhs = self.expand(&broadcasted_shape).unwrap();
        let rhs = rhs.expand(&broadcasted_shape).unwrap();

        lhs.par_iter_mut().zip(rhs).for_each(|(o, r)| {
            *o = o.clone() - r;
        });

        Ok(lhs)
    }
}

impl<T: TensorType + Mul<Output = T> + std::marker::Send + std::marker::Sync> Mul for Tensor<T> {
    type Output = Result<Tensor<T>, TensorError>;
    /// Elementwise multiplies tensors.
    /// # Arguments
    ///
    /// * `self` - Tensor
    /// * `rhs` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use std::ops::Mul;
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2, 3, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.mul(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[4, 3, 4, 1, 1, 1]), &[2, 3]).unwrap();     /// assert_eq!(result, expected);
    ///
    /// // Now test 1D mult
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2]),
    ///     &[1]).unwrap();
    /// let result = x.mul(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2, 3]).unwrap();     /// assert_eq!(result, expected);
    ///
    /// // Now test 2D mult
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2, 2]),
    ///     &[2]).unwrap();
    /// let result = x.mul(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2,3]).unwrap();     /// assert_eq!(result, expected);
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        let broadcasted_shape = get_broadcasted_shape(self.dims(), rhs.dims()).unwrap();
        let mut lhs = self.expand(&broadcasted_shape).unwrap();
        let rhs = rhs.expand(&broadcasted_shape).unwrap();

        lhs.par_iter_mut().zip(rhs).for_each(|(o, r)| {
            *o = o.clone() * r;
        });

        Ok(lhs)
    }
}

impl<T: TensorType + Mul<Output = T> + std::marker::Send + std::marker::Sync> Tensor<T> {
    /// Elementwise raise a tensor to the nth power.
    /// # Arguments
    ///
    /// * `self` - Tensor
    /// * `b` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use std::ops::Mul;
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 15, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.pow(3).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[8, 3375, 8, 1, 1, 0]), &[2, 3]).unwrap();     /// assert_eq!(result, expected);
    /// ```
    pub fn pow(&self, mut exp: u32) -> Result<Self, TensorError> {
        // calculate value of output
        let mut base = self.clone();
        let mut acc = base.map(|_| T::one().unwrap());

        while exp > 1 {
            if (exp & 1) == 1 {
                acc = acc.mul(base.clone())?;
            }
            exp /= 2;
            base = base.clone().mul(base)?;
        }

        // since exp!=0, finally the exp must be 1.
        // Deal with the final bit of the exponent separately, since
        // squaring the base afterwards is not necessary and may cause a
        // needless overflow.
        acc.mul(base)
    }
}

impl<T: TensorType + Div<Output = T> + std::marker::Send + std::marker::Sync> Div for Tensor<T> {
    type Output = Result<Tensor<T>, TensorError>;
    /// Elementwise divide a tensor with another tensor.
    /// # Arguments
    ///
    /// * `self` - Tensor
    /// * `rhs` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use std::ops::Div;
    /// let x = Tensor::<i32>::new(
    ///     Some(&[4, 1, 4, 1, 1, 4]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let y = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.div(y).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[2, 1, 2, 1, 1, 4]), &[2, 3]).unwrap();     
    /// assert_eq!(result, expected);
    ///
    /// // test 1D casting
    /// let x = Tensor::<i32>::new(
    ///     Some(&[4, 1, 4, 1, 1, 4]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let y = Tensor::<i32>::new(
    ///     Some(&[2]),
    ///     &[1],
    /// ).unwrap();
    /// let result = x.div(y).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[2, 0, 2, 0, 0, 2]), &[2, 3]).unwrap();     /// assert_eq!(result, expected);
    /// ```
    fn div(self, rhs: Self) -> Self::Output {
        let broadcasted_shape = get_broadcasted_shape(self.dims(), rhs.dims()).unwrap();
        let mut lhs = self.expand(&broadcasted_shape).unwrap();
        let rhs = rhs.expand(&broadcasted_shape).unwrap();

        lhs.par_iter_mut().zip(rhs).for_each(|(o, r)| {
            *o = o.clone() / r;
        });

        Ok(lhs)
    }
}

/// Returns the broadcasted shape of two tensors
/// ```
/// use ezkl::tensor::get_broadcasted_shape;
/// let a = vec![2, 3];
/// let b = vec![2, 3];
/// let c = get_broadcasted_shape(&a, &b).unwrap();
/// assert_eq!(c, vec![2, 3]);
///
/// let a = vec![2, 3];
/// let b = vec![3];
/// let c = get_broadcasted_shape(&a, &b).unwrap();
/// assert_eq!(c, vec![2, 3]);
///
/// let a = vec![2, 3];
/// let b = vec![2, 1];
/// let c = get_broadcasted_shape(&a, &b).unwrap();
/// assert_eq!(c, vec![2, 3]);
///
/// let a = vec![2, 3];
/// let b = vec![1, 3];
/// let c = get_broadcasted_shape(&a, &b).unwrap();
/// assert_eq!(c, vec![2, 3]);
///
/// let a = vec![2, 3];
/// let b = vec![1, 1];
/// let c = get_broadcasted_shape(&a, &b).unwrap();
/// assert_eq!(c, vec![2, 3]);
/// ```
pub fn get_broadcasted_shape(
    shape_a: &[usize],
    shape_b: &[usize],
) -> Result<Vec<usize>, Box<dyn Error>> {
    let num_dims_a = shape_a.len();
    let num_dims_b = shape_b.len();

    // reewrite the below using match
    if num_dims_a == num_dims_b {
        let mut broadcasted_shape = Vec::with_capacity(num_dims_a);
        for (dim_a, dim_b) in shape_a.iter().zip(shape_b.iter()) {
            let max_dim = dim_a.max(dim_b);
            broadcasted_shape.push(*max_dim);
        }
        Ok(broadcasted_shape)
    } else if num_dims_a < num_dims_b {
        Ok(shape_b.to_vec())
    } else {
        Ok(shape_a.to_vec())
    }
}
////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor() {
        let data: Vec<f32> = vec![-1.0f32, 0.0, 1.0, 2.5];
        let tensor = Tensor::<f32>::new(Some(&data), &[2, 2]).unwrap();
        assert_eq!(&tensor[..], &data[..]);
    }

    #[test]
    fn tensor_clone() {
        let x = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3]).unwrap();
        assert_eq!(x, x.clone());
    }

    #[test]
    fn tensor_eq() {
        let a = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3]).unwrap();
        let mut b = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3, 1]).unwrap();
        b.reshape(&[3]).unwrap();
        let c = Tensor::<i32>::new(Some(&[1, 2, 4]), &[3]).unwrap();
        let d = Tensor::<i32>::new(Some(&[1, 2, 4]), &[3, 1]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }
    #[test]
    fn tensor_slice() {
        let a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap();
        let b = Tensor::<i32>::new(Some(&[1, 4]), &[2, 1]).unwrap();
        assert_eq!(a.get_slice(&[0..2, 0..1]).unwrap(), b);
    }
}
