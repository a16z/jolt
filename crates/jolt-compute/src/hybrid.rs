//! Hybrid compute backend that wraps a primary (GPU) and fallback (CPU) backend.
//!
//! Buffers start on the primary backend and automatically downgrade to the
//! fallback when they shrink below a configurable threshold during
//! `interpolate_pairs_inplace`. This captures the key observation that GPU
//! kernel launch overhead dominates for small buffers, while GPU parallelism
//! dominates for large ones.
//!
//! The transition is **one-way**: once a buffer migrates to the fallback, it
//! stays there for the rest of the sumcheck. All buffers in a stage bind in
//! lockstep (same scalar, same round), so they transition together.

use jolt_field::Field;
use jolt_ir::KernelDescriptor;

use crate::{BindingOrder, ComputeBackend, Scalar};

/// Hybrid backend that delegates to a primary or fallback backend based on
/// buffer size.
///
/// `P` is the primary backend (typically GPU — Metal, CUDA, WebGPU).
/// `F` is the fallback backend (typically CPU).
///
/// All uploads go to the primary. When `interpolate_pairs_inplace` shrinks a
/// buffer below `threshold`, its data is downloaded from the primary and
/// re-uploaded to the fallback. Subsequent operations on that buffer use
/// the fallback backend.
pub struct HybridBackend<P: ComputeBackend, Fb: ComputeBackend> {
    primary: P,
    fallback: Fb,
    /// Buffers with active element count at or below this value migrate to
    /// the fallback on the next `interpolate_pairs_inplace`.
    threshold: usize,
}

impl<P: ComputeBackend, Fb: ComputeBackend> HybridBackend<P, Fb> {
    pub fn new(primary: P, fallback: Fb, threshold: usize) -> Self {
        Self {
            primary,
            fallback,
            threshold,
        }
    }

    pub fn primary(&self) -> &P {
        &self.primary
    }

    pub fn fallback(&self) -> &Fb {
        &self.fallback
    }

    pub fn threshold(&self) -> usize {
        self.threshold
    }
}

/// Buffer that lives on either the primary or fallback backend.
///
/// Starts as `Primary` after `upload`. Transitions to `Fallback` when the
/// buffer shrinks below the hybrid backend's threshold during binding.
/// This transition is one-way — a fallback buffer never moves back to the
/// primary.
pub enum HybridBuffer<T: Scalar, P: ComputeBackend, Fb: ComputeBackend> {
    Primary(P::Buffer<T>),
    Fallback(Fb::Buffer<T>),
}

// SAFETY: `HybridBuffer` delegates to `P::Buffer` or `Fb::Buffer`, both of
// which are `Send + Sync` as required by `ComputeBackend`. The enum adds no
// interior mutability or thread-unsafe state.
unsafe impl<T: Scalar, P: ComputeBackend, Fb: ComputeBackend> Send for HybridBuffer<T, P, Fb>
where
    P::Buffer<T>: Send,
    Fb::Buffer<T>: Send,
{
}
// SAFETY: See `Send` impl above.
unsafe impl<T: Scalar, P: ComputeBackend, Fb: ComputeBackend> Sync for HybridBuffer<T, P, Fb>
where
    P::Buffer<T>: Sync,
    Fb::Buffer<T>: Sync,
{
}

/// Compiled kernel holding both backend's compiled forms.
///
/// Always compiles for both backends at kernel compile time so that we can
/// dispatch to either without recompilation when buffers transition.
pub struct HybridKernel<Fld: Field, P: ComputeBackend, Fb: ComputeBackend> {
    primary: P::CompiledKernel<Fld>,
    fallback: Fb::CompiledKernel<Fld>,
}

// SAFETY: `HybridKernel` wraps `P::CompiledKernel` and `Fb::CompiledKernel`,
// both `Send + Sync` as required by `ComputeBackend`. No additional state.
unsafe impl<Fld: Field, P: ComputeBackend, Fb: ComputeBackend> Send for HybridKernel<Fld, P, Fb>
where
    P::CompiledKernel<Fld>: Send,
    Fb::CompiledKernel<Fld>: Send,
{
}
// SAFETY: See `Send` impl above.
unsafe impl<Fld: Field, P: ComputeBackend, Fb: ComputeBackend> Sync for HybridKernel<Fld, P, Fb>
where
    P::CompiledKernel<Fld>: Sync,
    Fb::CompiledKernel<Fld>: Sync,
{
}

impl<P: ComputeBackend, Fb: ComputeBackend> HybridBackend<P, Fb> {
    fn should_migrate(&self, len: usize) -> bool {
        len <= self.threshold
    }
}

impl<P: ComputeBackend, Fb: ComputeBackend> ComputeBackend for HybridBackend<P, Fb> {
    type Buffer<T: Scalar> = HybridBuffer<T, P, Fb>;
    type CompiledKernel<Fld: Field> = HybridKernel<Fld, P, Fb>;

    fn compile_kernel_with_challenges<Fld: Field>(
        &self,
        desc: &KernelDescriptor,
        challenges: &[Fld],
    ) -> Self::CompiledKernel<Fld> {
        HybridKernel {
            primary: self
                .primary
                .compile_kernel_with_challenges(desc, challenges),
            fallback: self
                .fallback
                .compile_kernel_with_challenges(desc, challenges),
        }
    }

    fn upload<T: Scalar>(&self, data: &[T]) -> Self::Buffer<T> {
        // Small data goes directly to fallback — no point in a GPU round-trip.
        if data.len() <= self.threshold {
            HybridBuffer::Fallback(self.fallback.upload(data))
        } else {
            HybridBuffer::Primary(self.primary.upload(data))
        }
    }

    fn download<T: Scalar>(&self, buf: &Self::Buffer<T>) -> Vec<T> {
        match buf {
            HybridBuffer::Primary(b) => self.primary.download(b),
            HybridBuffer::Fallback(b) => self.fallback.download(b),
        }
    }

    fn alloc<T: Scalar>(&self, len: usize) -> Self::Buffer<T> {
        if len <= self.threshold {
            HybridBuffer::Fallback(self.fallback.alloc(len))
        } else {
            HybridBuffer::Primary(self.primary.alloc(len))
        }
    }

    fn len<T: Scalar>(&self, buf: &Self::Buffer<T>) -> usize {
        match buf {
            HybridBuffer::Primary(b) => self.primary.len(b),
            HybridBuffer::Fallback(b) => self.fallback.len(b),
        }
    }

    fn interpolate_pairs<T, Fld>(&self, buf: Self::Buffer<T>, scalar: Fld) -> Self::Buffer<Fld>
    where
        T: Scalar,
        Fld: Field + From<T>,
    {
        match buf {
            HybridBuffer::Primary(b) => {
                let result_len = self.primary.len(&b) / 2;
                if result_len <= self.threshold {
                    // Migrate: download, bind on fallback.
                    let data = self.primary.download(&b);
                    let fb_buf = self.fallback.upload(&data);
                    HybridBuffer::Fallback(self.fallback.interpolate_pairs(fb_buf, scalar))
                } else {
                    HybridBuffer::Primary(self.primary.interpolate_pairs(b, scalar))
                }
            }
            HybridBuffer::Fallback(b) => {
                HybridBuffer::Fallback(self.fallback.interpolate_pairs(b, scalar))
            }
        }
    }

    fn interpolate_pairs_inplace<Fld: Field>(
        &self,
        buf: &mut Self::Buffer<Fld>,
        scalar: Fld,
        order: BindingOrder,
    ) {
        // Determine if we need to migrate after this bind.
        let current_len = self.len(buf);
        let result_len = current_len / 2;

        match buf {
            HybridBuffer::Primary(b) => {
                if self.should_migrate(result_len) {
                    // Bind on primary first (it holds the data), then migrate.
                    self.primary.interpolate_pairs_inplace(b, scalar, order);
                    let data = self.primary.download(b);
                    *buf = HybridBuffer::Fallback(self.fallback.upload(&data));
                } else {
                    self.primary.interpolate_pairs_inplace(b, scalar, order);
                }
            }
            HybridBuffer::Fallback(b) => {
                self.fallback.interpolate_pairs_inplace(b, scalar, order);
            }
        }
    }

    fn interpolate_pairs_batch<Fld: Field>(
        &self,
        bufs: Vec<Self::Buffer<Fld>>,
        scalar: Fld,
    ) -> Vec<Self::Buffer<Fld>> {
        // Partition into primary and fallback, batch each, reassemble.
        // Track original indices to preserve order.
        let mut primary_bufs = Vec::new();
        let mut fallback_bufs = Vec::new();
        let mut indices: Vec<(bool, usize)> = Vec::with_capacity(bufs.len()); // (is_primary, idx_in_partition)

        for buf in bufs {
            match buf {
                HybridBuffer::Primary(b) => {
                    indices.push((true, primary_bufs.len()));
                    primary_bufs.push(b);
                }
                HybridBuffer::Fallback(b) => {
                    indices.push((false, fallback_bufs.len()));
                    fallback_bufs.push(b);
                }
            }
        }

        let mut primary_results = self.primary.interpolate_pairs_batch(primary_bufs, scalar);
        let mut fallback_results = self.fallback.interpolate_pairs_batch(fallback_bufs, scalar);

        // Reassemble in original order, migrating small primary results.
        let mut results = Vec::with_capacity(indices.len());
        for &(is_primary, idx) in &indices {
            if is_primary {
                // We need to take from primary_results — use a sentinel swap.
                let b = std::mem::replace(&mut primary_results[idx], self.primary.alloc(0));
                let len = self.primary.len(&b);
                if len <= self.threshold {
                    let data = self.primary.download(&b);
                    results.push(HybridBuffer::Fallback(self.fallback.upload(&data)));
                } else {
                    results.push(HybridBuffer::Primary(b));
                }
            } else {
                let b = std::mem::replace(&mut fallback_results[idx], self.fallback.alloc(0));
                results.push(HybridBuffer::Fallback(b));
            }
        }
        results
    }

    fn interpolate_pairs_batch_inplace<Fld: Field>(
        &self,
        bufs: &mut [Self::Buffer<Fld>],
        scalar: Fld,
        order: BindingOrder,
    ) {
        for buf in bufs.iter_mut() {
            self.interpolate_pairs_inplace(buf, scalar, order);
        }
    }

    fn pairwise_reduce<Fld: Field>(
        &self,
        inputs: &[&Self::Buffer<Fld>],
        weights: &Self::Buffer<Fld>,
        kernel: &Self::CompiledKernel<Fld>,
        num_evals: usize,
        order: BindingOrder,
    ) -> Vec<Fld> {
        // All buffers in a stage bind in lockstep, so they should all be
        // on the same backend. Check the first input to determine which.
        if let Some(first) = inputs.first() {
            match first {
                HybridBuffer::Primary(_) => {
                    let p_inputs: Vec<&P::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Primary(inner) => inner,
                            HybridBuffer::Fallback(_) => {
                                panic!("mixed buffer backends in pairwise_reduce")
                            }
                        })
                        .collect();
                    let p_weights = match weights {
                        HybridBuffer::Primary(w) => w,
                        HybridBuffer::Fallback(_) => {
                            panic!("weight buffer backend mismatch in pairwise_reduce")
                        }
                    };
                    self.primary.pairwise_reduce(
                        &p_inputs,
                        p_weights,
                        &kernel.primary,
                        num_evals,
                        order,
                    )
                }
                HybridBuffer::Fallback(_) => {
                    let f_inputs: Vec<&Fb::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Fallback(inner) => inner,
                            HybridBuffer::Primary(_) => {
                                panic!("mixed buffer backends in pairwise_reduce")
                            }
                        })
                        .collect();
                    let f_weights = match weights {
                        HybridBuffer::Fallback(w) => w,
                        HybridBuffer::Primary(_) => {
                            panic!("weight buffer backend mismatch in pairwise_reduce")
                        }
                    };
                    self.fallback.pairwise_reduce(
                        &f_inputs,
                        f_weights,
                        &kernel.fallback,
                        num_evals,
                        order,
                    )
                }
            }
        } else {
            vec![Fld::zero(); num_evals]
        }
    }

    fn pairwise_reduce_fixed<Fld: Field, const D: usize>(
        &self,
        inputs: &[&Self::Buffer<Fld>],
        weights: &Self::Buffer<Fld>,
        kernel: &Self::CompiledKernel<Fld>,
        order: BindingOrder,
    ) -> [Fld; D] {
        if let Some(first) = inputs.first() {
            match first {
                HybridBuffer::Primary(_) => {
                    let p_inputs: Vec<&P::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Primary(inner) => inner,
                            HybridBuffer::Fallback(_) => {
                                panic!("mixed buffer backends in pairwise_reduce_fixed")
                            }
                        })
                        .collect();
                    let p_weights = match weights {
                        HybridBuffer::Primary(w) => w,
                        HybridBuffer::Fallback(_) => {
                            panic!("weight buffer backend mismatch in pairwise_reduce_fixed")
                        }
                    };
                    self.primary.pairwise_reduce_fixed::<Fld, D>(
                        &p_inputs,
                        p_weights,
                        &kernel.primary,
                        order,
                    )
                }
                HybridBuffer::Fallback(_) => {
                    let f_inputs: Vec<&Fb::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Fallback(inner) => inner,
                            HybridBuffer::Primary(_) => {
                                panic!("mixed buffer backends in pairwise_reduce_fixed")
                            }
                        })
                        .collect();
                    let f_weights = match weights {
                        HybridBuffer::Fallback(w) => w,
                        HybridBuffer::Primary(_) => {
                            panic!("weight buffer backend mismatch in pairwise_reduce_fixed")
                        }
                    };
                    self.fallback.pairwise_reduce_fixed::<Fld, D>(
                        &f_inputs,
                        f_weights,
                        &kernel.fallback,
                        order,
                    )
                }
            }
        } else {
            [Fld::zero(); D]
        }
    }

    fn pairwise_reduce_unweighted<Fld: Field>(
        &self,
        inputs: &[&Self::Buffer<Fld>],
        kernel: &Self::CompiledKernel<Fld>,
        num_evals: usize,
        order: BindingOrder,
    ) -> Vec<Fld> {
        if let Some(first) = inputs.first() {
            match first {
                HybridBuffer::Primary(_) => {
                    let p_inputs: Vec<&P::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Primary(inner) => inner,
                            HybridBuffer::Fallback(_) => {
                                panic!("mixed buffer backends in pairwise_reduce_unweighted")
                            }
                        })
                        .collect();
                    self.primary.pairwise_reduce_unweighted(
                        &p_inputs,
                        &kernel.primary,
                        num_evals,
                        order,
                    )
                }
                HybridBuffer::Fallback(_) => {
                    let f_inputs: Vec<&Fb::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Fallback(inner) => inner,
                            HybridBuffer::Primary(_) => {
                                panic!("mixed buffer backends in pairwise_reduce_unweighted")
                            }
                        })
                        .collect();
                    self.fallback.pairwise_reduce_unweighted(
                        &f_inputs,
                        &kernel.fallback,
                        num_evals,
                        order,
                    )
                }
            }
        } else {
            vec![Fld::zero(); num_evals]
        }
    }

    fn tensor_pairwise_reduce<Fld: Field>(
        &self,
        inputs: &[&Self::Buffer<Fld>],
        outer_weights: &Self::Buffer<Fld>,
        inner_weights: &Self::Buffer<Fld>,
        kernel: &Self::CompiledKernel<Fld>,
        num_evals: usize,
    ) -> Vec<Fld> {
        if let Some(first) = inputs.first() {
            match first {
                HybridBuffer::Primary(_) => {
                    let p_inputs: Vec<&P::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Primary(inner) => inner,
                            HybridBuffer::Fallback(_) => {
                                panic!("mixed buffer backends in tensor_pairwise_reduce")
                            }
                        })
                        .collect();
                    let p_outer = match outer_weights {
                        HybridBuffer::Primary(w) => w,
                        HybridBuffer::Fallback(_) => {
                            panic!("outer weight buffer backend mismatch")
                        }
                    };
                    let p_inner = match inner_weights {
                        HybridBuffer::Primary(w) => w,
                        HybridBuffer::Fallback(_) => {
                            panic!("inner weight buffer backend mismatch")
                        }
                    };
                    self.primary.tensor_pairwise_reduce(
                        &p_inputs,
                        p_outer,
                        p_inner,
                        &kernel.primary,
                        num_evals,
                    )
                }
                HybridBuffer::Fallback(_) => {
                    let f_inputs: Vec<&Fb::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Fallback(inner) => inner,
                            HybridBuffer::Primary(_) => {
                                panic!("mixed buffer backends in tensor_pairwise_reduce")
                            }
                        })
                        .collect();
                    let f_outer = match outer_weights {
                        HybridBuffer::Fallback(w) => w,
                        HybridBuffer::Primary(_) => {
                            panic!("outer weight buffer backend mismatch")
                        }
                    };
                    let f_inner = match inner_weights {
                        HybridBuffer::Fallback(w) => w,
                        HybridBuffer::Primary(_) => {
                            panic!("inner weight buffer backend mismatch")
                        }
                    };
                    self.fallback.tensor_pairwise_reduce(
                        &f_inputs,
                        f_outer,
                        f_inner,
                        &kernel.fallback,
                        num_evals,
                    )
                }
            }
        } else {
            vec![Fld::zero(); num_evals]
        }
    }

    fn pairwise_reduce_multi<Fld: Field>(
        &self,
        inputs: &[&Self::Buffer<Fld>],
        weights: &Self::Buffer<Fld>,
        kernels: &[(&Self::CompiledKernel<Fld>, usize)],
        order: BindingOrder,
    ) -> Vec<Vec<Fld>> {
        if let Some(first) = inputs.first() {
            match first {
                HybridBuffer::Primary(_) => {
                    let p_inputs: Vec<&P::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Primary(inner) => inner,
                            HybridBuffer::Fallback(_) => {
                                panic!("mixed buffer backends in pairwise_reduce_multi")
                            }
                        })
                        .collect();
                    let p_weights = match weights {
                        HybridBuffer::Primary(w) => w,
                        HybridBuffer::Fallback(_) => {
                            panic!("weight buffer backend mismatch in pairwise_reduce_multi")
                        }
                    };
                    let p_kernels: Vec<(&P::CompiledKernel<Fld>, usize)> =
                        kernels.iter().map(|(k, n)| (&k.primary, *n)).collect();
                    self.primary
                        .pairwise_reduce_multi(&p_inputs, p_weights, &p_kernels, order)
                }
                HybridBuffer::Fallback(_) => {
                    let f_inputs: Vec<&Fb::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Fallback(inner) => inner,
                            HybridBuffer::Primary(_) => {
                                panic!("mixed buffer backends in pairwise_reduce_multi")
                            }
                        })
                        .collect();
                    let f_weights = match weights {
                        HybridBuffer::Fallback(w) => w,
                        HybridBuffer::Primary(_) => {
                            panic!("weight buffer backend mismatch in pairwise_reduce_multi")
                        }
                    };
                    let f_kernels: Vec<(&Fb::CompiledKernel<Fld>, usize)> =
                        kernels.iter().map(|(k, n)| (&k.fallback, *n)).collect();
                    self.fallback
                        .pairwise_reduce_multi(&f_inputs, f_weights, &f_kernels, order)
                }
            }
        } else {
            kernels.iter().map(|(_, n)| vec![Fld::zero(); *n]).collect()
        }
    }

    fn product_table<Fld: Field>(&self, point: &[Fld]) -> Self::Buffer<Fld> {
        let len = 1usize << point.len();
        if len <= self.threshold {
            HybridBuffer::Fallback(self.fallback.product_table(point))
        } else {
            HybridBuffer::Primary(self.primary.product_table(point))
        }
    }

    fn sum<Fld: Field>(&self, buf: &Self::Buffer<Fld>) -> Fld {
        match buf {
            HybridBuffer::Primary(b) => self.primary.sum(b),
            HybridBuffer::Fallback(b) => self.fallback.sum(b),
        }
    }

    fn dot_product<Fld: Field>(&self, a: &Self::Buffer<Fld>, b: &Self::Buffer<Fld>) -> Fld {
        match (a, b) {
            (HybridBuffer::Primary(a), HybridBuffer::Primary(b)) => self.primary.dot_product(a, b),
            (HybridBuffer::Fallback(a), HybridBuffer::Fallback(b)) => {
                self.fallback.dot_product(a, b)
            }
            _ => panic!("mixed buffer backends in dot_product"),
        }
    }

    fn scale<Fld: Field>(&self, buf: &mut Self::Buffer<Fld>, scalar: Fld) {
        match buf {
            HybridBuffer::Primary(b) => self.primary.scale(b, scalar),
            HybridBuffer::Fallback(b) => self.fallback.scale(b, scalar),
        }
    }

    fn add<Fld: Field>(&self, a: &Self::Buffer<Fld>, b: &Self::Buffer<Fld>) -> Self::Buffer<Fld> {
        match (a, b) {
            (HybridBuffer::Primary(a), HybridBuffer::Primary(b)) => {
                HybridBuffer::Primary(self.primary.add(a, b))
            }
            (HybridBuffer::Fallback(a), HybridBuffer::Fallback(b)) => {
                HybridBuffer::Fallback(self.fallback.add(a, b))
            }
            _ => panic!("mixed buffer backends in add"),
        }
    }

    fn sub<Fld: Field>(&self, a: &Self::Buffer<Fld>, b: &Self::Buffer<Fld>) -> Self::Buffer<Fld> {
        match (a, b) {
            (HybridBuffer::Primary(a), HybridBuffer::Primary(b)) => {
                HybridBuffer::Primary(self.primary.sub(a, b))
            }
            (HybridBuffer::Fallback(a), HybridBuffer::Fallback(b)) => {
                HybridBuffer::Fallback(self.fallback.sub(a, b))
            }
            _ => panic!("mixed buffer backends in sub"),
        }
    }

    fn accumulate<Fld: Field>(
        &self,
        buf: &mut Self::Buffer<Fld>,
        scalar: Fld,
        other: &Self::Buffer<Fld>,
    ) {
        match (buf, other) {
            (HybridBuffer::Primary(b), HybridBuffer::Primary(o)) => {
                self.primary.accumulate(b, scalar, o);
            }
            (HybridBuffer::Fallback(b), HybridBuffer::Fallback(o)) => {
                self.fallback.accumulate(b, scalar, o);
            }
            _ => panic!("mixed buffer backends in accumulate"),
        }
    }

    fn scale_batch<Fld: Field>(&self, bufs: &mut [Self::Buffer<Fld>], scalar: Fld) {
        for buf in bufs.iter_mut() {
            self.scale(buf, scalar);
        }
    }

    fn fused_interpolate_reduce<Fld: Field>(
        &self,
        inputs: &mut [Self::Buffer<Fld>],
        weights: &mut Self::Buffer<Fld>,
        interpolation_scalar: Fld,
        kernel: &Self::CompiledKernel<Fld>,
        num_evals: usize,
    ) -> Vec<Fld> {
        if inputs.is_empty() {
            return vec![Fld::zero(); num_evals];
        }

        let is_primary = matches!(inputs[0], HybridBuffer::Primary(_));

        if is_primary {
            // Temporarily move inner buffers out to form a contiguous slice.
            // alloc(0) creates a zero-sized placeholder — no real allocation.
            let mut p_bufs: Vec<P::Buffer<Fld>> = inputs
                .iter_mut()
                .map(|b| match b {
                    HybridBuffer::Primary(inner) => {
                        std::mem::replace(inner, self.primary.alloc(0))
                    }
                    HybridBuffer::Fallback(_) => {
                        panic!("mixed buffer backends in fused_interpolate_reduce")
                    }
                })
                .collect();
            let mut p_weights = match weights {
                HybridBuffer::Primary(w) => std::mem::replace(w, self.primary.alloc(0)),
                HybridBuffer::Fallback(_) => {
                    panic!("weight buffer backend mismatch in fused_interpolate_reduce")
                }
            };

            let result = self.primary.fused_interpolate_reduce(
                &mut p_bufs,
                &mut p_weights,
                interpolation_scalar,
                &kernel.primary,
                num_evals,
            );

            // Check migration before putting buffers back
            let new_len = self.primary.len(&p_bufs[0]);
            if self.should_migrate(new_len) {
                for (slot, p_buf) in inputs.iter_mut().zip(p_bufs) {
                    let data = self.primary.download(&p_buf);
                    *slot = HybridBuffer::Fallback(self.fallback.upload(&data));
                }
                let w_data = self.primary.download(&p_weights);
                *weights = HybridBuffer::Fallback(self.fallback.upload(&w_data));
            } else {
                for (slot, p_buf) in inputs.iter_mut().zip(p_bufs) {
                    *slot = HybridBuffer::Primary(p_buf);
                }
                *weights = HybridBuffer::Primary(p_weights);
            }

            result
        } else {
            let mut f_bufs: Vec<Fb::Buffer<Fld>> = inputs
                .iter_mut()
                .map(|b| match b {
                    HybridBuffer::Fallback(inner) => {
                        std::mem::replace(inner, self.fallback.alloc(0))
                    }
                    HybridBuffer::Primary(_) => {
                        panic!("mixed buffer backends in fused_interpolate_reduce")
                    }
                })
                .collect();
            let mut f_weights = match weights {
                HybridBuffer::Fallback(w) => std::mem::replace(w, self.fallback.alloc(0)),
                HybridBuffer::Primary(_) => {
                    panic!("weight buffer backend mismatch in fused_interpolate_reduce")
                }
            };

            let result = self.fallback.fused_interpolate_reduce(
                &mut f_bufs,
                &mut f_weights,
                interpolation_scalar,
                &kernel.fallback,
                num_evals,
            );

            for (slot, f_buf) in inputs.iter_mut().zip(f_bufs) {
                *slot = HybridBuffer::Fallback(f_buf);
            }
            *weights = HybridBuffer::Fallback(f_weights);

            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use jolt_field::{Field, Fr};
    use num_traits::Zero;

    /// Minimal "mock" backend for testing without pulling in the full CpuBackend.
    /// Uses Vec<T> buffers and trivial implementations.
    ///
    /// We use this to avoid a dev-dependency on jolt-cpu in jolt-compute's own
    /// tests while still exercising the HybridBackend logic.
    struct MockBackend {
        name: &'static str,
    }

    impl MockBackend {
        fn new(name: &'static str) -> Self {
            Self { name }
        }
    }

    struct MockKernel<F: Field> {
        _marker: std::marker::PhantomData<F>,
    }

    impl ComputeBackend for MockBackend {
        type Buffer<T: Scalar> = Vec<T>;
        type CompiledKernel<F: Field> = MockKernel<F>;

        fn compile_kernel_with_challenges<F: Field>(
            &self,
            _desc: &KernelDescriptor,
            _challenges: &[F],
        ) -> MockKernel<F> {
            MockKernel {
                _marker: std::marker::PhantomData,
            }
        }

        fn upload<T: Scalar>(&self, data: &[T]) -> Vec<T> {
            data.to_vec()
        }

        fn download<T: Scalar>(&self, buf: &Vec<T>) -> Vec<T> {
            buf.clone()
        }

        fn alloc<T: Scalar>(&self, len: usize) -> Vec<T> {
            // SAFETY: All `Scalar` types (integers, bool, field elements) have
            // all-zeros as a valid bit pattern.
            vec![unsafe { std::mem::zeroed() }; len]
        }

        fn len<T: Scalar>(&self, buf: &Vec<T>) -> usize {
            buf.len()
        }

        fn interpolate_pairs<T, Fld>(&self, buf: Vec<T>, scalar: Fld) -> Vec<Fld>
        where
            T: Scalar,
            Fld: Field + From<T>,
        {
            let n = buf.len() / 2;
            let mut result = Vec::with_capacity(n);
            for i in 0..n {
                let lo = Fld::from(buf[2 * i]);
                let hi = Fld::from(buf[2 * i + 1]);
                result.push(lo + scalar * (hi - lo));
            }
            result
        }

        fn interpolate_pairs_inplace<Fld: Field>(
            &self,
            buf: &mut Vec<Fld>,
            scalar: Fld,
            order: BindingOrder,
        ) {
            let n = buf.len();
            let half = n / 2;
            match order {
                BindingOrder::LowToHigh => {
                    for i in 0..half {
                        let lo = buf[2 * i];
                        let hi = buf[2 * i + 1];
                        buf[i] = lo + scalar * (hi - lo);
                    }
                }
                BindingOrder::HighToLow => {
                    for i in 0..half {
                        let lo = buf[i];
                        let hi = buf[i + half];
                        buf[i] = lo + scalar * (hi - lo);
                    }
                }
            }
            buf.truncate(half);
        }

        fn pairwise_reduce<Fld: Field>(
            &self,
            _inputs: &[&Vec<Fld>],
            _weights: &Vec<Fld>,
            _kernel: &MockKernel<Fld>,
            num_evals: usize,
            _order: BindingOrder,
        ) -> Vec<Fld> {
            // Return identifiable values based on backend name for testing dispatch.
            if self.name == "primary" {
                vec![Fld::from_u64(1); num_evals]
            } else {
                vec![Fld::from_u64(2); num_evals]
            }
        }

        fn tensor_pairwise_reduce<Fld: Field>(
            &self,
            _inputs: &[&Vec<Fld>],
            _outer_weights: &Vec<Fld>,
            _inner_weights: &Vec<Fld>,
            _kernel: &MockKernel<Fld>,
            num_evals: usize,
        ) -> Vec<Fld> {
            vec![Fld::zero(); num_evals]
        }

        fn product_table<Fld: Field>(&self, point: &[Fld]) -> Vec<Fld> {
            let len = 1usize << point.len();
            vec![Fld::one(); len]
        }

        fn sum<Fld: Field>(&self, buf: &Vec<Fld>) -> Fld {
            buf.iter().fold(Fld::zero(), |acc, x| acc + *x)
        }

        fn dot_product<Fld: Field>(&self, a: &Vec<Fld>, b: &Vec<Fld>) -> Fld {
            a.iter()
                .zip(b.iter())
                .fold(Fld::zero(), |acc, (x, y)| acc + *x * *y)
        }

        fn scale<Fld: Field>(&self, buf: &mut Vec<Fld>, scalar: Fld) {
            for x in buf.iter_mut() {
                *x *= scalar;
            }
        }

        fn add<Fld: Field>(&self, a: &Vec<Fld>, b: &Vec<Fld>) -> Vec<Fld> {
            a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
        }

        fn sub<Fld: Field>(&self, a: &Vec<Fld>, b: &Vec<Fld>) -> Vec<Fld> {
            a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect()
        }

        fn accumulate<Fld: Field>(&self, buf: &mut Vec<Fld>, scalar: Fld, other: &Vec<Fld>) {
            for (x, y) in buf.iter_mut().zip(other.iter()) {
                *x += scalar * *y;
            }
        }
    }

    fn make_hybrid(threshold: usize) -> HybridBackend<MockBackend, MockBackend> {
        HybridBackend::new(
            MockBackend::new("primary"),
            MockBackend::new("fallback"),
            threshold,
        )
    }

    fn is_primary<T: Scalar>(buf: &HybridBuffer<T, MockBackend, MockBackend>) -> bool {
        matches!(buf, HybridBuffer::Primary(_))
    }

    #[test]
    fn upload_large_goes_to_primary() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..16).map(Fr::from_u64).collect();
        let buf = hybrid.upload(&data);
        assert!(is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 16);
    }

    #[test]
    fn upload_small_goes_to_fallback() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..4).map(Fr::from_u64).collect();
        let buf = hybrid.upload(&data);
        assert!(!is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 4);
    }

    #[test]
    fn interpolate_pairs_inplace_migrates_at_threshold() {
        let hybrid = make_hybrid(4);
        // Start with 8 elements on primary.
        let data: Vec<Fr> = (0..8).map(Fr::from_u64).collect();
        let mut buf = hybrid.upload(&data);
        assert!(is_primary(&buf));

        // After bind: 4 elements — at threshold, should migrate.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(2), BindingOrder::LowToHigh);
        assert!(!is_primary(&buf), "should have migrated to fallback");
        assert_eq!(hybrid.len(&buf), 4);
    }

    #[test]
    fn interpolate_pairs_inplace_stays_primary_above_threshold() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..16).map(Fr::from_u64).collect();
        let mut buf = hybrid.upload(&data);
        assert!(is_primary(&buf));

        // After bind: 8 elements — above threshold, stays primary.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(2), BindingOrder::LowToHigh);
        assert!(is_primary(&buf), "should stay on primary");
        assert_eq!(hybrid.len(&buf), 8);
    }

    #[test]
    fn interpolate_pairs_fallback_stays_fallback() {
        let hybrid = make_hybrid(4);
        // Upload small — goes to fallback.
        let data: Vec<Fr> = (0..4).map(Fr::from_u64).collect();
        let mut buf = hybrid.upload(&data);
        assert!(!is_primary(&buf));

        // Bind on fallback — stays fallback.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(2), BindingOrder::LowToHigh);
        assert!(!is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 2);
    }

    #[test]
    fn pairwise_reduce_dispatches_to_correct_backend() {
        use jolt_ir::{KernelDescriptor, KernelShape};

        let hybrid = make_hybrid(4);
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 4,
                num_products: 1,
            },
            degree: 4,
            tensor_split: None,
        };
        let kernel = hybrid.compile_kernel::<Fr>(&desc);

        // Primary inputs → dispatches to primary → returns all 1s.
        let large: Vec<Fr> = vec![Fr::from_u64(1); 16];
        let weights_large: Vec<Fr> = vec![Fr::from_u64(1); 8];
        let buf_a = hybrid.upload(&large);
        let buf_w = hybrid.upload(&weights_large);
        assert!(is_primary(&buf_a));

        let result = hybrid.pairwise_reduce(&[&buf_a], &buf_w, &kernel, 4, BindingOrder::LowToHigh);
        assert_eq!(result, vec![Fr::from_u64(1); 4]);

        // Fallback inputs → dispatches to fallback → returns all 2s.
        let small: Vec<Fr> = vec![Fr::from_u64(1); 4];
        let weights_small: Vec<Fr> = vec![Fr::from_u64(1); 2];
        let buf_b = hybrid.upload(&small);
        let buf_ws = hybrid.upload(&weights_small);
        assert!(!is_primary(&buf_b));

        let result =
            hybrid.pairwise_reduce(&[&buf_b], &buf_ws, &kernel, 4, BindingOrder::LowToHigh);
        assert_eq!(result, vec![Fr::from_u64(2); 4]);
    }

    #[test]
    fn interpolate_pairs_consuming_migrates() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..8).map(Fr::from_u64).collect();
        let buf = hybrid.upload(&data);
        assert!(is_primary(&buf));

        // Result has 4 elements — at threshold, should migrate.
        let result = hybrid.interpolate_pairs(buf, Fr::from_u64(2));
        assert!(!is_primary(&result));
        assert_eq!(hybrid.len(&result), 4);
    }

    #[test]
    fn product_table_respects_threshold() {
        let hybrid = make_hybrid(8);

        // 2^2 = 4 elements — below threshold → fallback.
        let small_table = hybrid.product_table::<Fr>(&[Fr::from_u64(1), Fr::from_u64(2)]);
        assert!(!is_primary(&small_table));

        // 2^4 = 16 elements — above threshold → primary.
        let large_table = hybrid.product_table::<Fr>(&[
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ]);
        assert!(is_primary(&large_table));
    }

    #[test]
    fn download_works_from_both_backends() {
        let hybrid = make_hybrid(4);
        let large: Vec<Fr> = (0..8).map(Fr::from_u64).collect();
        let small: Vec<Fr> = (0..4).map(Fr::from_u64).collect();

        let buf_p = hybrid.upload(&large);
        let buf_f = hybrid.upload(&small);

        assert_eq!(hybrid.download(&buf_p), large);
        assert_eq!(hybrid.download(&buf_f), small);
    }

    #[test]
    fn sum_dispatches_correctly() {
        let hybrid = make_hybrid(4);
        let large: Vec<Fr> = vec![Fr::from_u64(3); 8];
        let small: Vec<Fr> = vec![Fr::from_u64(5); 2];

        let buf_p = hybrid.upload(&large);
        let buf_f = hybrid.upload(&small);

        assert_eq!(hybrid.sum(&buf_p), Fr::from_u64(24));
        assert_eq!(hybrid.sum(&buf_f), Fr::from_u64(10));
    }

    #[test]
    fn full_lifecycle_primary_to_fallback() {
        let hybrid = make_hybrid(4);

        // Start large on primary.
        let data: Vec<Fr> = (0..32).map(Fr::from_u64).collect();
        let mut buf = hybrid.upload(&data);
        assert!(is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 32);

        // Round 1: 32 → 16, stays primary.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(1), BindingOrder::LowToHigh);
        assert!(is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 16);

        // Round 2: 16 → 8, stays primary.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(1), BindingOrder::LowToHigh);
        assert!(is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 8);

        // Round 3: 8 → 4, migrates to fallback.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(1), BindingOrder::LowToHigh);
        assert!(!is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 4);

        // Round 4: 4 → 2, stays on fallback.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(1), BindingOrder::LowToHigh);
        assert!(!is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 2);
    }

    #[test]
    fn bind_values_correct_after_migration() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = vec![
            Fr::from_u64(10),
            Fr::from_u64(20),
            Fr::from_u64(30),
            Fr::from_u64(40),
            Fr::from_u64(50),
            Fr::from_u64(60),
            Fr::from_u64(70),
            Fr::from_u64(80),
        ];
        let mut buf = hybrid.upload(&data);

        // Bind with scalar=0 → takes all "lo" values: [10, 30, 50, 70].
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::zero(), BindingOrder::LowToHigh);
        let result = hybrid.download(&buf);
        assert_eq!(
            result,
            vec![
                Fr::from_u64(10),
                Fr::from_u64(30),
                Fr::from_u64(50),
                Fr::from_u64(70),
            ]
        );
        // Should have migrated (4 ≤ threshold=4).
        assert!(!is_primary(&buf));
    }

    #[test]
    #[should_panic(expected = "mixed buffer backends")]
    fn mixed_backends_panics() {
        use jolt_ir::{KernelDescriptor, KernelShape};

        let hybrid = make_hybrid(4);
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 2,
                num_products: 1,
            },
            degree: 2,
            tensor_split: None,
        };
        let kernel = hybrid.compile_kernel::<Fr>(&desc);

        let large: Vec<Fr> = vec![Fr::from_u64(1); 16];
        let small: Vec<Fr> = vec![Fr::from_u64(1); 4];
        let buf_p = hybrid.upload(&large);
        let buf_f = hybrid.upload(&small);

        // Weights on primary, but second input on fallback — should panic.
        let weights: Vec<Fr> = vec![Fr::from_u64(1); 8];
        let w_buf = hybrid.upload(&weights);

        let _ = hybrid.pairwise_reduce(
            &[&buf_p, &buf_f],
            &w_buf,
            &kernel,
            2,
            BindingOrder::LowToHigh,
        );
    }
}
