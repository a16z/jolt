//! Hybrid compute backend that wraps a primary (GPU) and fallback (CPU) backend.
//!
//! Buffers start on the primary backend and automatically downgrade to the
//! fallback when they shrink below a configurable threshold during
//! interpolation. This captures the key observation that GPU kernel launch
//! overhead dominates for small buffers, while GPU parallelism dominates for
//! large ones.
//!
//! The transition is **one-way**: once a buffer migrates to the fallback, it
//! stays there for the rest of the sumcheck.

use jolt_compiler::KernelSpec;
use jolt_field::Field;

use jolt_compute::{BindingOrder, Buf, ComputeBackend, DeviceBuffer, Scalar};

/// Hybrid backend that delegates to a primary or fallback backend based on
/// buffer size.
///
/// `P` is the primary backend (typically GPU -- Metal, CUDA, WebGPU).
/// `Fb` is the fallback backend (typically CPU).
///
/// All uploads go to the primary unless the data is already below threshold.
/// When interpolation shrinks a buffer below `threshold`, its data is
/// downloaded from the primary and re-uploaded to the fallback.
pub struct HybridBackend<P: ComputeBackend, Fb: ComputeBackend> {
    primary: P,
    fallback: Fb,
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

    fn should_migrate(&self, len: usize) -> bool {
        len <= self.threshold
    }
}

/// Buffer that lives on either the primary or fallback backend.
///
/// Starts as `Primary` after `upload`. Transitions to `Fallback` when the
/// buffer shrinks below the hybrid backend's threshold during binding.
/// This transition is one-way.
pub enum HybridBuffer<T: Scalar, P: ComputeBackend, Fb: ComputeBackend> {
    Primary(P::Buffer<T>),
    Fallback(Fb::Buffer<T>),
}

impl<T: Scalar, P: ComputeBackend, Fb: ComputeBackend> HybridBuffer<T, P, Fb> {
    pub fn is_primary(&self) -> bool {
        matches!(self, Self::Primary(_))
    }
}

// SAFETY: Rust's auto-trait inference doesn't propagate Send/Sync through GAT
// projections (P::Buffer<T>) in enum variants. ComputeBackend requires
// Buffer<T: Scalar>: Send + Sync, so these bounds are always satisfied --
// the manual impls bridge a compiler limitation with GATs.
unsafe impl<T: Scalar, P: ComputeBackend, Fb: ComputeBackend> Send for HybridBuffer<T, P, Fb>
where
    P::Buffer<T>: Send,
    Fb::Buffer<T>: Send,
{
}
// SAFETY: see above.
unsafe impl<T: Scalar, P: ComputeBackend, Fb: ComputeBackend> Sync for HybridBuffer<T, P, Fb>
where
    P::Buffer<T>: Sync,
    Fb::Buffer<T>: Sync,
{
}

/// Compiled kernel holding both backend's compiled forms.
///
/// Always compiles for both backends upfront so dispatch can switch without
/// recompilation when buffers transition mid-sumcheck.
pub struct HybridKernel<Fld: Field, P: ComputeBackend, Fb: ComputeBackend> {
    primary: P::CompiledKernel<Fld>,
    fallback: Fb::CompiledKernel<Fld>,
}

// SAFETY: same GAT auto-trait limitation as HybridBuffer -- see above.
unsafe impl<Fld: Field, P: ComputeBackend, Fb: ComputeBackend> Send for HybridKernel<Fld, P, Fb>
where
    P::CompiledKernel<Fld>: Send,
    Fb::CompiledKernel<Fld>: Send,
{
}
// SAFETY: see above.
unsafe impl<Fld: Field, P: ComputeBackend, Fb: ComputeBackend> Sync for HybridKernel<Fld, P, Fb>
where
    P::CompiledKernel<Fld>: Sync,
    Fb::CompiledKernel<Fld>: Sync,
{
}

/// Clone a `Buf<Self, F>` for the primary sub-backend by downloading and
/// re-uploading the inner buffer data. This creates an owned `Buf<P, F>`
/// suitable for passing to `P::reduce`.
///
/// For CPU sub-backends this is two Vec clones. For GPU sub-backends this
/// is a device-host-device roundtrip -- acceptable for the hybrid's
/// read-only reduce path, but a future optimization could add
/// `clone_on_device` to `ComputeBackend`.
fn clone_primary_buf<F: Field, P: ComputeBackend, Fb: ComputeBackend>(
    primary: &P,
    db: &Buf<HybridBackend<P, Fb>, F>,
) -> Buf<P, F> {
    match db {
        DeviceBuffer::Field(HybridBuffer::Primary(b)) => {
            DeviceBuffer::Field(primary.upload(&primary.download(b)))
        }
        DeviceBuffer::U64(HybridBuffer::Primary(b)) => {
            DeviceBuffer::U64(primary.upload(&primary.download(b)))
        }
        _ => panic!("expected primary buffer"),
    }
}

/// Same as [`clone_primary_buf`] but for the fallback sub-backend.
fn clone_fallback_buf<F: Field, P: ComputeBackend, Fb: ComputeBackend>(
    fallback: &Fb,
    db: &Buf<HybridBackend<P, Fb>, F>,
) -> Buf<Fb, F> {
    match db {
        DeviceBuffer::Field(HybridBuffer::Fallback(b)) => {
            DeviceBuffer::Field(fallback.upload(&fallback.download(b)))
        }
        DeviceBuffer::U64(HybridBuffer::Fallback(b)) => {
            DeviceBuffer::U64(fallback.upload(&fallback.download(b)))
        }
        _ => panic!("expected fallback buffer"),
    }
}

/// Take the inner sub-backend buffer out of a `Buf<HybridBackend, F>`,
/// leaving a zero-sized placeholder. Caller must overwrite the slot before
/// the placeholder is observed.
fn take_primary_buf<F: Field, P: ComputeBackend, Fb: ComputeBackend>(
    primary: &P,
    db: &mut Buf<HybridBackend<P, Fb>, F>,
) -> Buf<P, F> {
    match db {
        DeviceBuffer::Field(HybridBuffer::Primary(inner)) => {
            DeviceBuffer::Field(std::mem::replace(inner, primary.alloc(0)))
        }
        DeviceBuffer::U64(HybridBuffer::Primary(inner)) => {
            DeviceBuffer::U64(std::mem::replace(inner, primary.alloc(0)))
        }
        _ => panic!("expected primary buffer"),
    }
}

/// Same as [`take_primary_buf`] but for the fallback sub-backend.
fn take_fallback_buf<F: Field, P: ComputeBackend, Fb: ComputeBackend>(
    fallback: &Fb,
    db: &mut Buf<HybridBackend<P, Fb>, F>,
) -> Buf<Fb, F> {
    match db {
        DeviceBuffer::Field(HybridBuffer::Fallback(inner)) => {
            DeviceBuffer::Field(std::mem::replace(inner, fallback.alloc(0)))
        }
        DeviceBuffer::U64(HybridBuffer::Fallback(inner)) => {
            DeviceBuffer::U64(std::mem::replace(inner, fallback.alloc(0)))
        }
        _ => panic!("expected fallback buffer"),
    }
}

/// Restore a sub-backend buffer back into the hybrid `DeviceBuffer` slot
/// as a `Primary` variant.
fn restore_primary_buf<F: Field, P: ComputeBackend, Fb: ComputeBackend>(
    slot: &mut Buf<HybridBackend<P, Fb>, F>,
    inner: Buf<P, F>,
) {
    match (slot, inner) {
        (DeviceBuffer::Field(hybrid), DeviceBuffer::Field(buf)) => {
            *hybrid = HybridBuffer::Primary(buf);
        }
        (DeviceBuffer::U64(hybrid), DeviceBuffer::U64(buf)) => {
            *hybrid = HybridBuffer::Primary(buf);
        }
        _ => panic!("DeviceBuffer variant mismatch during restore"),
    }
}

/// Restore a sub-backend buffer back into the hybrid `DeviceBuffer` slot
/// as a `Fallback` variant.
fn restore_fallback_buf<F: Field, P: ComputeBackend, Fb: ComputeBackend>(
    slot: &mut Buf<HybridBackend<P, Fb>, F>,
    inner: Buf<Fb, F>,
) {
    match (slot, inner) {
        (DeviceBuffer::Field(hybrid), DeviceBuffer::Field(buf)) => {
            *hybrid = HybridBuffer::Fallback(buf);
        }
        (DeviceBuffer::U64(hybrid), DeviceBuffer::U64(buf)) => {
            *hybrid = HybridBuffer::Fallback(buf);
        }
        _ => panic!("DeviceBuffer variant mismatch during restore"),
    }
}

/// Migrate a `Buf<HybridBackend, F>` slot from primary to fallback in-place.
fn migrate_buf_to_fallback<F: Field, P: ComputeBackend, Fb: ComputeBackend>(
    primary: &P,
    fallback: &Fb,
    slot: &mut Buf<HybridBackend<P, Fb>, F>,
) {
    match slot {
        DeviceBuffer::Field(hybrid) => match hybrid {
            HybridBuffer::Primary(inner) => {
                let data = primary.download(inner);
                *hybrid = HybridBuffer::Fallback(fallback.upload(&data));
            }
            HybridBuffer::Fallback(_) => {}
        },
        DeviceBuffer::U64(hybrid) => match hybrid {
            HybridBuffer::Primary(inner) => {
                let data = primary.download(inner);
                *hybrid = HybridBuffer::Fallback(fallback.upload(&data));
            }
            HybridBuffer::Fallback(_) => {}
        },
    }
}

impl<P: ComputeBackend, Fb: ComputeBackend> ComputeBackend for HybridBackend<P, Fb> {
    type Buffer<T: Scalar> = HybridBuffer<T, P, Fb>;
    type CompiledKernel<Fld: Field> = HybridKernel<Fld, P, Fb>;

    fn compile<F: Field>(&self, spec: &KernelSpec) -> Self::CompiledKernel<F> {
        HybridKernel {
            primary: self.primary.compile(spec),
            fallback: self.fallback.compile(spec),
        }
    }

    fn reduce<F: Field>(
        &self,
        kernel: &Self::CompiledKernel<F>,
        inputs: &[&Buf<Self, F>],
        challenges: &[F],
    ) -> Vec<F> {
        if inputs.is_empty() {
            return Vec::new();
        }

        // All inputs must be on the same sub-backend. Check the first one's
        // field buffer to determine which path to take.
        let on_primary = match inputs[0] {
            DeviceBuffer::Field(h) => h.is_primary(),
            DeviceBuffer::U64(h) => h.is_primary(),
        };

        if on_primary {
            // Create temporary owned Buf<P, F> values by cloning inner data.
            // For CPU sub-backends this is Vec::clone; for GPU this is a
            // device roundtrip (acceptable for the read-only reduce path).
            let temp_bufs: Vec<Buf<P, F>> = inputs
                .iter()
                .map(|db| clone_primary_buf(&self.primary, db))
                .collect();
            let refs: Vec<&Buf<P, F>> = temp_bufs.iter().collect();
            self.primary.reduce(&kernel.primary, &refs, challenges)
        } else {
            let temp_bufs: Vec<Buf<Fb, F>> = inputs
                .iter()
                .map(|db| clone_fallback_buf(&self.fallback, db))
                .collect();
            let refs: Vec<&Buf<Fb, F>> = temp_bufs.iter().collect();
            self.fallback.reduce(&kernel.fallback, &refs, challenges)
        }
    }

    fn bind<F: Field>(
        &self,
        kernel: &Self::CompiledKernel<F>,
        inputs: &mut [Buf<Self, F>],
        scalar: F,
    ) {
        if inputs.is_empty() {
            return;
        }

        // All inputs must be on the same sub-backend.
        let on_primary = match &inputs[0] {
            DeviceBuffer::Field(h) => h.is_primary(),
            DeviceBuffer::U64(h) => h.is_primary(),
        };

        if on_primary {
            // Take inner buffers out, delegate to primary, restore.
            let mut sub_bufs: Vec<Buf<P, F>> = inputs
                .iter_mut()
                .map(|db| take_primary_buf(&self.primary, db))
                .collect();

            self.primary.bind(&kernel.primary, &mut sub_bufs, scalar);

            // Check migration: use the first field buffer's post-bind size.
            let post_len = sub_bufs
                .iter()
                .find_map(|b| {
                    if b.is_field() {
                        Some(self.primary.len(b.as_field()))
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let migrate = self.should_migrate(post_len);

            if migrate {
                for (slot, sub_buf) in inputs.iter_mut().zip(sub_bufs) {
                    // Wrap back as primary first, then migrate.
                    restore_primary_buf(slot, sub_buf);
                    migrate_buf_to_fallback(&self.primary, &self.fallback, slot);
                }
            } else {
                for (slot, sub_buf) in inputs.iter_mut().zip(sub_bufs) {
                    restore_primary_buf(slot, sub_buf);
                }
            }
        } else {
            let mut sub_bufs: Vec<Buf<Fb, F>> = inputs
                .iter_mut()
                .map(|db| take_fallback_buf(&self.fallback, db))
                .collect();

            self.fallback.bind(&kernel.fallback, &mut sub_bufs, scalar);

            for (slot, sub_buf) in inputs.iter_mut().zip(sub_bufs) {
                restore_fallback_buf(slot, sub_buf);
            }
        }
    }

    fn interpolate_inplace<F: Field>(
        &self,
        buf: &mut Self::Buffer<F>,
        scalar: F,
        order: BindingOrder,
    ) {
        let migrate = buf.is_primary() && self.should_migrate(self.len(buf) / 2);

        match buf {
            HybridBuffer::Primary(b) => self.primary.interpolate_inplace(b, scalar, order),
            HybridBuffer::Fallback(b) => self.fallback.interpolate_inplace(b, scalar, order),
        }

        if migrate {
            let data = self.download(buf);
            *buf = HybridBuffer::Fallback(self.fallback.upload(&data));
        }
    }

    fn upload<T: Scalar>(&self, data: &[T]) -> Self::Buffer<T> {
        if self.should_migrate(data.len()) {
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
        if self.should_migrate(len) {
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

    fn eq_table<F: Field>(&self, point: &[F]) -> Self::Buffer<F> {
        let len = 1usize << point.len();
        if self.should_migrate(len) {
            HybridBuffer::Fallback(self.fallback.eq_table(point))
        } else {
            HybridBuffer::Primary(self.primary.eq_table(point))
        }
    }

    fn lt_table<F: Field>(&self, point: &[F]) -> Self::Buffer<F> {
        let len = 1usize << point.len();
        if self.should_migrate(len) {
            HybridBuffer::Fallback(self.fallback.lt_table(point))
        } else {
            HybridBuffer::Primary(self.primary.lt_table(point))
        }
    }

    fn eq_plus_one_table<F: Field>(&self, point: &[F]) -> (Self::Buffer<F>, Self::Buffer<F>) {
        let len = 1usize << point.len();
        if self.should_migrate(len) {
            let (eq, epo) = self.fallback.eq_plus_one_table(point);
            (HybridBuffer::Fallback(eq), HybridBuffer::Fallback(epo))
        } else {
            let (eq, epo) = self.primary.eq_plus_one_table(point);
            (HybridBuffer::Primary(eq), HybridBuffer::Primary(epo))
        }
    }

    fn duplicate_interleave<F: Field>(&self, _buf: &Self::Buffer<F>) -> Self::Buffer<F> {
        panic!("HybridBackend: not yet wired")
    }

    fn regroup_constraints<F: Field>(
        &self,
        _buf: &Self::Buffer<F>,
        _group_indices: &[Vec<usize>],
        _old_stride: usize,
        _new_stride: usize,
        _num_cycles: usize,
    ) -> Self::Buffer<F> {
        panic!("HybridBackend: not yet wired")
    }

    fn evaluate_claim<F: Field>(
        &self,
        _formula: &jolt_compiler::module::ClaimFormula,
        _evaluations: &std::collections::HashMap<jolt_compiler::PolynomialId, F>,
        _staged_evals: &std::collections::HashMap<(jolt_compiler::PolynomialId, usize), F>,
        _challenges: &[F],
    ) -> F {
        panic!("HybridBackend: not yet wired")
    }

    fn evaluate_mle<F: Field>(&self, _evals: &[F], _point: &[F]) -> F {
        panic!("HybridBackend: not yet wired")
    }

    fn uniskip_encode<F: Field>(
        &self,
        _raw_evals: &mut [F],
        _domain_size: usize,
        _domain_start: i64,
        _tau: F,
        _zero_base: bool,
        _num_coeffs: usize,
    ) -> Vec<F> {
        panic!("HybridBackend: not yet wired")
    }

    fn compressed_encode<F: Field>(&self, _evals: &[F]) -> Vec<F> {
        panic!("HybridBackend: not yet wired")
    }

    fn interpolate_evaluate<F: Field>(&self, _evals: &[F], _point: F) -> F {
        panic!("HybridBackend: not yet wired")
    }

    fn extend_evals<F: Field>(&self, _evals: &[F], _target_len: usize) -> Vec<F> {
        panic!("HybridBackend: not yet wired")
    }

    fn scale_from_host<F: Field>(&self, _data: &[F], _scale: F) -> Self::Buffer<F> {
        panic!("HybridBackend: not yet wired")
    }

    fn transpose_from_host<F: Field>(
        &self,
        _data: &[F],
        _rows: usize,
        _cols: usize,
    ) -> Self::Buffer<F> {
        panic!("HybridBackend: not yet wired")
    }

    fn eq_gather<F: Field>(&self, _eq_point: &[F], _index_data: &[F]) -> Self::Buffer<F> {
        panic!("HybridBackend: not yet wired")
    }

    fn eq_pushforward<F: Field>(
        &self,
        _eq_point: &[F],
        _index_data: &[F],
        _output_size: usize,
    ) -> Self::Buffer<F> {
        panic!("HybridBackend: not yet wired")
    }

    fn eq_project<F: Field>(
        &self,
        _source_data: &[F],
        _eq_point: &[F],
        _inner_size: usize,
        _outer_size: usize,
    ) -> Self::Buffer<F> {
        panic!("HybridBackend: not yet wired")
    }

    fn lagrange_project<F: Field>(
        &self,
        _buf: &Self::Buffer<F>,
        _challenge: F,
        _domain_start: i64,
        _domain_size: usize,
        _stride: usize,
        _group_offsets: &[usize],
        _scale: F,
    ) -> Self::Buffer<F> {
        panic!("HybridBackend: not yet wired")
    }

    fn segmented_reduce<F: Field>(
        &self,
        _kernel: &Self::CompiledKernel<F>,
        _inputs: &[&Self::Buffer<F>],
        _outer_eq: &[F],
        _inner_only: &[bool],
        _inner_size: usize,
        _challenges: &[F],
    ) -> Vec<F> {
        panic!("HybridBackend: not yet wired")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use jolt_compiler::{Factor, Formula, Iteration, ProductTerm};
    use jolt_field::{Field, Fr};

    struct MockBackend {
        name: &'static str,
    }

    impl MockBackend {
        fn new(name: &'static str) -> Self {
            Self { name }
        }
    }

    struct MockKernel<F: Field> {
        num_evals: usize,
        binding_order: BindingOrder,
        _marker: std::marker::PhantomData<F>,
    }

    impl ComputeBackend for MockBackend {
        type Buffer<T: Scalar> = Vec<T>;
        type CompiledKernel<F: Field> = MockKernel<F>;

        fn compile<F: Field>(&self, spec: &KernelSpec) -> MockKernel<F> {
            MockKernel {
                num_evals: spec.num_evals,
                binding_order: spec.binding_order,
                _marker: std::marker::PhantomData,
            }
        }

        fn reduce<F: Field>(
            &self,
            kernel: &MockKernel<F>,
            _inputs: &[&Buf<Self, F>],
            _challenges: &[F],
        ) -> Vec<F> {
            if self.name == "primary" {
                vec![F::from_u64(1); kernel.num_evals]
            } else {
                vec![F::from_u64(2); kernel.num_evals]
            }
        }

        fn bind<F: Field>(&self, kernel: &MockKernel<F>, inputs: &mut [Buf<Self, F>], scalar: F) {
            for buf in inputs.iter_mut() {
                match buf {
                    DeviceBuffer::Field(v) => {
                        interpolate_mock(v, scalar, kernel.binding_order);
                    }
                    DeviceBuffer::U64(_) => {}
                }
            }
        }

        fn interpolate_inplace<F: Field>(&self, buf: &mut Vec<F>, scalar: F, order: BindingOrder) {
            interpolate_mock(buf, scalar, order);
        }

        fn upload<T: Scalar>(&self, data: &[T]) -> Vec<T> {
            data.to_vec()
        }

        fn download<T: Scalar>(&self, buf: &Vec<T>) -> Vec<T> {
            buf.clone()
        }

        fn alloc<T: Scalar>(&self, len: usize) -> Vec<T> {
            // SAFETY: all Scalar types used in tests (Fr, integers, bool)
            // have all-zeros as a valid bit pattern.
            vec![unsafe { std::mem::zeroed() }; len]
        }

        fn len<T: Scalar>(&self, buf: &Vec<T>) -> usize {
            buf.len()
        }

        fn eq_table<F: Field>(&self, point: &[F]) -> Vec<F> {
            vec![F::one(); 1 << point.len()]
        }

        fn lt_table<F: Field>(&self, point: &[F]) -> Vec<F> {
            vec![F::zero(); 1 << point.len()]
        }

        fn eq_plus_one_table<F: Field>(&self, point: &[F]) -> (Vec<F>, Vec<F>) {
            let len = 1 << point.len();
            (vec![F::one(); len], vec![F::zero(); len])
        }

        fn duplicate_interleave<F: Field>(&self, _buf: &Vec<F>) -> Vec<F> {
            panic!("mock")
        }
        fn regroup_constraints<F: Field>(
            &self,
            _buf: &Vec<F>,
            _gi: &[Vec<usize>],
            _os: usize,
            _ns: usize,
            _nc: usize,
        ) -> Vec<F> {
            panic!("mock")
        }
        fn evaluate_claim<F: Field>(
            &self,
            _f: &jolt_compiler::module::ClaimFormula,
            _e: &std::collections::HashMap<jolt_compiler::PolynomialId, F>,
            _se: &std::collections::HashMap<(jolt_compiler::PolynomialId, usize), F>,
            _c: &[F],
        ) -> F {
            panic!("mock")
        }
        fn evaluate_mle<F: Field>(&self, _evals: &[F], _point: &[F]) -> F {
            panic!("mock")
        }
        fn uniskip_encode<F: Field>(
            &self,
            _r: &mut [F],
            _ds: usize,
            _dstart: i64,
            _tau: F,
            _zb: bool,
            _nc: usize,
        ) -> Vec<F> {
            panic!("mock")
        }
        fn compressed_encode<F: Field>(&self, _evals: &[F]) -> Vec<F> {
            panic!("mock")
        }
        fn interpolate_evaluate<F: Field>(&self, _evals: &[F], _point: F) -> F {
            panic!("mock")
        }
        fn extend_evals<F: Field>(&self, _evals: &[F], _target_len: usize) -> Vec<F> {
            panic!("mock")
        }
        fn scale_from_host<F: Field>(&self, _data: &[F], _scale: F) -> Vec<F> {
            panic!("mock")
        }
        fn transpose_from_host<F: Field>(&self, _data: &[F], _r: usize, _c: usize) -> Vec<F> {
            panic!("mock")
        }
        fn eq_gather<F: Field>(&self, _eq_point: &[F], _index_data: &[F]) -> Vec<F> {
            panic!("mock")
        }
        fn eq_pushforward<F: Field>(
            &self,
            _eq_point: &[F],
            _index_data: &[F],
            _output_size: usize,
        ) -> Vec<F> {
            panic!("mock")
        }
        fn eq_project<F: Field>(
            &self,
            _src: &[F],
            _eq: &[F],
            _inner: usize,
            _outer: usize,
        ) -> Vec<F> {
            panic!("mock")
        }
        fn lagrange_project<F: Field>(
            &self,
            _buf: &Vec<F>,
            _ch: F,
            _ds: i64,
            _dsz: usize,
            _stride: usize,
            _go: &[usize],
            _scale: F,
        ) -> Vec<F> {
            panic!("mock")
        }
        fn segmented_reduce<F: Field>(
            &self,
            _k: &MockKernel<F>,
            _inputs: &[&Vec<F>],
            _oeq: &[F],
            _io: &[bool],
            _is: usize,
            _ch: &[F],
        ) -> Vec<F> {
            panic!("mock")
        }
    }

    fn interpolate_mock<F: Field>(buf: &mut Vec<F>, scalar: F, order: BindingOrder) {
        let half = buf.len() / 2;
        match order {
            BindingOrder::LowToHigh => {
                for i in 0..half {
                    buf[i] = buf[2 * i] + scalar * (buf[2 * i + 1] - buf[2 * i]);
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

    fn make_hybrid(threshold: usize) -> HybridBackend<MockBackend, MockBackend> {
        HybridBackend::new(
            MockBackend::new("primary"),
            MockBackend::new("fallback"),
            threshold,
        )
    }

    fn make_spec() -> KernelSpec {
        KernelSpec {
            formula: Formula::from_terms(vec![ProductTerm {
                coefficient: 1,
                factors: vec![
                    Factor::Input(0),
                    Factor::Input(1),
                    Factor::Input(2),
                    Factor::Input(3),
                ],
            }]),
            num_evals: 4,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        }
    }

    #[test]
    fn upload_large_goes_to_primary() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..16).map(Fr::from_u64).collect();
        let buf = hybrid.upload(&data);
        assert!(buf.is_primary());
        assert_eq!(hybrid.len(&buf), 16);
    }

    #[test]
    fn upload_small_goes_to_fallback() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..4).map(Fr::from_u64).collect();
        let buf = hybrid.upload(&data);
        assert!(!buf.is_primary());
        assert_eq!(hybrid.len(&buf), 4);
    }

    #[test]
    fn interpolate_inplace_migrates_at_threshold() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..8).map(Fr::from_u64).collect();
        let mut buf = hybrid.upload(&data);
        assert!(buf.is_primary());

        hybrid.interpolate_inplace(&mut buf, Fr::from_u64(2), BindingOrder::LowToHigh);
        assert!(!buf.is_primary(), "should have migrated to fallback");
        assert_eq!(hybrid.len(&buf), 4);
    }

    #[test]
    fn interpolate_inplace_stays_primary_above_threshold() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..16).map(Fr::from_u64).collect();
        let mut buf = hybrid.upload(&data);
        assert!(buf.is_primary());

        hybrid.interpolate_inplace(&mut buf, Fr::from_u64(2), BindingOrder::LowToHigh);
        assert!(buf.is_primary(), "should stay on primary");
        assert_eq!(hybrid.len(&buf), 8);
    }

    #[test]
    fn interpolate_inplace_fallback_stays_fallback() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..4).map(Fr::from_u64).collect();
        let mut buf = hybrid.upload(&data);
        assert!(!buf.is_primary());

        hybrid.interpolate_inplace(&mut buf, Fr::from_u64(2), BindingOrder::LowToHigh);
        assert!(!buf.is_primary());
        assert_eq!(hybrid.len(&buf), 2);
    }

    #[test]
    fn reduce_dispatches_to_correct_backend() {
        let hybrid = make_hybrid(4);
        let spec = make_spec();
        let kernel = hybrid.compile::<Fr>(&spec);

        // Primary buffers -> primary.reduce returns all 1s
        let large: Vec<Fr> = vec![Fr::from_u64(1); 16];
        let buf_a: HybridBuffer<Fr, MockBackend, MockBackend> = hybrid.upload(&large);
        assert!(buf_a.is_primary());

        let db_a = DeviceBuffer::Field(buf_a);
        let result = hybrid.reduce(&kernel, &[&db_a], &[]);
        assert_eq!(result, vec![Fr::from_u64(1); 4]);

        // Fallback buffers -> fallback.reduce returns all 2s
        let small: Vec<Fr> = vec![Fr::from_u64(1); 4];
        let buf_b: HybridBuffer<Fr, MockBackend, MockBackend> = hybrid.upload(&small);
        assert!(!buf_b.is_primary());

        let db_b = DeviceBuffer::Field(buf_b);
        let result = hybrid.reduce(&kernel, &[&db_b], &[]);
        assert_eq!(result, vec![Fr::from_u64(2); 4]);
    }

    #[test]
    fn bind_delegates_and_migrates() {
        let hybrid = make_hybrid(4);
        let spec = KernelSpec {
            formula: Formula::from_terms(vec![ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0)],
            }]),
            num_evals: 2,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        };
        let kernel = hybrid.compile::<Fr>(&spec);

        let data: Vec<Fr> = (0..8).map(Fr::from_u64).collect();
        let buf: HybridBuffer<Fr, MockBackend, MockBackend> = hybrid.upload(&data);
        assert!(buf.is_primary());

        let mut inputs = vec![DeviceBuffer::Field(buf)];
        hybrid.bind(&kernel, &mut inputs, Fr::from_u64(0));

        // 8 -> 4, at threshold -> migrated to fallback
        match &inputs[0] {
            DeviceBuffer::Field(h) => {
                assert!(!h.is_primary(), "should have migrated to fallback");
                assert_eq!(hybrid.len(h), 4);
            }
            DeviceBuffer::U64(_) => panic!("expected Field"),
        }
    }

    #[test]
    fn bind_stays_primary_above_threshold() {
        let hybrid = make_hybrid(4);
        let spec = KernelSpec {
            formula: Formula::from_terms(vec![ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0)],
            }]),
            num_evals: 2,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        };
        let kernel = hybrid.compile::<Fr>(&spec);

        let data: Vec<Fr> = (0..16).map(Fr::from_u64).collect();
        let buf: HybridBuffer<Fr, MockBackend, MockBackend> = hybrid.upload(&data);
        assert!(buf.is_primary());

        let mut inputs = vec![DeviceBuffer::Field(buf)];
        hybrid.bind(&kernel, &mut inputs, Fr::from_u64(0));

        // 16 -> 8, above threshold -> stays primary
        match &inputs[0] {
            DeviceBuffer::Field(h) => {
                assert!(h.is_primary(), "should stay on primary");
                assert_eq!(hybrid.len(h), 8);
            }
            DeviceBuffer::U64(_) => panic!("expected Field"),
        }
    }

    #[test]
    fn bind_fallback_stays_fallback() {
        let hybrid = make_hybrid(4);
        let spec = KernelSpec {
            formula: Formula::from_terms(vec![ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0)],
            }]),
            num_evals: 2,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        };
        let kernel = hybrid.compile::<Fr>(&spec);

        let data: Vec<Fr> = (0..4).map(Fr::from_u64).collect();
        let buf: HybridBuffer<Fr, MockBackend, MockBackend> = hybrid.upload(&data);
        assert!(!buf.is_primary());

        let mut inputs = vec![DeviceBuffer::Field(buf)];
        hybrid.bind(&kernel, &mut inputs, Fr::from_u64(0));

        match &inputs[0] {
            DeviceBuffer::Field(h) => {
                assert!(!h.is_primary());
                assert_eq!(hybrid.len(h), 2);
            }
            DeviceBuffer::U64(_) => panic!("expected Field"),
        }
    }

    #[test]
    fn eq_table_respects_threshold() {
        let hybrid = make_hybrid(8);

        let small_table = hybrid.eq_table::<Fr>(&[Fr::from_u64(1), Fr::from_u64(2)]);
        assert!(!small_table.is_primary());

        let large_table = hybrid.eq_table::<Fr>(&[
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ]);
        assert!(large_table.is_primary());
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
    fn full_lifecycle_primary_to_fallback() {
        let hybrid = make_hybrid(4);

        let data: Vec<Fr> = (0..32).map(Fr::from_u64).collect();
        let mut buf = hybrid.upload(&data);
        assert!(buf.is_primary());
        assert_eq!(hybrid.len(&buf), 32);

        // 32 -> 16, stays primary
        hybrid.interpolate_inplace(&mut buf, Fr::from_u64(1), BindingOrder::LowToHigh);
        assert!(buf.is_primary());
        assert_eq!(hybrid.len(&buf), 16);

        // 16 -> 8, stays primary
        hybrid.interpolate_inplace(&mut buf, Fr::from_u64(1), BindingOrder::LowToHigh);
        assert!(buf.is_primary());
        assert_eq!(hybrid.len(&buf), 8);

        // 8 -> 4, migrates to fallback
        hybrid.interpolate_inplace(&mut buf, Fr::from_u64(1), BindingOrder::LowToHigh);
        assert!(!buf.is_primary());
        assert_eq!(hybrid.len(&buf), 4);

        // 4 -> 2, stays on fallback
        hybrid.interpolate_inplace(&mut buf, Fr::from_u64(1), BindingOrder::LowToHigh);
        assert!(!buf.is_primary());
        assert_eq!(hybrid.len(&buf), 2);
    }

    #[test]
    fn bind_values_correct_after_migration_low_to_high() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = [10, 20, 30, 40, 50, 60, 70, 80]
            .into_iter()
            .map(Fr::from_u64)
            .collect();
        let mut buf = hybrid.upload(&data);

        // scalar=0 takes lo values: [10, 30, 50, 70]
        use num_traits::Zero;
        hybrid.interpolate_inplace(&mut buf, Fr::zero(), BindingOrder::LowToHigh);
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
        assert!(!buf.is_primary());
    }

    #[test]
    fn bind_values_correct_after_migration_high_to_low() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = [10, 20, 30, 40, 50, 60, 70, 80]
            .into_iter()
            .map(Fr::from_u64)
            .collect();

        // H2L layout: lo = first half [10,20,30,40], hi = second half [50,60,70,80]
        // scalar=0 -> keeps lo half
        let mut buf = hybrid.upload(&data);
        use num_traits::Zero;
        hybrid.interpolate_inplace(&mut buf, Fr::zero(), BindingOrder::HighToLow);
        let result = hybrid.download(&buf);
        assert_eq!(
            result,
            vec![
                Fr::from_u64(10),
                Fr::from_u64(20),
                Fr::from_u64(30),
                Fr::from_u64(40),
            ]
        );
        assert!(!buf.is_primary());
    }

    #[test]
    #[should_panic(expected = "expected primary buffer")]
    fn mixed_backends_panics_in_reduce() {
        let hybrid = make_hybrid(4);
        let spec = make_spec();
        let kernel = hybrid.compile::<Fr>(&spec);

        let large: Vec<Fr> = vec![Fr::from_u64(1); 16];
        let small: Vec<Fr> = vec![Fr::from_u64(1); 4];
        let buf_p: HybridBuffer<Fr, MockBackend, MockBackend> = hybrid.upload(&large);
        let buf_f: HybridBuffer<Fr, MockBackend, MockBackend> = hybrid.upload(&small);
        assert!(buf_p.is_primary());
        assert!(!buf_f.is_primary());

        // First input is primary, second is fallback -> clone_primary_buf panics
        let db_p = DeviceBuffer::Field(buf_p);
        let db_f = DeviceBuffer::Field(buf_f);
        let _ = hybrid.reduce(&kernel, &[&db_p, &db_f], &[]);
    }
}
