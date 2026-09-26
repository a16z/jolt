//! The field-operation conformance check shared by the field suites: every
//! kernel of `tests/shaders/field_ops.metal`, run on the GPU and compared
//! with `jolt_field` byte for byte.

use std::fmt::Debug;

use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::{MetalError, MetalField};

const FIELD_OPS: &str = include_str!("../shaders/field_ops.metal");
const ADD: &str = "jolt_test_field_add";
const SUB: &str = "jolt_test_field_sub";
const MUL: &str = "jolt_test_field_mul";
const NEG: &str = "jolt_test_field_neg";
const SQUARE: &str = "jolt_test_field_square";
const MUL_U64: &str = "jolt_test_field_mul_u64";
const MUL_I64: &str = "jolt_test_field_mul_i64";
const FROM_U64: &str = "jolt_test_field_from_u64";
const FROM_I64: &str = "jolt_test_field_from_i64";
pub const WRITE_NON_CANONICAL: &str = "jolt_test_field_write_non_canonical";
const KERNELS: [&str; 10] = [
    ADD,
    SUB,
    MUL,
    NEG,
    SQUARE,
    MUL_U64,
    MUL_I64,
    FROM_U64,
    FROM_I64,
    WRITE_NON_CANONICAL,
];

pub const U64_EDGES: [u64; 11] = [
    0,
    1,
    2,
    3,
    (1 << 31) - 1,
    (1 << 32) - 1,
    1 << 32,
    (1 << 63) - 1,
    1 << 63,
    u64::MAX - 1,
    u64::MAX,
];

pub const I64_EDGES: [i64; 13] = [
    0,
    1,
    -1,
    2,
    -2,
    i32::MAX as i64,
    i32::MIN as i64,
    (1 << 32) - 1,
    -(1 << 32),
    i64::MAX,
    i64::MAX - 1,
    i64::MIN,
    i64::MIN + 1,
];

/// Inputs for every operation of `field_ops.metal`.
#[derive(Default)]
pub struct Inputs<F> {
    /// Operands of `add`, `sub`, and `mul`.
    pub pairs: Vec<(F, F)>,
    /// Operands of `neg` and `square`.
    pub singles: Vec<F>,
    /// Operands of `mul_u64`; the scalars are also the inputs of `from_u64`.
    pub u64_pairs: Vec<(F, u64)>,
    /// Operands of `mul_i64`; the scalars are also the inputs of `from_i64`.
    pub i64_pairs: Vec<(F, i64)>,
}

/// Compiles the field headers, `field_ops.metal` instantiated for `F`, and
/// `extra` sources with their `extra_kernels` instantiated for `F`.
pub fn library<F: MetalField>(
    device: &Device,
    extra: &[(&str, &str)],
    extra_kernels: &[&str],
) -> ShaderLibrary {
    let spec = FIELD_HEADERS
        .iter()
        .chain(&[("field_ops.metal", FIELD_OPS)])
        .chain(extra)
        .fold(LibrarySpec::new(), |spec, (name, text)| {
            spec.source(name, text)
        });
    let spec = KERNELS
        .iter()
        .chain(extra_kernels)
        .fold(spec, |spec, kernel| spec.instantiate::<F>(kernel));
    ShaderLibrary::compile(device, &spec).unwrap()
}

/// Runs `kernel`, instantiated for `F`, over `len` threads with `inputs`
/// bound first and a fresh output buffer last, and returns the checked
/// output.
pub fn run<F: MetalField>(
    device: &Device,
    library: &ShaderLibrary,
    kernel: &str,
    inputs: Vec<Binding<'_>>,
    len: usize,
) -> Result<Vec<F>, MetalError> {
    let pipeline = library.pipeline(&host_name::<F>(kernel)).unwrap();
    let mut out = DeviceBuffer::<F>::zeroed(device, len).unwrap();
    {
        let mut bindings = inputs;
        bindings.push(Binding::buffer(&out));
        let group = (pipeline.thread_execution_width() * 8)
            .min(pipeline.max_total_threads_per_threadgroup());
        let mut batch = Batch::new(device).unwrap();
        batch
            .dispatch(pipeline, &bindings, Grid::linear(len, group))
            .unwrap();
        let _ = batch.commit_and_wait().unwrap();
    }
    out.read().map(<[F]>::to_vec)
}

/// Records the first mismatch and the mismatch count for one operation.
pub fn compare<T: PartialEq + Debug>(
    failures: &mut Vec<String>,
    op: &str,
    got: &[T],
    want: &[T],
    input: impl Fn(usize) -> String,
) {
    assert_eq!(got.len(), want.len());
    let mut wrong = got
        .iter()
        .zip(want)
        .enumerate()
        .filter(|(_, (g, w))| g != w);
    if let Some((index, (g, w))) = wrong.next() {
        failures.push(format!(
            "{op}: {} of {} results differ; first at {index}: {} gave {g:?}, expected {w:?}",
            wrong.count() + 1,
            got.len(),
            input(index),
        ));
    }
}

type BinaryOp<F> = fn(F, F) -> F;
type UnaryOp<F> = fn(F) -> F;

/// Runs every operation of `field_ops.metal` on `inputs` and records each
/// disagreement with `jolt_field` in `failures`.
pub fn check_ops<F: MetalField + Debug>(
    device: &Device,
    library: &ShaderLibrary,
    inputs: &Inputs<F>,
    failures: &mut Vec<String>,
) {
    let (a, b): (Vec<F>, Vec<F>) = inputs.pairs.iter().copied().unzip();
    let (a_dev, b_dev) = (
        DeviceBuffer::from_slice(device, &a).unwrap(),
        DeviceBuffer::from_slice(device, &b).unwrap(),
    );
    let binary: [(&str, BinaryOp<F>); 3] = [
        (ADD, |x, y| x + y),
        (SUB, |x, y| x - y),
        (MUL, |x, y| x * y),
    ];
    for (kernel, op) in binary {
        let got = run::<F>(
            device,
            library,
            kernel,
            vec![Binding::buffer(&a_dev), Binding::buffer(&b_dev)],
            a.len(),
        )
        .unwrap();
        let want: Vec<F> = a.iter().zip(&b).map(|(&x, &y)| op(x, y)).collect();
        compare(failures, kernel, &got, &want, |i| {
            format!("{:?}", inputs.pairs[i])
        });
    }

    let x_dev = DeviceBuffer::from_slice(device, &inputs.singles).unwrap();
    let unary: [(&str, UnaryOp<F>); 2] = [(NEG, |x| -x), (SQUARE, |x| x.square())];
    for (kernel, op) in unary {
        let got = run::<F>(
            device,
            library,
            kernel,
            vec![Binding::buffer(&x_dev)],
            inputs.singles.len(),
        )
        .unwrap();
        let want: Vec<F> = inputs.singles.iter().map(|&v| op(v)).collect();
        compare(failures, kernel, &got, &want, |i| {
            format!("{:?}", inputs.singles[i])
        });
    }

    let (a, s): (Vec<F>, Vec<u64>) = inputs.u64_pairs.iter().copied().unzip();
    let (a_dev, s_dev) = (
        DeviceBuffer::from_slice(device, &a).unwrap(),
        DeviceBuffer::from_slice(device, &s).unwrap(),
    );
    let got = run::<F>(
        device,
        library,
        MUL_U64,
        vec![Binding::buffer(&a_dev), Binding::buffer(&s_dev)],
        s.len(),
    )
    .unwrap();
    let want: Vec<F> = a.iter().zip(&s).map(|(x, &s)| x.mul_u64(s)).collect();
    compare(failures, MUL_U64, &got, &want, |i| {
        format!("{:?}", inputs.u64_pairs[i])
    });
    let got = run::<F>(
        device,
        library,
        FROM_U64,
        vec![Binding::buffer(&s_dev)],
        s.len(),
    )
    .unwrap();
    let want: Vec<F> = s.iter().map(|&s| F::from_u64(s)).collect();
    compare(failures, FROM_U64, &got, &want, |i| format!("{}", s[i]));

    let (a, s): (Vec<F>, Vec<i64>) = inputs.i64_pairs.iter().copied().unzip();
    let (a_dev, s_dev) = (
        DeviceBuffer::from_slice(device, &a).unwrap(),
        DeviceBuffer::from_slice(device, &s).unwrap(),
    );
    let got = run::<F>(
        device,
        library,
        MUL_I64,
        vec![Binding::buffer(&a_dev), Binding::buffer(&s_dev)],
        s.len(),
    )
    .unwrap();
    let want: Vec<F> = a.iter().zip(&s).map(|(x, &s)| x.mul_i64(s)).collect();
    compare(failures, MUL_I64, &got, &want, |i| {
        format!("{:?}", inputs.i64_pairs[i])
    });
    let got = run::<F>(
        device,
        library,
        FROM_I64,
        vec![Binding::buffer(&s_dev)],
        s.len(),
    )
    .unwrap();
    let want: Vec<F> = s.iter().map(|&s| F::from_i64(s)).collect();
    compare(failures, FROM_I64, &got, &want, |i| format!("{}", s[i]));
}
