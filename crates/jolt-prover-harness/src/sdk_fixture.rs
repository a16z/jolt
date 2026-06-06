use crate::{HarnessError, HarnessResult};
use jolt_program::{
    execution::{JoltProgram, OwnedTrace, TraceOutput},
    preprocess::JoltProgramPreprocessing,
};

#[derive(Debug, Clone)]
pub struct SdkGuestTraceRequest {
    pub guest_package: &'static str,
    pub inputs: Vec<u8>,
    pub untrusted_advice: Vec<u8>,
    pub trusted_advice: Vec<u8>,
    pub max_padded_trace_length: usize,
    pub field_inline: bool,
}

impl SdkGuestTraceRequest {
    pub fn new(guest_package: &'static str, inputs: Vec<u8>) -> Self {
        Self {
            guest_package,
            inputs,
            untrusted_advice: Vec::new(),
            trusted_advice: Vec::new(),
            max_padded_trace_length: 1 << 16,
            field_inline: false,
        }
    }

    pub const fn with_max_padded_trace_length(mut self, max_padded_trace_length: usize) -> Self {
        self.max_padded_trace_length = max_padded_trace_length;
        self
    }

    pub const fn with_field_inline(mut self, field_inline: bool) -> Self {
        self.field_inline = field_inline;
        self
    }

    pub fn with_untrusted_advice(mut self, untrusted_advice: Vec<u8>) -> Self {
        self.untrusted_advice = untrusted_advice;
        self
    }

    pub fn with_trusted_advice(mut self, trusted_advice: Vec<u8>) -> Self {
        self.trusted_advice = trusted_advice;
        self
    }
}

#[derive(Debug, Clone)]
pub struct SdkGuestTraceFixture {
    pub program: JoltProgram,
    pub preprocessing: JoltProgramPreprocessing,
    pub trace: TraceOutput<OwnedTrace>,
    pub padded_trace_length: usize,
}

#[cfg(feature = "core-fixtures")]
pub fn trace_sdk_guest(request: SdkGuestTraceRequest) -> HarnessResult<SdkGuestTraceFixture> {
    use jolt_core::host::Program;

    let mut program = Program::new(request.guest_package);
    if request.field_inline {
        enable_field_inline(&mut program)?;
    }

    let jolt_program = program.jolt_program().map_err(|error| {
        sdk_error(
            request.guest_package,
            "build Jolt program from SDK guest",
            error,
        )
    })?;
    let mut backend = tracer::TracerBackend::new();
    let trace = program
        .trace_with_backend(
            &mut backend,
            &request.inputs,
            &request.untrusted_advice,
            &request.trusted_advice,
        )
        .map_err(|error| sdk_error(request.guest_package, "trace SDK guest", error))?;
    let trace_len = trace.trace.rows().len();
    let padded_trace_length = trace_len.next_power_of_two().max(2);
    if padded_trace_length > request.max_padded_trace_length {
        return Err(HarnessError::InvalidIngestion {
            surface: request.guest_package.to_owned(),
            reason: format!(
                "trace length {trace_len} pads to {padded_trace_length}, exceeding max {}",
                request.max_padded_trace_length
            ),
        });
    }

    let preprocessing = JoltProgramPreprocessing::new(
        jolt_program.expanded_bytecode.clone(),
        jolt_program.memory_init.clone(),
        trace.device.memory_layout.clone(),
        jolt_program.entry_address,
        request.max_padded_trace_length,
        jolt_program.profile,
    )
    .map_err(|error| sdk_error(request.guest_package, "preprocess SDK guest trace", error))?;

    Ok(SdkGuestTraceFixture {
        program: jolt_program,
        preprocessing,
        trace,
        padded_trace_length,
    })
}

#[cfg(not(feature = "core-fixtures"))]
pub fn trace_sdk_guest(request: SdkGuestTraceRequest) -> HarnessResult<SdkGuestTraceFixture> {
    Err(HarnessError::InvalidIngestion {
        surface: request.guest_package.to_owned(),
        reason: "SDK guest tracing requires the core-fixtures feature".to_owned(),
    })
}

#[cfg(all(feature = "core-fixtures", feature = "field-inline"))]
fn enable_field_inline(program: &mut jolt_core::host::Program) -> HarnessResult<()> {
    program.enable_field_inline();
    Ok(())
}

#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
fn enable_field_inline(program: &mut jolt_core::host::Program) -> HarnessResult<()> {
    let _ = program;
    Err(HarnessError::MissingFeature {
        feature: "field-inline",
        context: "trace SDK field-inline guest",
    })
}

#[cfg(feature = "core-fixtures")]
fn sdk_error(
    guest_package: &'static str,
    context: &'static str,
    error: impl core::fmt::Display,
) -> HarnessError {
    HarnessError::InvalidIngestion {
        surface: guest_package.to_owned(),
        reason: format!("{context}: {error}"),
    }
}
