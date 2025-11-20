use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::panic;
use std::ptr;
use std::slice;

use ark_bn254::Fr;
use common::jolt_device::MemoryLayout;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::Serializable;
use tracer::instruction::Instruction;
use tracing_oslog::OsLogger;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

// Provide dummy heap pointer for jolt-platform (only used in guest code, not needed for host FFI)
#[no_mangle]
static __HEAP_PTR: usize = 0;

type RV64IMACProver<'a> = JoltCpuProver<'a, Fr, DoryCommitmentScheme, Blake2bTranscript>;
type RV64IMACPreprocessing = JoltProverPreprocessing<Fr, DoryCommitmentScheme>;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn set_last_error(err: String) {
    LAST_ERROR.with(|last| {
        *last.borrow_mut() = Some(
            CString::new(err)
                .unwrap_or_else(|_| CString::new("Error message contained null byte").unwrap()),
        );
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|last| {
        *last.borrow_mut() = None;
    });
}

/// Get the last error message. Returns NULL if no error has occurred.
/// The returned string is valid until the next error occurs or the thread exits.
#[no_mangle]
pub extern "C" fn jolt_last_error() -> *const c_char {
    LAST_ERROR.with(|last| match &*last.borrow() {
        Some(err) => err.as_ptr(),
        None => ptr::null(),
    })
}

/// Opaque type representing JoltProverPreprocessing
pub struct JoltProverPreprocessingHandle {
    preprocessing: RV64IMACPreprocessing,
}

/// Opaque type representing JoltCpuProver
pub struct JoltCpuProverHandle<'a> {
    prover: RV64IMACProver<'a>,
}

#[no_mangle]
pub extern "C" fn jolt_ios_logging_init() {
    tracing_subscriber::registry()
        .with(OsLogger::new("testingtesting.JoltProver", "jolt_core"))
        .init();
}

/// Reset Dory global state
/// This should be called before generating a proof to clear any cached state
/// from previous proofs with different dimensions
#[no_mangle]
pub extern "C" fn jolt_reset_dory_globals() {
    jolt_core::poly::commitment::dory::DoryGlobals::reset();
}

/// Create preprocessing from bytecode, memory layout, memory init, and max trace length.
///
/// # Arguments
/// * `bytecode_ptr` - Pointer to serialized bytecode (Vec<Instruction>)
/// * `bytecode_len` - Length of bytecode buffer
/// * `memory_layout_ptr` - Pointer to serialized MemoryLayout
/// * `memory_layout_len` - Length of memory layout buffer
/// * `memory_init_ptr` - Pointer to serialized memory initialization Vec<(u64, u8)>
/// * `memory_init_len` - Length of memory init buffer
/// * `max_trace_length` - Maximum trace length
///
/// Returns NULL on error. Call jolt_last_error() to get error message.
#[no_mangle]
pub extern "C" fn jolt_prover_preprocessing_gen(
    bytecode_ptr: *const u8,
    bytecode_len: usize,
    memory_layout_ptr: *const u8,
    memory_layout_len: usize,
    memory_init_ptr: *const u8,
    memory_init_len: usize,
    max_trace_length: usize,
) -> *mut JoltProverPreprocessingHandle {
    clear_last_error();

    let result = panic::catch_unwind(|| {
        if bytecode_ptr.is_null() || memory_layout_ptr.is_null() || memory_init_ptr.is_null() {
            set_last_error("Null pointer passed to jolt_prover_preprocessing_gen".to_string());
            return ptr::null_mut();
        }

        // TODO this might not be the right place to do this
        // Initialize inline handlers (keccak256/SHA3, SHA2, etc.)
        let _ = jolt_inlines_keccak256::init_inlines();
        let _ = jolt_inlines_sha2::init_inlines();

        let bytecode_bytes = unsafe { slice::from_raw_parts(bytecode_ptr, bytecode_len) };
        let memory_layout_bytes =
            unsafe { slice::from_raw_parts(memory_layout_ptr, memory_layout_len) };
        let memory_init_bytes = unsafe { slice::from_raw_parts(memory_init_ptr, memory_init_len) };

        let bytecode: Vec<Instruction> = match postcard::from_bytes(bytecode_bytes) {
            Ok(b) => b,
            Err(e) => {
                set_last_error(format!("Failed to deserialize bytecode: {}", e));
                return ptr::null_mut();
            }
        };

        let memory_layout: MemoryLayout = match postcard::from_bytes(memory_layout_bytes) {
            Ok(m) => m,
            Err(e) => {
                set_last_error(format!("Failed to deserialize memory layout: {}", e));
                return ptr::null_mut();
            }
        };

        let memory_init: Vec<(u64, u8)> = match postcard::from_bytes(memory_init_bytes) {
            Ok(m) => m,
            Err(e) => {
                set_last_error(format!("Failed to deserialize memory init: {}", e));
                return ptr::null_mut();
            }
        };

        let preprocessing =
            RV64IMACPreprocessing::gen(bytecode, memory_layout, memory_init, max_trace_length);

        Box::into_raw(Box::new(JoltProverPreprocessingHandle { preprocessing }))
    });

    result.unwrap_or_else(|e| {
        let msg = if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "Unknown panic in jolt_prover_preprocessing_gen".to_string()
        };
        set_last_error(msg);
        ptr::null_mut()
    })
}

/// Save preprocessing to a file.
///
/// # Arguments
/// * `preprocessing` - Preprocessing handle
/// * `file_path` - Null-terminated path to output file
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn jolt_prover_preprocessing_save(
    preprocessing: *const JoltProverPreprocessingHandle,
    file_path: *const c_char,
) -> c_int {
    clear_last_error();

    let result = panic::catch_unwind(|| {
        if preprocessing.is_null() || file_path.is_null() {
            set_last_error("Null pointer passed to jolt_prover_preprocessing_save".to_string());
            return -1;
        }

        let preprocessing_ref = unsafe { &*preprocessing };
        let path_str = unsafe { CStr::from_ptr(file_path) };
        let path = match path_str.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid UTF-8 in file path: {}", e));
                return -1;
            }
        };

        match preprocessing_ref.preprocessing.save_to_file(path) {
            Ok(_) => 0,
            Err(e) => {
                set_last_error(format!("Failed to save preprocessing: {}", e));
                -1
            }
        }
    });

    result.unwrap_or_else(|e| {
        let msg = if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "Unknown panic in jolt_prover_preprocessing_save".to_string()
        };
        set_last_error(msg);
        -1
    })
}

/// Load preprocessing from a file.
///
/// # Arguments
/// * `file_path` - Null-terminated path to preprocessing file
///
/// Returns NULL on error. Call jolt_last_error() to get error message.
#[no_mangle]
pub extern "C" fn jolt_prover_preprocessing_load(
    file_path: *const c_char,
) -> *mut JoltProverPreprocessingHandle {
    clear_last_error();

    let result = panic::catch_unwind(|| {
        if file_path.is_null() {
            set_last_error("Null pointer passed to jolt_prover_preprocessing_load".to_string());
            return ptr::null_mut();
        }

        let path_str = unsafe { CStr::from_ptr(file_path) };
        let path = match path_str.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid UTF-8 in file path: {}", e));
                return ptr::null_mut();
            }
        };

        match RV64IMACPreprocessing::from_file(path) {
            Ok(preprocessing) => {
                Box::into_raw(Box::new(JoltProverPreprocessingHandle { preprocessing }))
            }
            Err(e) => {
                set_last_error(format!("Failed to load preprocessing: {}", e));
                ptr::null_mut()
            }
        }
    });

    result.unwrap_or_else(|e| {
        let msg = if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "Unknown panic in jolt_prover_preprocessing_load".to_string()
        };
        set_last_error(msg);
        ptr::null_mut()
    })
}

/// Free preprocessing handle.
///
/// # Safety
/// The preprocessing pointer must be valid and not used after this call.
#[no_mangle]
pub extern "C" fn jolt_prover_preprocessing_free(
    preprocessing: *mut JoltProverPreprocessingHandle,
) {
    if !preprocessing.is_null() {
        unsafe {
            drop(Box::from_raw(preprocessing));
        }
    }
}

/// Create a JoltCpuProver from an ELF file.
///
/// # Arguments
/// * `preprocessing` - Preprocessing handle (must outlive the returned prover)
/// * `elf_contents` - Pointer to ELF file contents
/// * `elf_len` - Length of ELF buffer
/// * `inputs` - Pointer to input bytes
/// * `inputs_len` - Length of inputs buffer
/// * `untrusted_advice` - Pointer to untrusted advice bytes
/// * `untrusted_advice_len` - Length of untrusted advice buffer
/// * `trusted_advice` - Pointer to trusted advice bytes
/// * `trusted_advice_len` - Length of trusted advice buffer
///
/// Returns NULL on error. Call jolt_last_error() to get error message.
///
/// # Safety
/// The preprocessing must remain valid for the lifetime of the returned prover.
#[no_mangle]
pub extern "C" fn jolt_cpu_prover_gen_from_elf(
    preprocessing: *const JoltProverPreprocessingHandle,
    elf_contents: *const u8,
    elf_len: usize,
    inputs: *const u8,
    inputs_len: usize,
    untrusted_advice: *const u8,
    untrusted_advice_len: usize,
    trusted_advice: *const u8,
    trusted_advice_len: usize,
) -> *mut JoltCpuProverHandle<'static> {
    clear_last_error();

    let result = panic::catch_unwind(|| {
        if preprocessing.is_null() || elf_contents.is_null() {
            set_last_error("Null pointer passed to jolt_cpu_prover_gen_from_elf".to_string());
            return ptr::null_mut();
        }

        let preprocessing_ref = unsafe { &*preprocessing };
        let elf_slice = unsafe { slice::from_raw_parts(elf_contents, elf_len) };
        let inputs_slice = if inputs.is_null() {
            &[]
        } else {
            unsafe { slice::from_raw_parts(inputs, inputs_len) }
        };
        let untrusted_advice_slice = if untrusted_advice.is_null() {
            &[]
        } else {
            unsafe { slice::from_raw_parts(untrusted_advice, untrusted_advice_len) }
        };
        let trusted_advice_slice = if trusted_advice.is_null() {
            &[]
        } else {
            unsafe { slice::from_raw_parts(trusted_advice, trusted_advice_len) }
        };

        // SAFETY: We're extending the lifetime here because we need to store the prover
        // in a Box. The caller must ensure the preprocessing outlives the prover.
        let preprocessing_static: &'static RV64IMACPreprocessing =
            unsafe { std::mem::transmute(&preprocessing_ref.preprocessing) };

        let prover = RV64IMACProver::gen_from_elf(
            preprocessing_static,
            elf_slice,
            inputs_slice,
            untrusted_advice_slice,
            trusted_advice_slice,
            None,
        );

        Box::into_raw(Box::new(JoltCpuProverHandle { prover }))
    });

    result.unwrap_or_else(|e| {
        let msg = if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "Unknown panic in jolt_cpu_prover_gen_from_elf".to_string()
        };
        set_last_error(msg);
        ptr::null_mut()
    })
}

/// Generate a proof and save it to a file.
///
/// # Arguments
/// * `prover` - Prover handle (will be consumed/freed by this call)
/// * `proof_output_path` - Null-terminated path to output proof file
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// The prover pointer is consumed by this call and must not be used afterwards.
#[no_mangle]
pub extern "C" fn jolt_cpu_prover_prove(
    prover: *mut JoltCpuProverHandle<'static>,
    proof_output_path: *const c_char,
) -> c_int {
    clear_last_error();

    let result = panic::catch_unwind(|| {
        if prover.is_null() || proof_output_path.is_null() {
            set_last_error("Null pointer passed to jolt_cpu_prover_prove".to_string());
            return -1;
        }

        let prover_handle = unsafe { Box::from_raw(prover) };
        let path_str = unsafe { CStr::from_ptr(proof_output_path) };
        let path = match path_str.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid UTF-8 in file path: {}", e));
                return -1;
            }
        };

        let (proof, _debug_info) = prover_handle.prover.prove();

        match proof.save_to_file(path) {
            Ok(_) => 0,
            Err(e) => {
                set_last_error(format!("Failed to save proof: {}", e));
                -1
            }
        }
    });

    result.unwrap_or_else(|e| {
        let msg = if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "Unknown panic in jolt_cpu_prover_prove".to_string()
        };
        set_last_error(msg);
        -1
    })
}

/// Free prover handle.
///
/// # Safety
/// The prover pointer must be valid and not used after this call.
/// Note: jolt_cpu_prover_prove already frees the prover, so don't call this after proving.
#[no_mangle]
pub extern "C" fn jolt_cpu_prover_free(prover: *mut JoltCpuProverHandle<'static>) {
    if !prover.is_null() {
        unsafe {
            drop(Box::from_raw(prover));
        }
    }
}

// ============================================================================
// Postcard Serialization Helpers
// ============================================================================

/// Opaque handle for serialized data
pub struct SerializedDataHandle {
    data: Vec<u8>,
}

/// Serialize a single u32 value using postcard.
///
/// # Returns
/// A handle to the serialized data, or NULL on error.
/// The caller must free the handle using jolt_serialized_free().
#[no_mangle]
pub extern "C" fn jolt_serialize_u32(value: u32) -> *mut SerializedDataHandle {
    clear_last_error();

    match postcard::to_stdvec(&value) {
        Ok(data) => Box::into_raw(Box::new(SerializedDataHandle { data })),
        Err(e) => {
            set_last_error(format!("Failed to serialize u32: {}", e));
            ptr::null_mut()
        }
    }
}

/// Serialize a single u64 value using postcard.
///
/// # Returns
/// A handle to the serialized data, or NULL on error.
/// The caller must free the handle using jolt_serialized_free().
#[no_mangle]
pub extern "C" fn jolt_serialize_u64(value: u64) -> *mut SerializedDataHandle {
    clear_last_error();

    match postcard::to_stdvec(&value) {
        Ok(data) => Box::into_raw(Box::new(SerializedDataHandle { data })),
        Err(e) => {
            set_last_error(format!("Failed to serialize u64: {}", e));
            ptr::null_mut()
        }
    }
}

/// Serialize an array of u64 values using postcard.
///
/// # Arguments
/// * `values` - Pointer to array of u64 values
/// * `len` - Number of elements in the array
///
/// # Returns
/// A handle to the serialized data, or NULL on error.
/// The caller must free the handle using jolt_serialized_free().
#[no_mangle]
pub extern "C" fn jolt_serialize_u64_array(
    values: *const u64,
    len: usize,
) -> *mut SerializedDataHandle {
    clear_last_error();

    if values.is_null() {
        set_last_error("Null pointer passed to jolt_serialize_u64_array".to_string());
        return ptr::null_mut();
    }

    let slice = unsafe { slice::from_raw_parts(values, len) };
    let vec: Vec<u64> = slice.to_vec();

    match postcard::to_stdvec(&vec) {
        Ok(data) => Box::into_raw(Box::new(SerializedDataHandle { data })),
        Err(e) => {
            set_last_error(format!("Failed to serialize u64 array: {}", e));
            ptr::null_mut()
        }
    }
}

/// Serialize a string using postcard.
///
/// # Arguments
/// * `str_ptr` - Pointer to UTF-8 string bytes
/// * `str_len` - Length of string in bytes
///
/// # Returns
/// A handle to the serialized data, or NULL on error.
/// The caller must free the handle using jolt_serialized_free().
#[no_mangle]
pub extern "C" fn jolt_serialize_string(
    str_ptr: *const u8,
    str_len: usize,
) -> *mut SerializedDataHandle {
    clear_last_error();

    if str_ptr.is_null() {
        set_last_error("Null pointer passed to jolt_serialize_string".to_string());
        return ptr::null_mut();
    }

    let bytes = unsafe { slice::from_raw_parts(str_ptr, str_len) };
    let string = match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in string: {}", e));
            return ptr::null_mut();
        }
    };

    match postcard::to_stdvec(&string) {
        Ok(data) => Box::into_raw(Box::new(SerializedDataHandle { data })),
        Err(e) => {
            set_last_error(format!("Failed to serialize string: {}", e));
            ptr::null_mut()
        }
    }
}

/// Get pointer and length of serialized data.
///
/// # Arguments
/// * `handle` - Handle to serialized data
/// * `out_ptr` - Output pointer to data bytes
/// * `out_len` - Output length of data
///
/// # Returns
/// 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn jolt_serialized_get_data(
    handle: *const SerializedDataHandle,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
) -> c_int {
    clear_last_error();

    if handle.is_null() || out_ptr.is_null() || out_len.is_null() {
        set_last_error("Null pointer passed to jolt_serialized_get_data".to_string());
        return -1;
    }

    let handle_ref = unsafe { &*handle };
    unsafe {
        *out_ptr = handle_ref.data.as_ptr();
        *out_len = handle_ref.data.len();
    }

    0
}

/// Free serialized data handle.
///
/// # Safety
/// The handle pointer must be valid and not used after this call.
#[no_mangle]
pub extern "C" fn jolt_serialized_free(handle: *mut SerializedDataHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}
