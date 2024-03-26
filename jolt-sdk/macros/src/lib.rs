extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, PatType, ReturnType};

use common::constants::{
    INPUT_END_ADDRESS, INPUT_START_ADDRESS, OUTPUT_END_ADDRESS, OUTPUT_START_ADDRESS, PANIC_ADDRESS,
};

#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_clone = item.clone();
    let input_fn = parse_macro_input!(item_clone as ItemFn);

    let max_input_len = (INPUT_END_ADDRESS - INPUT_START_ADDRESS) as usize;

    let get_input_slice = quote! {
        let input_ptr = #INPUT_START_ADDRESS as *const u8;
        let input_slice = unsafe {
            core::slice::from_raw_parts(input_ptr, #max_input_len)
        };
    };

    let mut args = Vec::new();
    let mut arg_types = Vec::new();
    for arg in input_fn.sig.inputs {
        if let syn::FnArg::Typed(PatType { pat, ty, .. }) = arg {
            if let syn::Pat::Ident(pat_ident) = pat.as_ref() {
                let arg_name = &pat_ident.ident;
                let arg_type = &ty;

                let arg_fetch = quote! {
                    let x = input_slice[0];
                    let (#arg_name, input_slice) = jolt_sdk::postcard::take_from_bytes::<#arg_type>(input_slice).unwrap();
                };

                args.push(arg_fetch);
                arg_types.push(arg_type.clone());
            } else {
                panic!("cannot parse arg");
            }
        } else {
            panic!("cannot parse arg");
        }
    }

    // TODO: ensure that input slice hasn't overflown
    let check_input_len = quote! {};

    let block = &input_fn.block;
    let block = quote! {let to_return = (|| -> _ { #block })();};

    let max_output_len = (OUTPUT_END_ADDRESS - OUTPUT_START_ADDRESS) as usize;
    let handle_return = match &input_fn.sig.output {
        ReturnType::Default => quote! {},
        ReturnType::Type(_, _) => quote! {
            let output_ptr = #OUTPUT_START_ADDRESS as *mut u8;
            let output_slice = unsafe {
                core::slice::from_raw_parts_mut(output_ptr, #max_output_len)
            };

            jolt_sdk::postcard::to_slice(&to_return, output_slice).unwrap();
        },
    };

    let transformed_fn = quote! {
        #[cfg(feature = "guest")]
        use core::arch::global_asm;
        #[cfg(feature = "guest")]
        use core::panic::PanicInfo;

        #[cfg(feature = "guest")]
        global_asm!("\
            .global _start\n\
            .extern _STACK_PTR\n\
            .section .text.boot\n\
            _start:	la sp, _STACK_PTR\n\
                jal main\n\
                j .\n\
        ");

        #[cfg(feature = "guest")]
        #[no_mangle]
        pub extern "C" fn main() {
            let mut offset = 0;
            #get_input_slice
            #(#args;)*
            #check_input_len
            #block
            #handle_return
        }

        #[cfg(feature = "guest")]
        #[panic_handler]
        fn panic(_info: &PanicInfo) -> ! {
            unsafe {
                core::ptr::write_volatile(#PANIC_ADDRESS as *mut u8, 1);
            }

            loop {}
        }
    };

    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let inputs = &input_fn.sig.inputs;
    let output = &input_fn.sig.output;
    let body = &input_fn.block;

    let mut args = Vec::new();
    for arg in &input_fn.sig.inputs {
        if let syn::FnArg::Typed(PatType { pat, .. }) = arg {
            if let syn::Pat::Ident(pat_ident) = pat.as_ref() {
                let arg_name = &pat_ident.ident;
                let arg_set = quote! {
                    let mut program = program.input(&#arg_name);
                };

                args.push(arg_set);
            } else {
                panic!("cannot parse arg");
            }
        } else {
            panic!("cannot parse arg");
        }
    }

    let handle_return = match &input_fn.sig.output {
        ReturnType::Default => quote! {
            let ret_val = ();
        },
        ReturnType::Type(_, ty) => quote! {
            let ret_val = jolt_sdk::postcard::from_bytes::<#ty>(&output_bytes).unwrap();
        },
    };

    let guest_crate_name = std::env::var("CARGO_PKG_NAME").unwrap();

    let imports = quote! {
        #[cfg(not(feature = "guest"))]
        use jolt_sdk::{
            CurveGroup,
            PrimeField,
            host::Program,
            JoltPreprocessing,
            Jolt,
            JoltCommitments,
            RV32IJoltVM,
            RV32I,
            RV32IJoltProof,
            BytecodeRow,
            MemoryOp,
            MEMORY_OPS_PER_INSTRUCTION,
            instruction::add::ADDInstruction,
        };
    };

    let preprocess_fn_name = syn::Ident::new(&format!("preprocess_{}", fn_name), fn_name.span());
    let preprocess_fn = quote! {
        #[cfg(not(feature = "guest"))]
        pub fn #preprocess_fn_name() -> (Program, JoltPreprocessing<jolt_sdk::F, jolt_sdk::G>) {
            use jolt_sdk::tracer;

            let mut program = Program::new(#guest_crate_name);
            let bytecode = program.decode();

            // TODO(moodlezoup): Feed in size parameters via macro
            let preprocessing: JoltPreprocessing<jolt_sdk::F, jolt_sdk::G> = 
                RV32IJoltVM::preprocess(
                    bytecode,
                    1 << 20,
                    1 << 20,
                    1 << 22
                );

            (program, preprocessing)
        }
    };

    let prove_output_ty = match &input_fn.sig.output {
        ReturnType::Default => quote! { 
            ((), jolt_sdk::Proof) 
        },
        ReturnType::Type(_, ty) => quote! { 
            (#ty, jolt_sdk::Proof)
        },
    };

    let prove_fn_name = syn::Ident::new(&format!("prove_{}", fn_name), fn_name.span());
    let prove_fn = quote! {
        #[cfg(not(feature = "guest"))]
        pub fn #prove_fn_name(
            mut program: Program,
            preprocessing: JoltPreprocessing<jolt_sdk::F, jolt_sdk::G>,
            #inputs
        ) -> #prove_output_ty {
            #(#args;)*

            let bytecode = program.decode();
            let (io_device, bytecode_trace, instruction_trace, memory_trace, circuit_flags) =
                program.trace();

            let output_bytes = io_device.outputs.clone();

            let (jolt_proof, jolt_commitments) = RV32IJoltVM::prove(
                io_device,
                bytecode,
                bytecode_trace,
                memory_trace,
                instruction_trace,
                circuit_flags,
                preprocessing,
            );

            #handle_return

            let proof = jolt_sdk::Proof {
                proof: jolt_proof,
                commitments: jolt_commitments,
            };

            (ret_val, proof)
        }
    };

    let input_types: Vec<&syn::Type> = input_fn.sig.inputs.iter().map(|fn_arg| {
        if let syn::FnArg::Typed(pat_type) = fn_arg {
            &*pat_type.ty
        } else {
            panic!("unreachable");
        }
    }).collect();

    let input_names: Vec<&syn::PatIdent> = input_fn.sig.inputs.iter().map(|fn_arg| {
        if let syn::FnArg::Typed(pat_type) = fn_arg {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                pat_ident
            } else {
                panic!("unreachable");
            }
        } else {
            panic!("unreachable");
        }
    }).collect();

    let get_fn_name = syn::Ident::new(&format!("build_{}", fn_name), fn_name.span());
    let get_prover_verifier_fn = quote! {
        #[cfg(not(feature = "guest"))]
        pub fn #get_fn_name() -> (
            impl Fn(#(#input_types),*) -> #prove_output_ty,
            impl Fn(jolt_sdk::Proof) -> bool
        ) {

            let (program, preprocessing) = #preprocess_fn_name();
            let program = std::rc::Rc::new(program);
            let preprocessing = std::rc::Rc::new(preprocessing);

            let program_cp = program.clone();
            let preprocessing_cp = preprocessing.clone();

            let prove_closure = move |#inputs| {
                let program = (*program).clone();
                let preprocessing = (*preprocessing).clone();
                #prove_fn_name(program, preprocessing, #(#input_names),*)
            };


            let verify_closure = move |proof: jolt_sdk::Proof| {
                let program = (*program_cp).clone();
                let preprocessing = (*preprocessing_cp).clone();
                RV32IJoltVM::verify(preprocessing, proof.proof, proof.commitments).is_ok()
            };

            (prove_closure, verify_closure)
        }
    };


    let execute_fn_name = syn::Ident::new(&format!("execute_{}", fn_name), fn_name.span());
    let execute_fn = quote! {
        #[cfg(not(feature = "guest"))]
        pub fn #execute_fn_name(#inputs) #output {
            #body
        }
    };

    let output = quote! {
        #imports
        #transformed_fn
        #preprocess_fn
        #prove_fn
        #get_prover_verifier_fn
        #execute_fn
    };

    output.into()
}
