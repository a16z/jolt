#![feature(proc_macro_tracked_env)]

extern crate proc_macro;

use core::panic;
use std::collections::HashMap;

use common::{
    constants::{
        DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE, DEFAULT_MEMORY_SIZE, DEFAULT_STACK_SIZE,
    },
    rv_trace::MemoryLayout,
};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse_macro_input, AttributeArgs, Ident, ItemFn, Lit, Meta, MetaNameValue, NestedMeta, PatType,
    ReturnType, Type,
};

#[proc_macro_attribute]
pub fn provable(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as AttributeArgs);
    let func = parse_macro_input!(item as ItemFn);
    MacroBuilder::new(attr, func).build()
}

struct MacroBuilder {
    attr: AttributeArgs,
    func: ItemFn,
    std: bool,
    func_args: Vec<(Ident, Box<Type>)>,
}

impl MacroBuilder {
    fn new(attr: AttributeArgs, func: ItemFn) -> Self {
        let func_args = Self::get_func_args(&func);
        #[cfg(feature = "guest-std")]
        let std = true;
        #[cfg(not(feature = "guest-std"))]
        let std = false;

        Self {
            attr,
            func,
            std,
            func_args,
        }
    }

    fn build(&self) -> TokenStream {
        let build_fn = self.make_build_fn();
        let execute_fn = self.make_execute_function();
        let analyze_fn = self.make_analyze_function();
        let preprocess_fn = self.make_preprocess_func();
        let prove_fn = self.make_prove_func();

        let main_fn = if let Some(func) = self.get_func_selector() {
            if *self.get_func_name() == func {
                self.make_main_func()
            } else {
                quote! {}
            }
        } else {
            self.make_main_func()
        };

        quote! {
            #build_fn
            #execute_fn
            #analyze_fn
            #preprocess_fn
            #prove_fn
            #main_fn
        }
        .into()
    }

    fn make_build_fn(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let build_fn_name = Ident::new(&format!("build_{}", fn_name), fn_name.span());
        let prove_output_ty = self.get_prove_output_type();

        let input_names = self.func_args.iter().map(|(name, _)| name);
        let input_types = self.func_args.iter().map(|(_, ty)| ty);
        let inputs = &self.func.sig.inputs;
        let preprocess_fn_name = Ident::new(&format!("preprocess_{}", fn_name), fn_name.span());
        let prove_fn_name = Ident::new(&format!("prove_{}", fn_name), fn_name.span());
        let imports = self.make_imports();

        quote! {
            #[cfg(not(feature = "guest"))]
            pub fn #build_fn_name() -> (
                impl Fn(#(#input_types),*) -> #prove_output_ty,
                impl Fn(jolt::Proof) -> bool
            ) {
                #imports
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


                let verify_closure = move |proof: jolt::Proof| {
                    let program = (*program_cp).clone();
                    let preprocessing = (*preprocessing_cp).clone();
                    RV32IJoltVM::verify(preprocessing, proof.proof, proof.commitments).is_ok()
                };

                (prove_closure, verify_closure)
            }
        }
    }

    fn make_execute_function(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let inputs = &self.func.sig.inputs;
        let output = &self.func.sig.output;
        let body = &self.func.block;

        quote! {
             pub fn #fn_name(#inputs) #output {
                 #body
             }
        }
    }

    fn make_analyze_function(&self) -> TokenStream2 {
        let set_mem_size = self.make_set_linker_parameters();
        let guest_name = self.get_guest_name();
        let imports = self.make_imports();
        let set_std = self.make_set_std();

        let fn_name = self.get_func_name();
        let fn_name_str = fn_name.to_string();
        let analyze_fn_name = Ident::new(&format!("analyze_{}", fn_name), fn_name.span());
        let inputs = &self.func.sig.inputs;
        let set_program_args = self.func_args.iter().map(|(name, _)| {
            quote! {
                program.set_input(&#name);
            }
        });

        quote! {
             #[cfg(not(feature = "guest"))]
             pub fn #analyze_fn_name(#inputs) -> jolt::host::analyze::ProgramSummary {
                #imports

                let mut program = Program::new(#guest_name);
                program.set_func(#fn_name_str);
                #set_std
                #set_mem_size
                #(#set_program_args;)*

                program.trace_analyze::<jolt::F>()
             }
        }
    }

    fn make_preprocess_func(&self) -> TokenStream2 {
        let set_mem_size = self.make_set_linker_parameters();
        let guest_name = self.get_guest_name();
        let imports = self.make_imports();
        let set_std = self.make_set_std();

        let fn_name = self.get_func_name();
        let fn_name_str = fn_name.to_string();
        let preprocess_fn_name = Ident::new(&format!("preprocess_{}", fn_name), fn_name.span());
        quote! {
            #[cfg(not(feature = "guest"))]
            pub fn #preprocess_fn_name() -> (
                jolt::host::Program,
                jolt::JoltPreprocessing<jolt::F, jolt::CommitmentScheme>
            ) {
                #imports

                let mut program = Program::new(#guest_name);
                program.set_func(#fn_name_str);
                #set_std
                #set_mem_size
                let (bytecode, memory_init) = program.decode();

                // TODO(moodlezoup): Feed in size parameters via macro
                let preprocessing: JoltPreprocessing<jolt::F, jolt::CommitmentScheme> =
                    RV32IJoltVM::preprocess(
                        bytecode,
                        memory_init,
                        1 << 20,
                        1 << 20,
                        1 << 24
                    );

                (program, preprocessing)
            }
        }
    }

    fn make_prove_func(&self) -> TokenStream2 {
        let prove_output_ty = self.get_prove_output_type();

        let handle_return = match &self.func.sig.output {
            ReturnType::Default => quote! {
                let ret_val = ();
            },
            ReturnType::Type(_, ty) => quote! {
                let ret_val = jolt::postcard::from_bytes::<#ty>(&output_bytes).unwrap();
            },
        };

        let set_program_args = self.func_args.iter().map(|(name, _)| {
            quote! {
                program.set_input(&#name);
            }
        });

        let fn_name = self.get_func_name();
        let inputs = &self.func.sig.inputs;
        let imports = self.make_imports();

        let prove_fn_name = syn::Ident::new(&format!("prove_{}", fn_name), fn_name.span());
        quote! {
            #[cfg(not(feature = "guest"))]
            pub fn #prove_fn_name(
                mut program: jolt::host::Program,
                preprocessing: jolt::JoltPreprocessing<jolt::F, jolt::CommitmentScheme>,
                #inputs
            ) -> #prove_output_ty {
                #imports

                #(#set_program_args;)*

                let (io_device, trace, circuit_flags) =
                    program.trace();

                let output_bytes = io_device.outputs.clone();

                let (jolt_proof, jolt_commitments) = RV32IJoltVM::prove(
                    io_device,
                    trace,
                    circuit_flags,
                    preprocessing,
                );

                #handle_return

                let proof = jolt::Proof {
                    proof: jolt_proof,
                    commitments: jolt_commitments,
                };

                (ret_val, proof)
            }
        }
    }

    fn make_main_func(&self) -> TokenStream2 {
        let attributes = self.parse_attributes();
        let memory_layout =
            MemoryLayout::new(attributes.max_input_size, attributes.max_output_size);
        let input_start = memory_layout.input_start;
        let output_start = memory_layout.output_start;
        let max_input_len = attributes.max_input_size as usize;
        let max_output_len = attributes.max_output_size as usize;

        let get_input_slice = quote! {
            let input_ptr = #input_start as *const u8;
            let input_slice = unsafe {
                core::slice::from_raw_parts(input_ptr, #max_input_len)
            };
        };

        let args = &self.func_args;
        let args_fetch = args.iter().map(|(name, ty)| {
            quote! {
                let (#name, input_slice) =
                    jolt::postcard::take_from_bytes::<#ty>(input_slice).unwrap();
            }
        });

        // TODO: ensure that input slice hasn't overflown
        let check_input_len = quote! {};

        let block = &self.func.block;
        let block = quote! {let to_return = (|| -> _ { #block })();};

        let handle_return = match &self.func.sig.output {
            ReturnType::Default => quote! {},
            ReturnType::Type(_, ty) => quote! {
                let output_ptr = #output_start as *mut u8;
                let output_slice = unsafe {
                    core::slice::from_raw_parts_mut(output_ptr, #max_output_len)
                };

                jolt::postcard::to_slice::<#ty>(&to_return, output_slice).unwrap();
            },
        };

        let panic_fn = self.make_panic(memory_layout.panic);
        let declare_alloc = self.make_allocator();

        quote! {
            #[cfg(feature = "guest")]
            use core::arch::global_asm;

            #[cfg(feature = "guest")]
            global_asm!("\
                .global _start\n\
                .extern _STACK_PTR\n\
                .section .text.boot\n\
                _start:	la sp, _STACK_PTR\n\
                    jal main\n\
                    j .\n\
            ");

            #declare_alloc

            #[cfg(feature = "guest")]
            #[no_mangle]
            pub extern "C" fn main() {
                let mut offset = 0;
                #get_input_slice
                #(#args_fetch;)*
                #check_input_len
                #block
                #handle_return
            }

            #panic_fn
        }
    }

    fn make_panic(&self, panic_address: u64) -> TokenStream2 {
        if self.std {
            quote! {
                #[cfg(feature = "guest")]
                #[no_mangle]
                pub extern "C" fn jolt_panic() {
                    unsafe {
                        core::ptr::write_volatile(#panic_address as *mut u8, 1);
                    }

                    loop {}
                }
            }
        } else {
            quote! {
                #[cfg(feature = "guest")]
                use core::panic::PanicInfo;

                #[cfg(feature = "guest")]
                #[panic_handler]
                fn panic(_info: &PanicInfo) -> ! {
                    unsafe {
                        core::ptr::write_volatile(#panic_address as *mut u8, 1);
                    }

                    loop {}
                }
            }
        }
    }

    fn make_allocator(&self) -> TokenStream2 {
        if self.std {
            quote! {}
        } else {
            quote! {
                #[cfg(feature = "guest")]
                #[global_allocator]
                static ALLOCATOR: jolt::BumpAllocator = jolt::BumpAllocator;
            }
        }
    }

    fn make_imports(&self) -> TokenStream2 {
        quote! {
            #[cfg(not(feature = "guest"))]
            use jolt::{
                JoltField,
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
                tracer,
            };
        }
    }

    fn make_set_linker_parameters(&self) -> TokenStream2 {
        let attributes = self.parse_attributes();
        let mut code: Vec<TokenStream2> = Vec::new();

        let value = attributes.memory_size;
        code.push(quote! {
            program.set_memory_size(#value);
        });

        let value = attributes.stack_size;
        code.push(quote! {
            program.set_stack_size(#value);
        });

        let value = attributes.max_input_size;
        code.push(quote! {
            program.set_max_input_size(#value);
        });

        let value = attributes.max_output_size;
        code.push(quote! {
            program.set_max_output_size(#value);
        });

        quote! {
            #(#code;)*
        }
    }

    fn make_set_std(&self) -> TokenStream2 {
        if self.std {
            quote! {
                program.set_std(true);
            }
        } else {
            quote! {
                program.set_std(false);
            }
        }
    }

    fn parse_attributes(&self) -> Attributes {
        let mut attributes = HashMap::<_, u64>::new();
        for attr in &self.attr {
            match attr {
                NestedMeta::Meta(Meta::NameValue(MetaNameValue { path, lit, .. })) => {
                    let value: u64 = match lit {
                        Lit::Int(lit) => lit.base10_parse().unwrap(),
                        _ => panic!("expected integer literal"),
                    };
                    let ident = &path.get_ident().expect("Expected identifier");
                    match ident.to_string().as_str() {
                        "memory_size" => attributes.insert("memory_size", value),
                        "stack_size" => attributes.insert("stack_size", value),
                        "max_input_size" => attributes.insert("max_input_size", value),
                        "max_output_size" => attributes.insert("max_output_size", value),
                        _ => panic!("invalid attribute"),
                    };
                }
                _ => panic!("expected integer literal"),
            }
        }

        let memory_size = *attributes
            .get("memory_size")
            .unwrap_or(&DEFAULT_MEMORY_SIZE);
        let stack_size = *attributes.get("stack_size").unwrap_or(&DEFAULT_STACK_SIZE);
        let max_input_size = *attributes
            .get("max_input_size")
            .unwrap_or(&DEFAULT_MAX_INPUT_SIZE);
        let max_output_size = *attributes
            .get("max_output_size")
            .unwrap_or(&DEFAULT_MAX_OUTPUT_SIZE);

        Attributes {
            memory_size,
            stack_size,
            max_input_size,
            max_output_size,
        }
    }

    fn get_prove_output_type(&self) -> TokenStream2 {
        match &self.func.sig.output {
            ReturnType::Default => quote! {
                ((), jolt::Proof)
            },
            ReturnType::Type(_, ty) => quote! {
                (#ty, jolt::Proof)
            },
        }
    }

    fn get_func_args(func: &ItemFn) -> Vec<(Ident, Box<Type>)> {
        let mut args = Vec::new();
        for arg in &func.sig.inputs {
            if let syn::FnArg::Typed(PatType { pat, ty, .. }) = arg {
                if let syn::Pat::Ident(pat_ident) = pat.as_ref() {
                    args.push((pat_ident.ident.clone(), ty.clone()));
                } else {
                    panic!("cannot parse arg");
                }
            } else {
                panic!("cannot parse arg");
            }
        }

        args
    }

    fn get_func_name(&self) -> &Ident {
        &self.func.sig.ident
    }

    fn get_guest_name(&self) -> String {
        proc_macro::tracked_env::var("CARGO_PKG_NAME").unwrap()
    }

    fn get_func_selector(&self) -> Option<String> {
        proc_macro::tracked_env::var("JOLT_FUNC_NAME").ok()
    }
}

struct Attributes {
    memory_size: u64,
    stack_size: u64,
    max_input_size: u64,
    max_output_size: u64,
}
