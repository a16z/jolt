extern crate proc_macro;

use core::panic;

use common::{
    attributes::parse_attributes,
    jolt_device::{MemoryConfig, MemoryLayout},
};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse_macro_input, punctuated::Punctuated, token::Comma, Ident, ItemFn, Meta, PatType,
    ReturnType, Token, Type,
};

#[proc_macro_attribute]
pub fn provable(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr with Punctuated::<Meta, Token![,]>::parse_terminated);
    let func = parse_macro_input!(item as ItemFn);
    let mut builder = MacroBuilder::new(attr, func);

    let mut token_stream = builder.build();

    // Add wasm utilities and functions if the function is marked as wasm
    if builder.has_wasm_attr() {
        let wasm_token_stream: TokenStream = builder.make_wasm_function().into();
        token_stream.extend(wasm_token_stream);
    }

    token_stream
}

struct MacroBuilder {
    attr: Punctuated<Meta, Comma>,
    func: ItemFn,
    std: bool,
    pub_func_args: Vec<(Ident, Box<Type>)>,
    trusted_func_args: Vec<(Ident, Box<Type>)>,
    untrusted_func_args: Vec<(Ident, Box<Type>)>,
    has_private_input: bool,
}

impl MacroBuilder {
    fn new(attr: Punctuated<Meta, Comma>, func: ItemFn) -> Self {
        let (pub_func_args, trusted_func_args, untrusted_func_args) = Self::get_func_args(&func);
        let has_private_input = Self::any_arg_is_private_input(&func);
        #[cfg(feature = "guest-std")]
        let std = true;
        #[cfg(not(feature = "guest-std"))]
        let std = false;

        Self {
            attr,
            func,
            std,
            pub_func_args,
            trusted_func_args,
            untrusted_func_args,
            has_private_input,
        }
    }

    fn build(&mut self) -> TokenStream {
        let memory_config_fn = self.make_memory_config_fn();
        let build_prover_fn = self.make_build_prover_fn();
        let build_verifier_fn = self.make_build_verifier_fn();
        let analyze_fn = self.make_analyze_function();
        let trace_fn = self.make_trace_func();
        let trace_to_file_fn = self.make_trace_to_file_func();
        let compile_fn = self.make_compile_func();
        let preprocess_shared_fn = self.make_preprocess_shared_func();
        let preprocess_shared_committed_fn = self.make_preprocess_shared_committed_func();
        let preprocess_prover_fn = self.make_preprocess_prover_func();
        let preprocess_committed_prover_fn = self.make_preprocess_committed_prover_func();
        let preprocess_verifier_fn = self.make_preprocess_verifier_func();
        let verifier_preprocess_from_prover_fn = self.make_preprocess_from_prover_func();
        let commit_trusted_advice_fn = self.make_commit_trusted_advice_func();
        let prove_fn = self.make_prove_func();

        let attributes = parse_attributes(&self.attr);
        let mut execute_fn = quote! {};
        if !attributes.guest_only {
            execute_fn = self.make_execute_function();
        }

        let main_fn = if let Some(func) = self.get_func_selector() {
            if *self.get_func_name() == func {
                self.make_main_func()
            } else {
                quote! {}
            }
        } else {
            self.make_main_func()
        };

        let require_zk = self.make_require_zk_check();

        quote! {
            #require_zk
            #memory_config_fn
            #build_prover_fn
            #build_verifier_fn
            #execute_fn
            #analyze_fn
            #trace_fn
            #trace_to_file_fn
            #compile_fn
            #preprocess_shared_fn
            #preprocess_shared_committed_fn
            #preprocess_prover_fn
            #preprocess_committed_prover_fn
            #preprocess_verifier_fn
            #verifier_preprocess_from_prover_fn
            #commit_trusted_advice_fn
            #prove_fn
            #main_fn
        }
        .into()
    }

    fn make_memory_config_fn(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let attributes = parse_attributes(&self.attr);
        let max_input_size = proc_macro2::Literal::u64_unsuffixed(attributes.max_input_size);
        let max_output_size = proc_macro2::Literal::u64_unsuffixed(attributes.max_output_size);
        let max_trusted_advice_size =
            proc_macro2::Literal::u64_unsuffixed(attributes.max_trusted_advice_size);
        let max_untrusted_advice_size =
            proc_macro2::Literal::u64_unsuffixed(attributes.max_untrusted_advice_size);
        let stack_size = proc_macro2::Literal::u64_unsuffixed(attributes.stack_size);
        let heap_size = proc_macro2::Literal::u64_unsuffixed(attributes.heap_size);

        let memory_config_fn_name = Ident::new(&format!("memory_config_{fn_name}"), fn_name.span());
        let imports = self.make_imports();

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #memory_config_fn_name() -> jolt::MemoryConfig {
                #imports
                MemoryConfig {
                    max_input_size: #max_input_size,
                    max_output_size: #max_output_size,
                    max_trusted_advice_size: #max_trusted_advice_size,
                    max_untrusted_advice_size: #max_untrusted_advice_size,
                    stack_size: #stack_size,
                    heap_size: #heap_size,
                    program_size: None,
                }
            }
        }
    }

    fn make_build_prover_fn(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let build_prover_fn_name = Ident::new(&format!("build_prover_{fn_name}"), fn_name.span());
        let prove_output_ty = self.get_prove_output_type();

        // Include public, trusted_advice, and untrusted_advice arguments for the prover
        let ordered_func_args = self.get_all_func_args_in_order();
        let all_names: Vec<_> = ordered_func_args.iter().map(|(name, _)| name).collect();
        let all_types: Vec<_> = ordered_func_args.iter().map(|(_, ty)| ty).collect();

        let inputs_vec: Vec<_> = self.func.sig.inputs.iter().collect();
        let inputs = quote! { #(#inputs_vec),* };
        let prove_fn_name = Ident::new(&format!("prove_{fn_name}"), fn_name.span());
        let imports = self.make_imports();

        let has_trusted_advice = !self.trusted_func_args.is_empty();

        let commitment_param_in_closure = if has_trusted_advice {
            quote! { , trusted_advice_commitment: Option<jolt::VerifierTrustedAdviceCommitment>,
            trusted_advice_hint: Option<jolt::TrustedAdviceOpeningHint> }
        } else {
            quote! {}
        };

        let commitment_arg_in_call = if has_trusted_advice {
            quote! { , trusted_advice_commitment, trusted_advice_hint }
        } else {
            quote! {}
        };

        let return_type = if has_trusted_advice {
            quote! {
                impl Fn(#(#all_types),*, Option<jolt::VerifierTrustedAdviceCommitment>, Option<jolt::TrustedAdviceOpeningHint>) -> #prove_output_ty + Sync + Send
            }
        } else {
            quote! {
                impl Fn(#(#all_types),*) -> #prove_output_ty + Sync + Send
            }
        };

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #build_prover_fn_name<S: jolt::host::JoltProgramSource + Send + Sync + 'static>(
                program: S,
                preprocessing: jolt::JoltProverPreprocessing,
            ) -> #return_type
            {
                #imports
                let program = std::sync::Arc::new(program);
                let preprocessing = std::sync::Arc::new(preprocessing);

                let prove_closure = move |#inputs #commitment_param_in_closure| {
                    let preprocessing = (*preprocessing).clone();
                    #prove_fn_name(program.as_ref(), preprocessing, #(#all_names),* #commitment_arg_in_call)
                };

                prove_closure
            }
        }
    }

    fn make_build_verifier_fn(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let build_verifier_fn_name =
            Ident::new(&format!("build_verifier_{fn_name}"), fn_name.span());

        let input_types = self.pub_func_args.iter().map(|(_, ty)| ty);
        let output_type: Type = match &self.func.sig.output {
            ReturnType::Default => syn::parse_quote!(()),
            ReturnType::Type(_, ty) => syn::parse_quote!((#ty)),
        };
        let public_inputs = self.pub_func_args.iter().map(|(name, ty)| {
            quote! { #name: #ty }
        });
        let imports = self.make_imports();
        let set_program_args = self.pub_func_args.iter().map(|(name, _)| {
            quote! {
                io_device.inputs.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });

        let has_trusted_advice = !self.trusted_func_args.is_empty();

        let commitment_param_in_signature = if has_trusted_advice {
            quote! { Option<jolt::VerifierTrustedAdviceCommitment>, }
        } else {
            quote! {}
        };

        let commitment_param_in_closure = if has_trusted_advice {
            quote! { trusted_advice_commitment: Option<jolt::VerifierTrustedAdviceCommitment>, }
        } else {
            quote! {}
        };

        let commitment_arg_in_verify = if has_trusted_advice {
            quote! { trusted_advice_commitment.as_ref() }
        } else {
            quote! { None }
        };

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #build_verifier_fn_name(
                preprocessing: jolt::JoltVerifierPreprocessing,
            ) -> impl Fn(#(#input_types ,)* #output_type, bool, #commitment_param_in_signature jolt::RV64IMACProof) -> bool + Sync + Send
            {
                #imports
                let preprocessing = std::sync::Arc::new(preprocessing);

                let verify_closure = move |#(#public_inputs,)* output, panic, #commitment_param_in_closure proof: jolt::RV64IMACProof| {
                    let preprocessing = (*preprocessing).clone();
                    let memory_layout = preprocessing.program.memory_layout();
                    let memory_config = MemoryConfig {
                        max_input_size: memory_layout.max_input_size,
                        max_output_size: memory_layout.max_output_size,
                        max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
                        max_trusted_advice_size: memory_layout.max_trusted_advice_size,
                        stack_size: memory_layout.stack_size,
                        heap_size: memory_layout.heap_size,
                        program_size: Some(memory_layout.program_size),
                    };
                    let mut io_device = JoltDevice::new(&memory_config);

                    #(#set_program_args;)*
                    io_device.outputs.append(&mut jolt::postcard::to_stdvec(&output).unwrap());
                    io_device.panic = panic;

                    jolt::jolt_verifier::verify::<
                        jolt::VerifierField,
                        jolt::VerifierPCS,
                        jolt::VerifierVC,
                        jolt::VerifierTranscript,
                    >(
                        &preprocessing,
                        &io_device,
                        &proof,
                        #commitment_arg_in_verify,
                    ).is_ok()
                };

                verify_closure
            }
        }
    }

    fn make_execute_function(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let inputs = &self.func.sig.inputs;
        let output = &self.func.sig.output;
        let body = &self.func.block;
        let attrs = &self.func.attrs;

        quote! {
            #[cfg(not(target_arch = "wasm32"))]
            #(#attrs)*
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
        let set_backtrace = self.make_set_backtrace();
        let set_profile = self.make_set_profile();
        let enable_field_inline = self.make_enable_field_inline();

        let fn_name = self.get_func_name();
        let fn_name_str = fn_name.to_string();
        let analyze_fn_name = Ident::new(&format!("analyze_{fn_name}"), fn_name.span());
        let inputs = &self.func.sig.inputs;
        let set_pub_args = self.pub_func_args.iter().map(|(name, _)| {
            quote! {
                input_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_untrusted_advice_args = self.untrusted_func_args.iter().map(|(name, _)| {
            quote! {
                untrusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_trusted_advice_args = self.trusted_func_args.iter().map(|(name, _)| {
            quote! {
                trusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });

        quote! {
             #[cfg(not(target_arch = "wasm32"))]
             #[cfg(not(feature = "guest"))]
             pub fn #analyze_fn_name(#inputs) -> jolt::host::analyze::ProgramSummary {
                #imports

                let mut program = Program::new(#guest_name);
                program.set_func(#fn_name_str);
                #set_std
                #set_profile
                #set_backtrace
                #enable_field_inline
                #set_mem_size

                let mut input_bytes = vec![];
                #(#set_pub_args;)*
                let mut untrusted_advice_bytes = vec![];
                #(#set_untrusted_advice_args;)*
                let mut trusted_advice_bytes = vec![];
                #(#set_trusted_advice_args;)*

                program.trace_analyze(&input_bytes, &untrusted_advice_bytes, &trusted_advice_bytes)
             }
        }
    }

    fn make_trace_func(&self) -> TokenStream2 {
        let imports = self.make_imports();
        let guest_name = self.get_guest_name();
        let set_mem_size = self.make_set_linker_parameters();
        let set_std = self.make_set_std();
        let set_backtrace = self.make_set_backtrace();
        let set_profile = self.make_set_profile();
        let enable_field_inline = self.make_enable_field_inline();

        let fn_name = self.get_func_name();
        let fn_name_str = fn_name.to_string();
        let trace_fn_name = Ident::new(&format!("trace_{fn_name}"), fn_name.span());
        let trace_with_backend_fn_name =
            Ident::new(&format!("trace_{fn_name}_with_backend"), fn_name.span());
        let inputs_vec: Vec<_> = self.func.sig.inputs.iter().collect();
        let inputs = quote! { #(#inputs_vec),* };
        let ordered_func_args = self.get_all_func_args_in_order();
        let all_names: Vec<_> = ordered_func_args.iter().map(|(name, _)| name).collect();
        let set_pub_args = self.pub_func_args.iter().map(|(name, _)| {
            quote! {
                input_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_untrusted_advice_args = self.untrusted_func_args.iter().map(|(name, _)| {
            quote! {
                untrusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_trusted_advice_args = self.trusted_func_args.iter().map(|(name, _)| {
            quote! {
                trusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #trace_fn_name(#inputs) -> Result<jolt::TraceOutput<jolt::OwnedTrace>, jolt::TraceError> {
                #imports

                let mut backend = jolt::TracerBackend::new();
                #trace_with_backend_fn_name(&mut backend, #(#all_names),*)
            }

            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #trace_with_backend_fn_name<B: jolt::ExecutionBackend>(
                backend: &mut B,
                #inputs
            ) -> Result<jolt::TraceOutput<B::Trace>, jolt::TraceError> {
                #imports

                let mut program = Program::new(#guest_name);
                program.set_func(#fn_name_str);
                #set_std
                #set_profile
                #set_backtrace
                #enable_field_inline
                #set_mem_size

                let mut input_bytes = vec![];
                #(#set_pub_args;)*
                let mut untrusted_advice_bytes = vec![];
                #(#set_untrusted_advice_args;)*
                let mut trusted_advice_bytes = vec![];
                #(#set_trusted_advice_args;)*

                program.trace_with_backend(
                    backend,
                    &input_bytes,
                    &untrusted_advice_bytes,
                    &trusted_advice_bytes,
                )
            }
        }
    }

    fn make_trace_to_file_func(&self) -> TokenStream2 {
        let imports = self.make_imports();
        let guest_name = self.get_guest_name();
        let set_mem_size = self.make_set_linker_parameters();
        let set_std = self.make_set_std();
        let set_backtrace = self.make_set_backtrace();
        let set_profile = self.make_set_profile();
        let enable_field_inline = self.make_enable_field_inline();

        let fn_name = self.get_func_name();
        let fn_name_str = fn_name.to_string();
        let trace_to_file_fn_name = Ident::new(&format!("trace_{fn_name}_to_file"), fn_name.span());
        let inputs_vec: Vec<_> = self.func.sig.inputs.iter().collect();
        let inputs = quote! { #(#inputs_vec),* };
        let set_pub_args = self.pub_func_args.iter().map(|(name, _)| {
            quote! {
                input_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_untrusted_advice_args = self.untrusted_func_args.iter().map(|(name, _)| {
            quote! {
                untrusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_trusted_advice_args = self.trusted_func_args.iter().map(|(name, _)| {
            quote! {
                trusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #trace_to_file_fn_name(target_dir: &str, #inputs) {
                #imports

                let mut program = Program::new(#guest_name);
                let path = std::path::PathBuf::from(target_dir);
                program.set_func(#fn_name_str);
                #set_std
                #set_profile
                #set_backtrace
                #enable_field_inline
                #set_mem_size

                let mut input_bytes = vec![];
                #(#set_pub_args;)*
                let mut untrusted_advice_bytes = vec![];
                #(#set_untrusted_advice_args;)*
                let mut trusted_advice_bytes = vec![];
                #(#set_trusted_advice_args;)*

                program.trace_to_file(&input_bytes, &untrusted_advice_bytes, &trusted_advice_bytes, &path);
            }
        }
    }

    fn make_compile_func(&self) -> TokenStream2 {
        let imports = self.make_imports();
        let guest_name = self.get_guest_name();
        let set_mem_size = self.make_set_linker_parameters();
        let set_std = self.make_set_std();
        let set_backtrace = self.make_set_backtrace();
        let set_profile = self.make_set_profile();
        let enable_field_inline = self.make_enable_field_inline();

        let fn_name = self.get_func_name();
        let fn_name_str = fn_name.to_string();
        let compile_fn_name = Ident::new(&format!("compile_{fn_name}"), fn_name.span());
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #compile_fn_name(target_dir: &str) -> jolt::host::Program {
                #imports

                let mut program = Program::new(#guest_name);
                program.set_func(#fn_name_str);
                #set_std
                #set_profile
                #set_backtrace
                #enable_field_inline
                #set_mem_size

                // Build the compute_advice version first
                program.build_with_features(target_dir, &["compute_advice"]);

                // Build the normal version (without compute_advice)
                program.build_with_features(target_dir, &[]);

                program
            }
        }
    }

    fn make_preprocess_shared_func(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
        let max_trace_length = proc_macro2::Literal::u64_unsuffixed(attributes.max_trace_length);
        let max_input_size = proc_macro2::Literal::u64_unsuffixed(attributes.max_input_size);
        let max_output_size = proc_macro2::Literal::u64_unsuffixed(attributes.max_output_size);
        let max_untrusted_advice_size =
            proc_macro2::Literal::u64_unsuffixed(attributes.max_untrusted_advice_size);
        let max_trusted_advice_size =
            proc_macro2::Literal::u64_unsuffixed(attributes.max_trusted_advice_size);
        let stack_size = proc_macro2::Literal::u64_unsuffixed(attributes.stack_size);
        let heap_size = proc_macro2::Literal::u64_unsuffixed(attributes.heap_size);
        let imports = self.make_imports();

        let fn_name = self.get_func_name();
        let preprocess_shared_fn_name =
            Ident::new(&format!("preprocess_shared_{fn_name}"), fn_name.span());
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #preprocess_shared_fn_name(program: &mut dyn jolt::host::JoltProgramSource)
                -> Result<jolt::JoltSharedPreprocessing, jolt::PreprocessingError>
            {
                #imports

                let (bytecode, memory_init, program_size, e_entry) = program.decode();
                let memory_config = MemoryConfig {
                    max_input_size: #max_input_size,
                    max_output_size: #max_output_size,
                    max_untrusted_advice_size: #max_untrusted_advice_size,
                    max_trusted_advice_size: #max_trusted_advice_size,
                    stack_size: #stack_size,
                    heap_size: #heap_size,
                    program_size: Some(program_size),
                };
                let memory_layout = MemoryLayout::new(&memory_config);

                let program_data = jolt::JoltProgramPreprocessing::new(
                    bytecode,
                    memory_init,
                    memory_layout,
                    e_entry,
                    #max_trace_length,
                    program.instruction_profile(),
                )?;
                JoltSharedPreprocessing::new(program_data)
            }
        }
    }

    fn make_preprocess_shared_committed_func(&self) -> TokenStream2 {
        let imports = self.make_imports();
        let attributes = parse_attributes(&self.attr);
        let max_input_size = proc_macro2::Literal::u64_unsuffixed(attributes.max_input_size);
        let max_output_size = proc_macro2::Literal::u64_unsuffixed(attributes.max_output_size);
        let max_untrusted_advice_size =
            proc_macro2::Literal::u64_unsuffixed(attributes.max_untrusted_advice_size);
        let max_trusted_advice_size =
            proc_macro2::Literal::u64_unsuffixed(attributes.max_trusted_advice_size);
        let stack_size = proc_macro2::Literal::u64_unsuffixed(attributes.stack_size);
        let heap_size = proc_macro2::Literal::u64_unsuffixed(attributes.heap_size);
        let max_trace_length = proc_macro2::Literal::u64_unsuffixed(attributes.max_trace_length);

        let fn_name = self.get_func_name();
        let preprocess_shared_committed_fn_name = Ident::new(
            &format!("preprocess_shared_committed_{fn_name}"),
            fn_name.span(),
        );
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #preprocess_shared_committed_fn_name(
                program: &mut dyn jolt::host::JoltProgramSource,
                bytecode_chunk_count: usize,
            ) -> Result<jolt::JoltProverPreprocessing, jolt::PreprocessingError>
            {
                #imports

                let (bytecode, memory_init, program_size, e_entry) = program.decode();
                let memory_config = MemoryConfig {
                    max_input_size: #max_input_size,
                    max_output_size: #max_output_size,
                    max_untrusted_advice_size: #max_untrusted_advice_size,
                    max_trusted_advice_size: #max_trusted_advice_size,
                    stack_size: #stack_size,
                    heap_size: #heap_size,
                    program_size: Some(program_size),
                };
                let memory_layout = MemoryLayout::new(&memory_config);

                let program_data = jolt::JoltProgramPreprocessing::new(
                    bytecode,
                    memory_init,
                    memory_layout,
                    e_entry,
                    #max_trace_length,
                    program.instruction_profile(),
                )?;
                jolt::jolt_prover::dory::preprocess_committed(
                    program_data,
                    bytecode_chunk_count,
                )
            }
        }
    }

    fn make_preprocess_prover_func(&self) -> TokenStream2 {
        let imports = self.make_imports();

        let fn_name = self.get_func_name();
        let preprocess_prover_fn_name =
            Ident::new(&format!("preprocess_prover_{fn_name}"), fn_name.span());
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #preprocess_prover_fn_name(
                shared_preprocessing: jolt::JoltSharedPreprocessing
            )
                -> jolt::JoltProverPreprocessing
            {
                #imports
                jolt::jolt_prover::dory::from_shared(shared_preprocessing)
            }
        }
    }

    fn make_preprocess_committed_prover_func(&self) -> TokenStream2 {
        let imports = self.make_imports();

        let fn_name = self.get_func_name();
        let preprocess_committed_fn_name =
            Ident::new(&format!("preprocess_committed_{fn_name}"), fn_name.span());
        let preprocess_shared_committed_fn_name = Ident::new(
            &format!("preprocess_shared_committed_{fn_name}"),
            fn_name.span(),
        );
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #preprocess_committed_fn_name(
                program: &mut jolt::host::Program,
                bytecode_chunk_count: usize,
            )
                -> Result<
                    jolt::JoltProverPreprocessing,
                    jolt::PreprocessingError,
                >
            {
                #imports
                #preprocess_shared_committed_fn_name(program, bytecode_chunk_count)
            }
        }
    }

    fn make_preprocess_verifier_func(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let preprocess_verifier_fn_name =
            Ident::new(&format!("preprocess_verifier_{fn_name}"), fn_name.span());

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #preprocess_verifier_fn_name(
                shared_preprocess: jolt::JoltSharedPreprocessing,
                generators: <jolt::PCS as jolt::CommitmentScheme>::VerifierSetup,
                blindfold_setup: Option<jolt::BlindfoldSetup>,
            ) -> jolt::JoltVerifierPreprocessing
            {
                jolt::jolt_prover::dory::from_shared_parts(
                    &shared_preprocess,
                    generators,
                    blindfold_setup,
                )
            }
        }
    }

    fn make_preprocess_from_prover_func(&self) -> TokenStream2 {
        let imports = self.make_imports();

        let fn_name = self.get_func_name();
        let preprocess_verifier_fn_name = Ident::new(
            &format!("verifier_preprocessing_from_prover_{fn_name}"),
            fn_name.span(),
        );
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #preprocess_verifier_fn_name(prover_preprocessing: &jolt::JoltProverPreprocessing)
                -> jolt::JoltVerifierPreprocessing
            {
                #imports
                prover_preprocessing.verifier_preprocessing()
            }
        }
    }

    fn make_commit_trusted_advice_func(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let commit_fn_name =
            Ident::new(&format!("commit_trusted_advice_{fn_name}"), fn_name.span());
        let imports = self.make_imports();

        // If there are no trusted advice arguments, return None values
        if self.trusted_func_args.is_empty() {
            return quote! {
                #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
                pub fn #commit_fn_name(
                    _preprocessing: &jolt::JoltProverPreprocessing,
                ) -> (Option<jolt::VerifierTrustedAdviceCommitment>,
                      Option<jolt::TrustedAdviceOpeningHint>)
                {
                    (None, None)
                }
            };
        }

        let trusted_advice_inputs = self.trusted_func_args.iter().map(|(name, ty)| {
            quote! { #name: #ty }
        });

        let set_trusted_advice_args = self.trusted_func_args.iter().map(|(name, _)| {
            quote! {
                trusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #commit_fn_name(
                #(#trusted_advice_inputs,)*
                preprocessing: &jolt::JoltProverPreprocessing,
            ) -> (Option<jolt::VerifierTrustedAdviceCommitment>,
                  Option<jolt::TrustedAdviceOpeningHint>)
            {
                #imports
                let mut trusted_advice_bytes = vec![];
                #(#set_trusted_advice_args;)*
                let committed = jolt::jolt_prover::dory::commit_trusted_advice(
                    preprocessing,
                    &trusted_advice_bytes,
                ).expect("trusted advice fits the configured memory layout");
                (Some(committed.commitment), Some(committed.hint))
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
                let mut outputs = io_device.outputs.clone();
                outputs.resize(
                    preprocessing.verifier.program.memory_layout().max_output_size as usize,
                    0,
                );
                let ret_val = jolt::postcard::from_bytes::<#ty>(&outputs).unwrap();
            },
        };

        let set_program_args = self.pub_func_args.iter().map(|(name, _)| {
            quote! {
                input_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_program_untrusted_advice_args = self.untrusted_func_args.iter().map(|(name, _)| {
            quote! {
                untrusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_program_trusted_advice_args = self.trusted_func_args.iter().map(|(name, _)| {
            quote! {
                trusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });

        let fn_name = self.get_func_name();
        let inputs_vec: Vec<_> = self.func.sig.inputs.iter().collect();
        let inputs = quote! { #(#inputs_vec),* };
        let imports = self.make_imports();

        let prove_fn_name = syn::Ident::new(&format!("prove_{fn_name}"), fn_name.span());

        let has_trusted_advice = !self.trusted_func_args.is_empty();

        let commitment_param = if has_trusted_advice {
            quote! { , trusted_advice_commitment: Option<jolt::VerifierTrustedAdviceCommitment>,
            trusted_advice_hint: Option<jolt::TrustedAdviceOpeningHint> }
        } else {
            quote! {}
        };

        let commitment_arg = if has_trusted_advice {
            quote! { trusted_advice_commitment, trusted_advice_hint }
        } else {
            quote! { None, None }
        };

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            #[allow(clippy::too_many_arguments)]
            pub fn #prove_fn_name(
                program: &dyn jolt::host::JoltProgramSource,
                preprocessing: jolt::JoltProverPreprocessing,
                #inputs
                #commitment_param
            ) -> #prove_output_ty {
                #imports

                let mut input_bytes = vec![];
                #(#set_program_args;)*
                let mut untrusted_advice_bytes = vec![];
                #(#set_program_untrusted_advice_args;)*
                let mut trusted_advice_bytes = vec![];
                #(#set_program_trusted_advice_args;)*

                let advice_tape = jolt::compute_advice_tape(
                    program,
                    &input_bytes,
                    &untrusted_advice_bytes,
                    &trusted_advice_bytes,
                    preprocessing.verifier.program.memory_layout(),
                ).expect("compute-advice execution should succeed");
                let (jolt_proof, io_device) = jolt::prove_program(
                    program,
                    &preprocessing,
                    &input_bytes,
                    &untrusted_advice_bytes,
                    &trusted_advice_bytes,
                    #commitment_arg,
                    advice_tape,
                ).expect("execution trace exceeds the max_trace_length configured in #[jolt::provable]");

                #handle_return

                (ret_val, jolt_proof, io_device)
            }
        }
    }

    fn make_main_func(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
        let memory_layout = MemoryLayout::new(&MemoryConfig {
            max_input_size: attributes.max_input_size,
            max_output_size: attributes.max_output_size,
            max_untrusted_advice_size: attributes.max_untrusted_advice_size,
            max_trusted_advice_size: attributes.max_trusted_advice_size,
            stack_size: attributes.stack_size,
            heap_size: attributes.heap_size,
            // Not needed for the main function, but we need the io region information from MemoryLayout.
            program_size: Some(0),
        });
        let input_start = memory_layout.input_start;
        let output_start = memory_layout.output_start;
        let untrusted_advice_start = memory_layout.untrusted_advice_start;
        let trusted_advice_start = memory_layout.trusted_advice_start;
        let max_input_len = attributes.max_input_size as usize;
        let max_output_len = attributes.max_output_size as usize;
        let max_untrusted_advice_len = attributes.max_untrusted_advice_size as usize;
        let max_trusted_advice_len = attributes.max_trusted_advice_size as usize;
        let termination_bit = memory_layout.termination as usize;

        let get_input_slice = quote! {
            let input_ptr = #input_start as *const u8;
            let input_slice = unsafe {
                core::slice::from_raw_parts(input_ptr, #max_input_len)
            };
        };

        let get_untrusted_advice_slice = quote! {
            let untrusted_advice_ptr = #untrusted_advice_start as *const u8;
            let untrusted_advice_slice = unsafe {
                core::slice::from_raw_parts(untrusted_advice_ptr, #max_untrusted_advice_len)
            };
        };

        let get_trusted_advice_slice = quote! {
            let trusted_advice_ptr = #trusted_advice_start as *const u8;
            let trusted_advice_slice = unsafe {
                core::slice::from_raw_parts(trusted_advice_ptr, #max_trusted_advice_len)
            };
        };

        let pub_args_fetch = self.pub_func_args.iter().map(|(name, ty)| {
            quote! {
                let (#name, input_slice) =
                    jolt::postcard::take_from_bytes::<#ty>(input_slice).unwrap();
            }
        });

        let untrusted_advice_args_fetch = self.untrusted_func_args.iter().map(|(name, ty)| {
            quote! {
                let (#name, untrusted_advice_slice) =
                    jolt::postcard::take_from_bytes::<#ty>(untrusted_advice_slice).unwrap();
            }
        });

        let trusted_advice_args_fetch = self.trusted_func_args.iter().map(|(name, ty)| {
            quote! {
                let (#name, trusted_advice_slice) =
                    jolt::postcard::take_from_bytes::<#ty>(trusted_advice_slice).unwrap();
            }
        });

        let check_input_len = quote! {};

        let attrs = &self.func.attrs;
        let output = &self.func.sig.output;
        let body = &self.func.block;
        let fn_name = self.get_func_name();
        let inner_fn_name = syn::Ident::new(&format!("__jolt_guest_{fn_name}"), fn_name.span());
        let inputs_vec: Vec<_> = self.func.sig.inputs.iter().collect();
        let inputs = quote! { #(#inputs_vec),* };
        let ordered_func_args = self.get_all_func_args_in_order();
        let all_names: Vec<_> = ordered_func_args.iter().map(|(name, _)| name).collect();
        let block = quote! {
            #(#attrs)*
            fn #inner_fn_name(#inputs) #output #body
            let to_return = #inner_fn_name(#(#all_names),*);
        };

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

        // Boot code (_start) is provided by jolt-sdk's boot modules via ZeroOS.
        // Both std and no-std modes go through __platform_bootstrap before main().
        let custom_start = quote! {};

        quote! {
            #custom_start

            #declare_alloc

            #[cfg(feature = "guest")]
            #[no_mangle]
            pub extern "C" fn main() -> ! {
                let mut offset = 0;
                #get_input_slice
                #get_untrusted_advice_slice
                #get_trusted_advice_slice
                #(#pub_args_fetch;)*
                #(#untrusted_advice_args_fetch;)*
                #(#trusted_advice_args_fetch;)*
                #check_input_len
                #block
                #handle_return
                unsafe {
                    core::ptr::write_volatile(#termination_bit as *mut u8, 1);
                }
                // Never return - loop forever for clean termination
                // The emulator detects termination via PC stall (prev_pc == pc)
                loop {
                    unsafe { core::arch::asm!("j .", options(noreturn)); }
                }
            }

            #panic_fn
        }
    }

    /// Generate `jolt_panic()` function that writes to the panic address.
    /// This is called by the runtime's `#[panic_handler]` to signal panics to the prover.
    fn make_panic(&self, panic_address: u64) -> TokenStream2 {
        quote! {
            #[cfg(feature = "guest")]
            #[no_mangle]
            pub extern "C" fn jolt_panic() {
                unsafe {
                    core::ptr::write_volatile(#panic_address as *mut u8, 1);
                }
            }
        }
    }

    fn make_allocator(&self) -> TokenStream2 {
        // The allocator is provided by jolt-sdk's boot modules:
        // - std mode: guest_std_boot.rs uses linked_list_allocator
        // - no-std mode: ZeroOS jolt-platform provides the global allocator
        quote! {}
    }

    fn make_require_zk_check(&self) -> TokenStream2 {
        if !self.has_private_input {
            return quote! {};
        }
        let fn_name = self.get_func_name();
        let msg = format!(
            "Guest function `{fn_name}` uses `PrivateInput` which requires the `zk` feature. \
             Enable `features = [\"host\", \"zk\"]` on `jolt-sdk` in the host Cargo.toml."
        );
        quote! {
            #[cfg(all(not(feature = "guest"), not(target_arch = "wasm32")))]
            const _: () = assert!(jolt::_ZK_FEATURE_ENABLED, #msg);
        }
    }

    fn make_imports(&self) -> TokenStream2 {
        quote! {
            #[cfg(not(feature = "guest"))]
            use jolt::{
                host::Program,
                host::JoltProgramSource,
                MemoryConfig,
                MemoryLayout,
                JoltDevice,
            };
            use jolt::{
                JoltVerifierPreprocessing,
                JoltSharedPreprocessing
            };
        }
    }

    fn make_set_linker_parameters(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
        let mut code: Vec<TokenStream2> = Vec::new();

        let value = attributes.heap_size;
        code.push(quote! {
            program.set_heap_size(#value);
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

        let value = attributes.max_untrusted_advice_size;
        code.push(quote! {
            program.set_max_untrusted_advice_size(#value);
        });

        let value = attributes.max_trusted_advice_size;
        code.push(quote! {
            program.set_max_trusted_advice_size(#value);
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

    fn make_set_backtrace(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
        if let Some(features) = attributes.backtrace {
            quote! {
                program.set_backtrace(#features);
            }
        } else {
            quote! {}
        }
    }

    fn make_set_profile(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
        if let Some(profile) = attributes.profile {
            quote! {
                program.set_profile(#profile);
            }
        } else {
            quote! {}
        }
    }

    fn make_enable_field_inline(&self) -> TokenStream2 {
        quote! {
            #[cfg(feature = "field-inline")]
            {
                program.enable_field_inline();
            }
        }
    }

    fn get_prove_output_type(&self) -> TokenStream2 {
        match &self.func.sig.output {
            ReturnType::Default => quote! {
                ((), jolt::RV64IMACProof, jolt::JoltDevice)
            },
            ReturnType::Type(_, ty) => quote! {
                (#ty, jolt::RV64IMACProof, jolt::JoltDevice)
            },
        }
    }

    fn get_all_func_args_in_order(&self) -> Vec<(Ident, Box<Type>)> {
        self.func
            .sig
            .inputs
            .iter()
            .map(|arg| {
                if let syn::FnArg::Typed(PatType { pat, ty, .. }) = arg {
                    if let syn::Pat::Ident(pat_ident) = pat.as_ref() {
                        (pat_ident.ident.clone(), ty.clone())
                    } else {
                        panic!("cannot parse arg");
                    }
                } else {
                    panic!("cannot parse arg");
                }
            })
            .collect()
    }

    #[allow(clippy::type_complexity)]
    fn get_func_args(
        func: &ItemFn,
    ) -> (
        Vec<(Ident, Box<Type>)>,
        Vec<(Ident, Box<Type>)>,
        Vec<(Ident, Box<Type>)>,
    ) {
        let mut pub_args = Vec::new();
        let mut trusted_advice_args = Vec::new();
        let mut untrusted_advice_args = Vec::new();

        for arg in &func.sig.inputs {
            if let syn::FnArg::Typed(PatType { pat, ty, .. }) = arg {
                if let syn::Pat::Ident(pat_ident) = pat.as_ref() {
                    let ident = pat_ident.ident.clone();
                    let arg_type = ty.clone();

                    // Check if the type is wrapped in jolt::TrustedAdvice<> or jolt::UntrustedAdvice<>
                    if Self::is_trusted_advice_type(&arg_type) {
                        trusted_advice_args.push((ident, arg_type));
                    } else if Self::is_untrusted_advice_type(&arg_type) {
                        untrusted_advice_args.push((ident, arg_type));
                    } else {
                        pub_args.push((ident, arg_type));
                    }
                } else {
                    panic!("cannot parse arg");
                }
            } else {
                panic!("cannot parse arg");
            }
        }

        (pub_args, trusted_advice_args, untrusted_advice_args)
    }

    fn is_trusted_advice_type(ty: &Type) -> bool {
        if let Type::Path(type_path) = ty {
            if let Some(last_segment) = type_path.path.segments.last() {
                return last_segment.ident == "TrustedAdvice";
            }
        }
        false
    }

    fn is_untrusted_advice_type(ty: &Type) -> bool {
        if let Type::Path(type_path) = ty {
            if let Some(last_segment) = type_path.path.segments.last() {
                return last_segment.ident == "UntrustedAdvice"
                    || last_segment.ident == "PrivateInput";
            }
        }
        false
    }

    fn is_private_input_type(ty: &Type) -> bool {
        if let Type::Path(type_path) = ty {
            if let Some(last_segment) = type_path.path.segments.last() {
                return last_segment.ident == "PrivateInput";
            }
        }
        false
    }

    fn any_arg_is_private_input(func: &ItemFn) -> bool {
        func.sig.inputs.iter().any(|arg| {
            if let syn::FnArg::Typed(PatType { ty, .. }) = arg {
                Self::is_private_input_type(ty)
            } else {
                false
            }
        })
    }

    fn get_func_name(&self) -> &Ident {
        &self.func.sig.ident
    }

    fn get_guest_name(&self) -> String {
        std::env::var("CARGO_PKG_NAME").unwrap()
    }

    fn get_func_selector(&self) -> Option<String> {
        std::env::var("JOLT_FUNC_NAME").ok()
    }

    fn has_wasm_attr(&self) -> bool {
        parse_attributes(&self.attr).wasm
    }

    fn make_wasm_function(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let verify_wasm_fn_name = Ident::new(&format!("verify_{fn_name}"), fn_name.span());

        quote! {
            #[cfg(all(target_arch = "wasm32", not(feature = "guest")))]
            #[wasm_bindgen::prelude::wasm_bindgen]
            pub fn #verify_wasm_fn_name(
                preprocessing_data: &[u8],
                proof_bytes: &[u8],
                io_bytes: &[u8],
                trusted_advice_commitment_bytes: &[u8],
            ) -> bool {
                let preprocessing: jolt::JoltVerifierPreprocessing =
                    match jolt::deserialize_verifier_object(preprocessing_data) {
                    Ok(preprocessing) => preprocessing,
                    Err(_) => return false,
                };
                let proof: jolt::RV64IMACProof =
                    match jolt::deserialize_verifier_object(proof_bytes) {
                    Ok(proof) => proof,
                    Err(_) => return false,
                };
                let io_device: jolt::JoltDevice =
                    match jolt::deserialize_verifier_object(io_bytes) {
                    Ok(io_device) => io_device,
                    Err(_) => return false,
                };
                let trusted_advice_commitment:
                    Option<jolt::VerifierTrustedAdviceCommitment> =
                    if trusted_advice_commitment_bytes.is_empty() {
                        None
                    } else {
                        match jolt::deserialize_verifier_object(trusted_advice_commitment_bytes) {
                            Ok(commitment) => commitment,
                            Err(_) => return false,
                        }
                    };

                jolt::jolt_verifier::verify::<
                    jolt::VerifierField,
                    jolt::VerifierPCS,
                    jolt::VerifierVC,
                    jolt::VerifierTranscript,
                >(
                    &preprocessing,
                    &io_device,
                    &proof,
                    trusted_advice_commitment.as_ref(),
                ).is_ok()
            }
        }
    }
}

/// Proc macro for advice functions.
///
/// Generates two versions of the function:
/// - With `compute_advice` feature: executes the original body and writes result to advice tape
/// - Without `compute_advice` feature: reads result from advice tape
///
/// The return type must be wrapped in `UntrustedAdvice<T>`.
#[proc_macro_attribute]
pub fn advice(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    // Extract function components
    let fn_name = &func.sig.ident;
    let fn_vis = &func.vis;
    let fn_inputs = &func.sig.inputs;
    let fn_output = &func.sig.output;
    let fn_body = &func.block;
    let fn_attrs = &func.attrs;

    // Validate that no inputs are mutable
    for arg in fn_inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            // Case 1: mut x: T
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                if pat_ident.mutability.is_some() {
                    panic!(
                        "#[jolt::advice] mutable argument '{}' in function '{}'. Mutable arguments are not allowed in advice functions",
                        pat_ident.ident, fn_name
                    );
                }
            }
            // Case 2: x: &mut T
            if let syn::Type::Reference(type_ref) = &*pat_type.ty {
                if type_ref.mutability.is_some() {
                    panic!(
                        "#[jolt::advice] mutable argument '{}' in function '{}'. Mutable arguments are not allowed in advice functions",
                        if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                            pat_ident.ident.to_string()
                        } else {
                            "<unknown>".to_string()
                        },
                        fn_name
                    );
                }
            }
        }
    }

    // Validate return type is UntrustedAdvice<T>
    let inner_type = match fn_output {
        ReturnType::Type(_, ty) => {
            // Check if type is UntrustedAdvice<T>
            if let Type::Path(type_path) = &**ty {
                if let Some(segment) = type_path.path.segments.last() {
                    if segment.ident == "UntrustedAdvice" {
                        // Extract T from UntrustedAdvice<T>
                        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                            if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                                inner.clone()
                            } else {
                                panic!("#[jolt::advice] return type must be UntrustedAdvice<T>");
                            }
                        } else {
                            panic!("#[jolt::advice] return type must be UntrustedAdvice<T>");
                        }
                    } else {
                        panic!(
                            "#[jolt::advice] return type must be UntrustedAdvice<T>, found {}",
                            segment.ident
                        );
                    }
                } else {
                    panic!("#[jolt::advice] return type must be UntrustedAdvice<T>");
                }
            } else {
                panic!("#[jolt::advice] return type must be UntrustedAdvice<T>");
            }
        }
        ReturnType::Default => {
            panic!("#[jolt::advice] function must return UntrustedAdvice<T>");
        }
    };

    // Generate the dual-mode function
    let expanded = quote! {
        // Version with compute_advice: execute body and write to advice tape
        #[cfg(feature = "compute_advice")]
        #(#fn_attrs)*
        #fn_vis fn #fn_name(#fn_inputs) #fn_output {
            // execute body
            let result: #inner_type = #fn_body;
            // Serialize and write to advice tape
            <#inner_type as jolt::AdviceTapeIO>::write_to_advice_tape(&result);
            // return result
            jolt::UntrustedAdvice::new(result)
        }

        // Version without compute_advice: read from advice tape
        #[cfg(not(feature = "compute_advice"))]
        #[allow(unused_variables)]
        #(#fn_attrs)*
        #fn_vis fn #fn_name(#fn_inputs) #fn_output {
            // get result from advice tape
            let result: #inner_type = <#inner_type as jolt::AdviceTapeIO>::new_from_advice_tape();
            // wrap in UntrustedAdvice and return
            jolt::UntrustedAdvice::new(result)
        }
    };

    TokenStream::from(expanded)
}
