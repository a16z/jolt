extern crate proc_macro;

use core::panic;

use common::{
    attributes::parse_attributes,
    jolt_device::{MemoryConfig, MemoryLayout},
};
use proc_macro::TokenStream;
use proc_macro2::{Literal, TokenStream as TokenStream2};
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
            // Cargo cannot see this macro's own `std::env::var` read, but rustc
            // records `option_env!` reads as dep-info env-deps, so a changed
            // selector invalidates the cached guest build.
            #[cfg(feature = "guest")]
            const _: ::core::option::Option<&str> = ::core::option_env!("JOLT_FUNC_NAME");

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
        let max_input_size = Literal::u64_unsuffixed(attributes.max_input_size);
        let max_output_size = Literal::u64_unsuffixed(attributes.max_output_size);
        let max_trusted_advice_size = Literal::u64_unsuffixed(attributes.max_trusted_advice_size);
        let max_untrusted_advice_size =
            Literal::u64_unsuffixed(attributes.max_untrusted_advice_size);
        let stack_size = Literal::u64_unsuffixed(attributes.stack_size);
        let heap_size = Literal::u64_unsuffixed(attributes.heap_size);

        let memory_config_fn_name = Ident::new(&format!("memory_config_{fn_name}"), fn_name.span());

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #memory_config_fn_name() -> jolt::MemoryConfig {
                jolt::MemoryConfig {
                    max_input_size: #max_input_size,
                    max_output_size: #max_output_size,
                    max_trusted_advice_size: #max_trusted_advice_size,
                    max_untrusted_advice_size: #max_untrusted_advice_size,
                    stack_size: #stack_size,
                    heap_size: #heap_size,
                    program_size: ::core::option::Option::None,
                }
            }
        }
    }

    fn make_build_prover_fn(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let build_prover_fn_name = Ident::new(&format!("build_prover_{fn_name}"), fn_name.span());
        let prove_output_ty = self.get_prove_output_type();

        let ordered_func_args = self.get_all_func_args_in_order();
        let all_names: Vec<_> = ordered_func_args.iter().map(|(name, _)| name).collect();
        let all_types: Vec<_> = ordered_func_args.iter().map(|(_, ty)| ty).collect();

        let inputs_vec: Vec<_> = self.func.sig.inputs.iter().collect();
        let inputs = quote! { #(#inputs_vec),* };
        let prove_fn_name = Ident::new(&format!("prove_{fn_name}"), fn_name.span());

        let has_trusted_advice = !self.trusted_func_args.is_empty();

        let commitment_param_in_closure = if has_trusted_advice {
            quote! { , __jolt_trusted_advice_commitment: ::core::option::Option<jolt::VerifierTrustedAdviceCommitment>,
            __jolt_trusted_advice_hint: ::core::option::Option<jolt::TrustedAdviceOpeningHint> }
        } else {
            quote! {}
        };

        let commitment_arg_in_call = if has_trusted_advice {
            quote! { , __jolt_trusted_advice_commitment, __jolt_trusted_advice_hint }
        } else {
            quote! {}
        };

        let return_type = if has_trusted_advice {
            quote! {
                impl ::core::ops::Fn(#(#all_types),*, ::core::option::Option<jolt::VerifierTrustedAdviceCommitment>, ::core::option::Option<jolt::TrustedAdviceOpeningHint>) -> #prove_output_ty + ::core::marker::Sync + ::core::marker::Send
            }
        } else {
            quote! {
                impl ::core::ops::Fn(#(#all_types),*) -> #prove_output_ty + ::core::marker::Sync + ::core::marker::Send
            }
        };

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #build_prover_fn_name<__S: jolt::host::JoltProgramSource + ::core::marker::Send + ::core::marker::Sync + 'static>(
                program: __S,
                preprocessing: jolt::JoltProverPreprocessing,
            ) -> #return_type
            {
                let __jolt_program = ::std::sync::Arc::new(program);
                let __jolt_preprocessing = ::std::sync::Arc::new(preprocessing);

                let __jolt_prove_closure = move |#inputs #commitment_param_in_closure| {
                    let __jolt_preprocessing = (*__jolt_preprocessing).clone();
                    #prove_fn_name(__jolt_program.as_ref(), __jolt_preprocessing, #(#all_names),* #commitment_arg_in_call)
                };

                __jolt_prove_closure
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
        let set_program_args = self.pub_func_args.iter().map(|(name, _)| {
            quote! {
                __jolt_io_device.inputs.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });

        let has_trusted_advice = !self.trusted_func_args.is_empty();

        let commitment_param_in_signature = if has_trusted_advice {
            quote! { ::core::option::Option<jolt::VerifierTrustedAdviceCommitment>, }
        } else {
            quote! {}
        };

        let commitment_param_in_closure = if has_trusted_advice {
            quote! { __jolt_trusted_advice_commitment: ::core::option::Option<jolt::VerifierTrustedAdviceCommitment>, }
        } else {
            quote! {}
        };

        let commitment_arg_in_verify = if has_trusted_advice {
            quote! { __jolt_trusted_advice_commitment.as_ref() }
        } else {
            quote! { ::core::option::Option::None }
        };

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #build_verifier_fn_name(
                preprocessing: jolt::JoltVerifierPreprocessing,
            ) -> impl ::core::ops::Fn(#(#input_types ,)* #output_type, bool, #commitment_param_in_signature jolt::RV64IMACProof) -> bool + ::core::marker::Sync + ::core::marker::Send
            {
                let __jolt_preprocessing = ::std::sync::Arc::new(preprocessing);

                let __jolt_verify_closure = move |#(#public_inputs,)* __jolt_output, __jolt_panic, #commitment_param_in_closure __jolt_proof: jolt::RV64IMACProof| {
                    let __jolt_preprocessing = (*__jolt_preprocessing).clone();
                    let __jolt_memory_layout = __jolt_preprocessing.program.memory_layout();
                    let __jolt_memory_config = jolt::MemoryConfig {
                        max_input_size: __jolt_memory_layout.max_input_size,
                        max_output_size: __jolt_memory_layout.max_output_size,
                        max_untrusted_advice_size: __jolt_memory_layout.max_untrusted_advice_size,
                        max_trusted_advice_size: __jolt_memory_layout.max_trusted_advice_size,
                        stack_size: __jolt_memory_layout.stack_size,
                        heap_size: __jolt_memory_layout.heap_size,
                        program_size: ::core::option::Option::Some(__jolt_memory_layout.program_size),
                    };
                    let mut __jolt_io_device = jolt::JoltDevice::new(&__jolt_memory_config);

                    #(#set_program_args;)*
                    __jolt_io_device.outputs.append(&mut jolt::postcard::to_stdvec(&__jolt_output).unwrap());
                    __jolt_io_device.panic = __jolt_panic;

                    jolt::jolt_verifier::verify::<
                        jolt::VerifierField,
                        jolt::VerifierPCS,
                        jolt::VerifierVC,
                        jolt::VerifierTranscript,
                    >(
                        &__jolt_preprocessing,
                        &__jolt_io_device,
                        &__jolt_proof,
                        #commitment_arg_in_verify,
                    ).is_ok()
                };

                __jolt_verify_closure
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
                __jolt_input_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_untrusted_advice_args = self.untrusted_func_args.iter().map(|(name, _)| {
            quote! {
                __jolt_untrusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_trusted_advice_args = self.trusted_func_args.iter().map(|(name, _)| {
            quote! {
                __jolt_trusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });

        quote! {
             #[cfg(not(target_arch = "wasm32"))]
             #[cfg(not(feature = "guest"))]
             pub fn #analyze_fn_name(#inputs) -> jolt::host::analyze::ProgramSummary {
                let mut __jolt_program = jolt::host::Program::new(#guest_name);
                __jolt_program.set_func(#fn_name_str);
                #set_std
                #set_profile
                #set_backtrace
                #enable_field_inline
                #set_mem_size

                let mut __jolt_input_bytes = ::std::vec::Vec::new();
                #(#set_pub_args;)*
                let mut __jolt_untrusted_advice_bytes = ::std::vec::Vec::new();
                #(#set_untrusted_advice_args;)*
                let mut __jolt_trusted_advice_bytes = ::std::vec::Vec::new();
                #(#set_trusted_advice_args;)*

                __jolt_program.trace_analyze(&__jolt_input_bytes, &__jolt_untrusted_advice_bytes, &__jolt_trusted_advice_bytes)
             }
        }
    }

    fn make_trace_func(&self) -> TokenStream2 {
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
                __jolt_input_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_untrusted_advice_args = self.untrusted_func_args.iter().map(|(name, _)| {
            quote! {
                __jolt_untrusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_trusted_advice_args = self.trusted_func_args.iter().map(|(name, _)| {
            quote! {
                __jolt_trusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #trace_fn_name(#inputs) -> ::core::result::Result<jolt::TraceOutput<jolt::OwnedTrace>, jolt::TraceError> {
                let mut __jolt_backend = jolt::TracerBackend::new();
                #trace_with_backend_fn_name(&mut __jolt_backend, #(#all_names),*)
            }

            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #trace_with_backend_fn_name<__B: jolt::ExecutionBackend>(
                __jolt_backend: &mut __B,
                #inputs
            ) -> ::core::result::Result<jolt::TraceOutput<__B::Trace>, jolt::TraceError> {
                let mut __jolt_program = jolt::host::Program::new(#guest_name);
                __jolt_program.set_func(#fn_name_str);
                #set_std
                #set_profile
                #set_backtrace
                #enable_field_inline
                #set_mem_size

                let mut __jolt_input_bytes = ::std::vec::Vec::new();
                #(#set_pub_args;)*
                let mut __jolt_untrusted_advice_bytes = ::std::vec::Vec::new();
                #(#set_untrusted_advice_args;)*
                let mut __jolt_trusted_advice_bytes = ::std::vec::Vec::new();
                #(#set_trusted_advice_args;)*

                __jolt_program.trace_with_backend(
                    __jolt_backend,
                    &__jolt_input_bytes,
                    &__jolt_untrusted_advice_bytes,
                    &__jolt_trusted_advice_bytes,
                )
            }
        }
    }

    fn make_trace_to_file_func(&self) -> TokenStream2 {
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
                __jolt_input_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_untrusted_advice_args = self.untrusted_func_args.iter().map(|(name, _)| {
            quote! {
                __jolt_untrusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_trusted_advice_args = self.trusted_func_args.iter().map(|(name, _)| {
            quote! {
                __jolt_trusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #trace_to_file_fn_name(__jolt_target_dir: &str, #inputs) {
                let mut __jolt_program = jolt::host::Program::new(#guest_name);
                let __jolt_path = ::std::path::PathBuf::from(__jolt_target_dir);
                __jolt_program.set_func(#fn_name_str);
                #set_std
                #set_profile
                #set_backtrace
                #enable_field_inline
                #set_mem_size

                let mut __jolt_input_bytes = ::std::vec::Vec::new();
                #(#set_pub_args;)*
                let mut __jolt_untrusted_advice_bytes = ::std::vec::Vec::new();
                #(#set_untrusted_advice_args;)*
                let mut __jolt_trusted_advice_bytes = ::std::vec::Vec::new();
                #(#set_trusted_advice_args;)*

                __jolt_program.trace_to_file(&__jolt_input_bytes, &__jolt_untrusted_advice_bytes, &__jolt_trusted_advice_bytes, &__jolt_path);
            }
        }
    }

    fn make_compile_func(&self) -> TokenStream2 {
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
                let mut __jolt_program = jolt::host::Program::new(#guest_name);
                __jolt_program.set_func(#fn_name_str);
                #set_std
                #set_profile
                #set_backtrace
                #enable_field_inline
                #set_mem_size

                __jolt_program.build_with_features(target_dir, &["compute_advice"]);

                __jolt_program.build_with_features(target_dir, &[]);

                __jolt_program
            }
        }
    }

    fn make_preprocess_shared_func(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
        let max_trace_length = Literal::u64_unsuffixed(attributes.max_trace_length);

        let fn_name = self.get_func_name();
        let preprocess_shared_fn_name =
            Ident::new(&format!("preprocess_shared_{fn_name}"), fn_name.span());
        let memory_config_fn_name = Ident::new(&format!("memory_config_{fn_name}"), fn_name.span());
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #preprocess_shared_fn_name(program: &mut dyn jolt::host::JoltProgramSource)
                -> ::core::result::Result<jolt::JoltSharedPreprocessing, jolt::PreprocessingError>
            {
                jolt::preprocess_shared_program(
                    program,
                    #memory_config_fn_name(),
                    #max_trace_length,
                )
            }
        }
    }

    fn make_preprocess_prover_func(&self) -> TokenStream2 {
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
                jolt::jolt_prover::dory::from_shared(shared_preprocessing)
                    .expect("Dory prover preprocessing")
            }
        }
    }

    fn make_preprocess_committed_prover_func(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
        let max_trace_length = Literal::u64_unsuffixed(attributes.max_trace_length);

        let fn_name = self.get_func_name();
        let preprocess_committed_fn_name =
            Ident::new(&format!("preprocess_committed_{fn_name}"), fn_name.span());
        let memory_config_fn_name = Ident::new(&format!("memory_config_{fn_name}"), fn_name.span());
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #preprocess_committed_fn_name(
                program: &mut jolt::host::Program,
                bytecode_chunk_count: usize,
            )
                -> ::core::result::Result<
                    jolt::JoltProverPreprocessing,
                    jolt::PreprocessingError,
                >
            {
                jolt::preprocess_program(
                    program,
                    #memory_config_fn_name(),
                    #max_trace_length,
                    ::core::option::Option::Some(bytecode_chunk_count),
                )
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
                blindfold_setup: ::core::option::Option<jolt::BlindfoldSetup>,
            ) -> jolt::JoltVerifierPreprocessing
            {
                jolt::jolt_prover::dory::from_shared_parts(
                    &shared_preprocess,
                    generators,
                    blindfold_setup,
                )
                .expect("Dory verifier preprocessing")
            }
        }
    }

    fn make_preprocess_from_prover_func(&self) -> TokenStream2 {
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
                prover_preprocessing.verifier_preprocessing()
            }
        }
    }

    fn make_commit_trusted_advice_func(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let commit_fn_name =
            Ident::new(&format!("commit_trusted_advice_{fn_name}"), fn_name.span());

        if self.trusted_func_args.is_empty() {
            return quote! {
                #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
                pub fn #commit_fn_name(
                    _preprocessing: &jolt::JoltProverPreprocessing,
                ) -> (::core::option::Option<jolt::VerifierTrustedAdviceCommitment>,
                      ::core::option::Option<jolt::TrustedAdviceOpeningHint>)
                {
                    (::core::option::Option::None, ::core::option::Option::None)
                }
            };
        }

        let trusted_advice_inputs = self.trusted_func_args.iter().map(|(name, ty)| {
            quote! { #name: #ty }
        });

        let set_trusted_advice_args = self.trusted_func_args.iter().map(|(name, _)| {
            quote! {
                __jolt_trusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #commit_fn_name(
                #(#trusted_advice_inputs,)*
                __jolt_preprocessing: &jolt::JoltProverPreprocessing,
            ) -> (::core::option::Option<jolt::VerifierTrustedAdviceCommitment>,
                  ::core::option::Option<jolt::TrustedAdviceOpeningHint>)
            {
                let mut __jolt_trusted_advice_bytes = ::std::vec::Vec::new();
                #(#set_trusted_advice_args;)*
                let __jolt_committed = jolt::jolt_prover::dory::commit_trusted_advice(
                    __jolt_preprocessing,
                    &__jolt_trusted_advice_bytes,
                ).expect("trusted advice fits the configured memory layout");
                (::core::option::Option::Some(__jolt_committed.commitment), ::core::option::Option::Some(__jolt_committed.hint))
            }
        }
    }

    fn make_prove_func(&self) -> TokenStream2 {
        let prove_output_ty = self.get_prove_output_type();

        let handle_return = match &self.func.sig.output {
            ReturnType::Default => quote! {
                let __jolt_ret_val = ();
            },
            ReturnType::Type(_, ty) => quote! {
                let mut __jolt_outputs = __jolt_io_device.outputs.clone();
                __jolt_outputs.resize(
                    __jolt_preprocessing.verifier.program.memory_layout().max_output_size as usize,
                    0,
                );
                let __jolt_ret_val = jolt::postcard::from_bytes::<#ty>(&__jolt_outputs).unwrap();
            },
        };

        let set_program_args = self.pub_func_args.iter().map(|(name, _)| {
            quote! {
                __jolt_input_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_program_untrusted_advice_args = self.untrusted_func_args.iter().map(|(name, _)| {
            quote! {
                __jolt_untrusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });
        let set_program_trusted_advice_args = self.trusted_func_args.iter().map(|(name, _)| {
            quote! {
                __jolt_trusted_advice_bytes.append(&mut jolt::postcard::to_stdvec(&#name).unwrap())
            }
        });

        let fn_name = self.get_func_name();
        let inputs_vec: Vec<_> = self.func.sig.inputs.iter().collect();
        let inputs = quote! { #(#inputs_vec),* };

        let prove_fn_name = syn::Ident::new(&format!("prove_{fn_name}"), fn_name.span());

        let has_trusted_advice = !self.trusted_func_args.is_empty();

        let commitment_param = if has_trusted_advice {
            quote! { , __jolt_trusted_advice_commitment: ::core::option::Option<jolt::VerifierTrustedAdviceCommitment>,
            __jolt_trusted_advice_hint: ::core::option::Option<jolt::TrustedAdviceOpeningHint> }
        } else {
            quote! {}
        };

        let commitment_arg = if has_trusted_advice {
            quote! { __jolt_trusted_advice_commitment, __jolt_trusted_advice_hint }
        } else {
            quote! { ::core::option::Option::None, ::core::option::Option::None }
        };

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            #[allow(clippy::too_many_arguments)]
            pub fn #prove_fn_name(
                __jolt_program: &dyn jolt::host::JoltProgramSource,
                __jolt_preprocessing: jolt::JoltProverPreprocessing,
                #inputs
                #commitment_param
            ) -> #prove_output_ty {
                let mut __jolt_input_bytes = ::std::vec::Vec::new();
                #(#set_program_args;)*
                let mut __jolt_untrusted_advice_bytes = ::std::vec::Vec::new();
                #(#set_program_untrusted_advice_args;)*
                let mut __jolt_trusted_advice_bytes = ::std::vec::Vec::new();
                #(#set_program_trusted_advice_args;)*

                let __jolt_advice_tape = jolt::compute_advice_tape(
                    __jolt_program,
                    &__jolt_input_bytes,
                    &__jolt_untrusted_advice_bytes,
                    &__jolt_trusted_advice_bytes,
                    __jolt_preprocessing.verifier.program.memory_layout(),
                ).expect("compute-advice execution should succeed");
                let (__jolt_proof, __jolt_io_device) = jolt::prove_program(
                    __jolt_program,
                    &__jolt_preprocessing,
                    &__jolt_input_bytes,
                    &__jolt_untrusted_advice_bytes,
                    &__jolt_trusted_advice_bytes,
                    #commitment_arg,
                    __jolt_advice_tape,
                ).expect("execution trace exceeds the max_trace_length configured in #[jolt::provable]");

                #handle_return

                (__jolt_ret_val, __jolt_proof, __jolt_io_device)
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
            let __jolt_input_ptr = #input_start as *const u8;
            let __jolt_input_slice = unsafe {
                ::core::slice::from_raw_parts(__jolt_input_ptr, #max_input_len)
            };
        };

        let get_untrusted_advice_slice = quote! {
            let __jolt_untrusted_advice_ptr = #untrusted_advice_start as *const u8;
            let __jolt_untrusted_advice_slice = unsafe {
                ::core::slice::from_raw_parts(__jolt_untrusted_advice_ptr, #max_untrusted_advice_len)
            };
        };

        let get_trusted_advice_slice = quote! {
            let __jolt_trusted_advice_ptr = #trusted_advice_start as *const u8;
            let __jolt_trusted_advice_slice = unsafe {
                ::core::slice::from_raw_parts(__jolt_trusted_advice_ptr, #max_trusted_advice_len)
            };
        };

        let pub_args_fetch = self.pub_func_args.iter().map(|(name, ty)| {
            quote! {
                let (#name, __jolt_input_slice) =
                    jolt::postcard::take_from_bytes::<#ty>(__jolt_input_slice).unwrap();
            }
        });

        let untrusted_advice_args_fetch = self.untrusted_func_args.iter().map(|(name, ty)| {
            quote! {
                let (#name, __jolt_untrusted_advice_slice) =
                    jolt::postcard::take_from_bytes::<#ty>(__jolt_untrusted_advice_slice).unwrap();
            }
        });

        let trusted_advice_args_fetch = self.trusted_func_args.iter().map(|(name, ty)| {
            quote! {
                let (#name, __jolt_trusted_advice_slice) =
                    jolt::postcard::take_from_bytes::<#ty>(__jolt_trusted_advice_slice).unwrap();
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
            let __jolt_to_return = #inner_fn_name(#(#all_names),*);
        };

        let handle_return = match &self.func.sig.output {
            ReturnType::Default => quote! {},
            ReturnType::Type(_, ty) => quote! {
                let __jolt_output_ptr = #output_start as *mut u8;
                let __jolt_output_slice = unsafe {
                    ::core::slice::from_raw_parts_mut(__jolt_output_ptr, #max_output_len)
                };

                jolt::postcard::to_slice::<#ty>(&__jolt_to_return, __jolt_output_slice).unwrap();
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
                let mut __jolt_offset = 0;
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
                    ::core::ptr::write_volatile(#termination_bit as *mut u8, 1);
                }
                // Never return - loop forever for clean termination
                // The emulator detects termination via PC stall (prev_pc == pc)
                loop {
                    unsafe { ::core::arch::asm!("j .", options(noreturn)); }
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
                    ::core::ptr::write_volatile(#panic_address as *mut u8, 1);
                }
            }
        }
    }

    fn make_allocator(&self) -> TokenStream2 {
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
            const _: () = ::core::assert!(jolt::_ZK_FEATURE_ENABLED, #msg);
        }
    }

    fn make_set_linker_parameters(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
        let mut code: Vec<TokenStream2> = Vec::new();

        let value = attributes.heap_size;
        code.push(quote! {
            __jolt_program.set_heap_size(#value);
        });

        let value = attributes.stack_size;
        code.push(quote! {
            __jolt_program.set_stack_size(#value);
        });

        let value = attributes.max_input_size;
        code.push(quote! {
            __jolt_program.set_max_input_size(#value);
        });

        let value = attributes.max_output_size;
        code.push(quote! {
            __jolt_program.set_max_output_size(#value);
        });

        let value = attributes.max_untrusted_advice_size;
        code.push(quote! {
            __jolt_program.set_max_untrusted_advice_size(#value);
        });

        let value = attributes.max_trusted_advice_size;
        code.push(quote! {
            __jolt_program.set_max_trusted_advice_size(#value);
        });

        quote! {
            #(#code;)*
        }
    }

    fn make_set_std(&self) -> TokenStream2 {
        if self.std {
            quote! {
                __jolt_program.set_std(true);
            }
        } else {
            quote! {
                __jolt_program.set_std(false);
            }
        }
    }

    fn make_set_backtrace(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
        if let Some(features) = attributes.backtrace {
            quote! {
                __jolt_program.set_backtrace(#features);
            }
        } else {
            quote! {}
        }
    }

    fn make_set_profile(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
        if let Some(profile) = attributes.profile {
            quote! {
                __jolt_program.set_profile(#profile);
            }
        } else {
            quote! {}
        }
    }

    fn make_enable_field_inline(&self) -> TokenStream2 {
        quote! {
            #[cfg(feature = "field-inline")]
            {
                __jolt_program.enable_field_inline();
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
                    ::core::result::Result::Ok(preprocessing) => preprocessing,
                    ::core::result::Result::Err(_) => return false,
                };
                let proof: jolt::RV64IMACProof =
                    match jolt::deserialize_verifier_object(proof_bytes) {
                    ::core::result::Result::Ok(proof) => proof,
                    ::core::result::Result::Err(_) => return false,
                };
                let io_device: jolt::JoltDevice =
                    match jolt::deserialize_verifier_object(io_bytes) {
                    ::core::result::Result::Ok(io_device) => io_device,
                    ::core::result::Result::Err(_) => return false,
                };
                let trusted_advice_commitment:
                    ::core::option::Option<jolt::VerifierTrustedAdviceCommitment> =
                    if trusted_advice_commitment_bytes.is_empty() {
                        ::core::option::Option::None
                    } else {
                        match jolt::deserialize_verifier_object(trusted_advice_commitment_bytes) {
                            ::core::result::Result::Ok(commitment) => commitment,
                            ::core::result::Result::Err(_) => return false,
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

    let fn_name = &func.sig.ident;
    let fn_vis = &func.vis;
    let fn_inputs = &func.sig.inputs;
    let fn_output = &func.sig.output;
    let fn_body = &func.block;
    let fn_attrs = &func.attrs;

    for arg in fn_inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                if pat_ident.mutability.is_some() {
                    panic!(
                        "#[jolt::advice] mutable argument '{}' in function '{}'. Mutable arguments are not allowed in advice functions",
                        pat_ident.ident, fn_name
                    );
                }
            }
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

    let inner_type = match fn_output {
        ReturnType::Type(_, ty) => {
            if let Type::Path(type_path) = &**ty {
                if let Some(segment) = type_path.path.segments.last() {
                    if segment.ident == "UntrustedAdvice" {
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

    let expanded = quote! {
        #[cfg(feature = "compute_advice")]
        #(#fn_attrs)*
        #fn_vis fn #fn_name(#fn_inputs) #fn_output {
            let result: #inner_type = #fn_body;
            <#inner_type as jolt::AdviceTapeIO>::write_to_advice_tape(&result);
            jolt::UntrustedAdvice::new(result)
        }

        #[cfg(not(feature = "compute_advice"))]
        #[allow(unused_variables)]
        #(#fn_attrs)*
        #fn_vis fn #fn_name(#fn_inputs) #fn_output {
            let result: #inner_type = <#inner_type as jolt::AdviceTapeIO>::new_from_advice_tape();
            jolt::UntrustedAdvice::new(result)
        }
    };

    TokenStream::from(expanded)
}
