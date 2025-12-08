extern crate proc_macro;

use core::panic;

use common::{
    attributes::parse_attributes,
    jolt_device::{MemoryConfig, MemoryLayout},
};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::sync::Once;
use syn::{parse_macro_input, AttributeArgs, Ident, ItemFn, PatType, ReturnType, Type};

static WASM_IMPORTS_INIT: Once = Once::new();

#[proc_macro_attribute]
pub fn provable(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as AttributeArgs);
    let func = parse_macro_input!(item as ItemFn);
    let mut builder = MacroBuilder::new(attr, func);

    let mut token_stream = builder.build();

    // Add wasm utilities and functions if the function is marked as wasm
    if builder.has_wasm_attr() {
        // wasm utilities should only be added once
        WASM_IMPORTS_INIT.call_once(|| {
            let wasm_utilities: TokenStream = builder.make_wasm_utilities().into();
            token_stream.extend(wasm_utilities);
        });
        let wasm_token_stream: TokenStream = builder.make_wasm_function().into();
        token_stream.extend(wasm_token_stream);
    }

    token_stream
}

struct MacroBuilder {
    attr: AttributeArgs,
    func: ItemFn,
    std: bool,
    pub_func_args: Vec<(Ident, Box<Type>)>,
    trusted_func_args: Vec<(Ident, Box<Type>)>,
    untrusted_func_args: Vec<(Ident, Box<Type>)>,
}

impl MacroBuilder {
    fn new(attr: AttributeArgs, func: ItemFn) -> Self {
        let (pub_func_args, trusted_func_args, untrusted_func_args) = Self::get_func_args(&func);
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
        }
    }

    fn build(&mut self) -> TokenStream {
        let memory_config_fn = self.make_memory_config_fn();
        let build_prover_fn = self.make_build_prover_fn();
        let build_verifier_fn = self.make_build_verifier_fn();
        let analyze_fn = self.make_analyze_function();
        let trace_to_file_fn = self.make_trace_to_file_func();
        let compile_fn = self.make_compile_func();
        let preprocess_prover_fn = self.make_preprocess_prover_func();
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

        quote! {
            #memory_config_fn
            #build_prover_fn
            #build_verifier_fn
            #execute_fn
            #analyze_fn
            #trace_to_file_fn
            #compile_fn
            #preprocess_prover_fn
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
        let memory_size = proc_macro2::Literal::u64_unsuffixed(attributes.memory_size);

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
                    memory_size: #memory_size,
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
            quote! { , trusted_advice_commitment: Option<<jolt::PCS as jolt::CommitmentScheme>::Commitment> }
        } else {
            quote! {}
        };

        let commitment_arg_in_call = if has_trusted_advice {
            quote! { , trusted_advice_commitment }
        } else {
            quote! {}
        };

        let return_type = if has_trusted_advice {
            quote! {
                impl Fn(#(#all_types),*, Option<<jolt::PCS as jolt::CommitmentScheme>::Commitment>) -> #prove_output_ty + Sync + Send
            }
        } else {
            quote! {
                impl Fn(#(#all_types),*) -> #prove_output_ty + Sync + Send
            }
        };

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #build_prover_fn_name(
                program: jolt::host::Program,
                preprocessing: jolt::JoltProverPreprocessing<jolt::F, jolt::PCS>,
            ) -> #return_type
            {
                #imports
                let program = std::sync::Arc::new(program);
                let preprocessing = std::sync::Arc::new(preprocessing);

                let prove_closure = move |#inputs #commitment_param_in_closure| {
                    let program = (*program).clone();
                    let preprocessing = (*preprocessing).clone();
                    #prove_fn_name(program, preprocessing, #(#all_names),* #commitment_arg_in_call)
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
            quote! { Option<<jolt::PCS as jolt::CommitmentScheme>::Commitment>, }
        } else {
            quote! {}
        };

        let commitment_param_in_closure = if has_trusted_advice {
            quote! { trusted_advice_commitment: Option<<jolt::PCS as jolt::CommitmentScheme>::Commitment>, }
        } else {
            quote! {}
        };

        let commitment_arg_in_verify = if has_trusted_advice {
            quote! { trusted_advice_commitment }
        } else {
            quote! { None }
        };

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #build_verifier_fn_name(
                preprocessing: jolt::JoltVerifierPreprocessing<jolt::F, jolt::PCS>,
            ) -> impl Fn(#(#input_types ,)* #output_type, bool, #commitment_param_in_signature jolt::RV64IMACProof) -> bool + Sync + Send
            {
                #imports
                let preprocessing = std::sync::Arc::new(preprocessing);

                let verify_closure = move |#(#public_inputs,)* output, panic, #commitment_param_in_closure proof: jolt::RV64IMACProof| {
                    let preprocessing = (*preprocessing).clone();
                    let memory_config = MemoryConfig {
                        max_input_size: preprocessing.memory_layout.max_input_size,
                        max_output_size: preprocessing.memory_layout.max_output_size,
                        max_untrusted_advice_size: preprocessing.memory_layout.max_untrusted_advice_size,
                        max_trusted_advice_size: preprocessing.memory_layout.max_trusted_advice_size,
                        stack_size: preprocessing.memory_layout.stack_size,
                        memory_size: preprocessing.memory_layout.memory_size,
                        program_size: Some(preprocessing.memory_layout.program_size),
                    };
                    let mut io_device = JoltDevice::new(&memory_config);

                    #(#set_program_args;)*
                    io_device.outputs.append(&mut jolt::postcard::to_stdvec(&output).unwrap());
                    io_device.panic = panic;

                    let verifier = RV64IMACVerifier::new(&preprocessing, proof, io_device, #commitment_arg_in_verify, None);
                    verifier.is_ok_and(|verifier| verifier.verify().is_ok())
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
                #set_mem_size

                let mut input_bytes = vec![];
                #(#set_pub_args;)*
                let mut untrusted_advice_bytes = vec![];
                #(#set_untrusted_advice_args;)*
                let mut trusted_advice_bytes = vec![];
                #(#set_trusted_advice_args;)*

                program.trace_analyze::<jolt::F>(&input_bytes, &untrusted_advice_bytes, &trusted_advice_bytes)
             }
        }
    }

    fn make_trace_to_file_func(&self) -> TokenStream2 {
        let imports = self.make_imports();
        let guest_name = self.get_guest_name();
        let set_mem_size = self.make_set_linker_parameters();
        let set_std = self.make_set_std();

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
        let attributes = parse_attributes(&self.attr);
        let imports = self.make_imports();
        let guest_name = self.get_guest_name();
        let set_mem_size = self.make_set_linker_parameters();
        let set_std = self.make_set_std();

        let channel = if attributes.nightly {
            quote! { "nightly" }
        } else {
            quote! { "stable" }
        };
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
                #set_mem_size
                program.build_with_channel(target_dir, #channel);

                program
            }
        }
    }

    fn make_preprocess_prover_func(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
        let max_input_size = proc_macro2::Literal::u64_unsuffixed(attributes.max_input_size);
        let max_output_size = proc_macro2::Literal::u64_unsuffixed(attributes.max_output_size);
        let max_untrusted_advice_size =
            proc_macro2::Literal::u64_unsuffixed(attributes.max_untrusted_advice_size);
        let max_trusted_advice_size =
            proc_macro2::Literal::u64_unsuffixed(attributes.max_trusted_advice_size);
        let stack_size = proc_macro2::Literal::u64_unsuffixed(attributes.stack_size);
        let memory_size = proc_macro2::Literal::u64_unsuffixed(attributes.memory_size);
        let max_trace_length = proc_macro2::Literal::u64_unsuffixed(attributes.max_trace_length);
        let imports = self.make_imports();

        let fn_name = self.get_func_name();
        let preprocess_prover_fn_name =
            Ident::new(&format!("preprocess_prover_{fn_name}"), fn_name.span());
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #preprocess_prover_fn_name(program: &mut jolt::host::Program)
                -> jolt::JoltProverPreprocessing<jolt::F, jolt::PCS>
            {
                #imports

                let (bytecode, memory_init, program_size) = program.decode();
                let memory_config = MemoryConfig {
                    max_input_size: #max_input_size,
                    max_output_size: #max_output_size,
                    max_untrusted_advice_size: #max_untrusted_advice_size,
                    max_trusted_advice_size: #max_trusted_advice_size,
                    stack_size: #stack_size,
                    memory_size: #memory_size,
                    program_size: Some(program_size),
                };
                let memory_layout = MemoryLayout::new(&memory_config);

                // TODO(moodlezoup): Feed in size parameters via macro
                let preprocessing = JoltProverPreprocessing::<jolt::F, jolt::PCS>::gen(
                    bytecode,
                    memory_layout,
                    memory_init,
                    #max_trace_length,
                );

                preprocessing
            }
        }
    }

    fn make_preprocess_verifier_func(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
        let max_input_size = proc_macro2::Literal::u64_unsuffixed(attributes.max_input_size);
        let max_output_size = proc_macro2::Literal::u64_unsuffixed(attributes.max_output_size);
        let max_untrusted_advice_size =
            proc_macro2::Literal::u64_unsuffixed(attributes.max_untrusted_advice_size);
        let max_trusted_advice_size =
            proc_macro2::Literal::u64_unsuffixed(attributes.max_trusted_advice_size);
        let stack_size = proc_macro2::Literal::u64_unsuffixed(attributes.stack_size);
        let memory_size = proc_macro2::Literal::u64_unsuffixed(attributes.memory_size);
        let max_trace_length = proc_macro2::Literal::u64_unsuffixed(attributes.max_trace_length);
        let imports = self.make_imports();

        let fn_name = self.get_func_name();
        let preprocess_verifier_fn_name =
            Ident::new(&format!("preprocess_verifier_{fn_name}"), fn_name.span());
        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #preprocess_verifier_fn_name(program: &mut jolt::host::Program)
                -> jolt::JoltVerifierPreprocessing<jolt::F, jolt::PCS>
            {
                #imports

                let (bytecode, memory_init, program_size) = program.decode();
                let memory_config = MemoryConfig {
                    max_input_size: #max_input_size,
                    max_output_size: #max_output_size,
                    max_untrusted_advice_size: #max_untrusted_advice_size,
                    max_trusted_advice_size: #max_trusted_advice_size,
                    stack_size: #stack_size,
                    memory_size: #memory_size,
                    program_size: Some(program_size),
                };
                let memory_layout = MemoryLayout::new(&memory_config);

                // TODO(moodlezoup): Feed in size parameters via macro
                let prover_preprocessing = JoltProverPreprocessing::<jolt::F, jolt::PCS>::gen(
                    bytecode,
                    memory_layout,
                    memory_init,
                    #max_trace_length,
                );
                let preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
                preprocessing
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
            pub fn #preprocess_verifier_fn_name(prover_preprocessing: &jolt::JoltProverPreprocessing<jolt::F, jolt::PCS>)
                -> jolt::JoltVerifierPreprocessing<jolt::F, jolt::PCS>
            {
                #imports
                let preprocessing = JoltVerifierPreprocessing::from(prover_preprocessing);
                preprocessing
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
                    _preprocessing: &jolt::JoltProverPreprocessing<jolt::F, jolt::PCS>,
                ) -> (Option<<jolt::PCS as jolt::CommitmentScheme>::Commitment>,
                      Option<<jolt::PCS as jolt::CommitmentScheme>::OpeningProofHint>)
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
                preprocessing: &jolt::JoltProverPreprocessing<jolt::F, jolt::PCS>,
            ) -> (Option<<jolt::PCS as jolt::CommitmentScheme>::Commitment>,
                  Option<<jolt::PCS as jolt::CommitmentScheme>::OpeningProofHint>)
            {
                #imports
                use jolt::CommitmentScheme;
                use jolt::MultilinearPolynomial;
                use jolt::populate_memory_states;

                let mut trusted_advice_bytes = vec![];
                #(#set_trusted_advice_args;)*

                let max_trusted_advice_size = preprocessing.memory_layout.max_trusted_advice_size;

                let mut trusted_advice_vec = vec![0u64; (max_trusted_advice_size as usize) / 8];

                populate_memory_states(
                    0,
                    &trusted_advice_bytes,
                    Some(&mut trusted_advice_vec),
                    None,
                );

                // Initialize Dory globals with specified parameters
                let _guard = jolt::DoryGlobals::initialize(1, max_trusted_advice_size as usize / 8);

                let poly = MultilinearPolynomial::<jolt::F>::from(trusted_advice_vec);
                let (commitment, hint) = jolt::PCS::commit(&poly, &preprocessing.generators);

                (Some(commitment), Some(hint))
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
                outputs.resize(preprocessing.memory_layout.max_output_size as usize, 0);
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
            quote! { , trusted_advice_commitment: Option<<jolt::PCS as jolt::CommitmentScheme>::Commitment> }
        } else {
            quote! {}
        };

        let commitment_arg = if has_trusted_advice {
            quote! { trusted_advice_commitment }
        } else {
            quote! { None }
        };

        quote! {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "guest")))]
            pub fn #prove_fn_name(
                mut program: jolt::host::Program,
                preprocessing: jolt::JoltProverPreprocessing<jolt::F, jolt::PCS>,
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

                let elf_contents_opt = program.get_elf_contents();
                let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
                let prover = RV64IMACProver::gen_from_elf(&preprocessing,
                    &elf_contents,
                    &input_bytes,
                    &untrusted_advice_bytes,
                    &trusted_advice_bytes,
                    #commitment_arg,
                );
                let io_device = prover.program_io.clone();
                let (jolt_proof, _) = prover.prove();

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
            memory_size: attributes.memory_size,
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

        quote! {
            #[cfg(feature = "guest")]
            use core::arch::global_asm;

            #[cfg(feature = "guest")]
            global_asm!("\
                .global _start\n\
                .extern _STACK_PTR\n\
                .section .text.boot\n\
                _start:	la sp, _STACK_PTR\n\
                    call main\n\
                    j .\n\
            ");

            #declare_alloc

            #[cfg(feature = "guest")]
            #[no_mangle]
            pub extern "C" fn main() {
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
                RV64IMACProver,
                RV64IMACVerifier,
                RV64IMACProof,
                host::Program,
                JoltProverPreprocessing,
                JoltVerifierPreprocessing,
                MemoryConfig,
                MemoryLayout,
                JoltDevice,
            };
        }
    }

    fn make_wasm_utilities(&self) -> TokenStream2 {
        quote! {
            #[cfg(target_arch = "wasm32")]
            use wasm_bindgen::prelude::*;
            #[cfg(target_arch = "wasm32")]
            use std::vec::Vec;
            #[cfg(target_arch = "wasm32")]
            use rmp_serde::Deserializer;
            #[cfg(target_arch = "wasm32")]
            use serde::{Deserialize, Serialize};

            #[cfg(all(target_arch = "wasm32", not(feature = "guest")))]
            use jolt::host::ELFInstruction;

            #[cfg(all(target_arch = "wasm32", not(feature = "guest")))]
            #[derive(Serialize, Deserialize)]
            struct DecodedData {
                bytecode: Vec<ELFInstruction>,
                memory_init: Vec<(u64, u8)>,
            }

            #[cfg(target_arch = "wasm32")]
            fn deserialize_from_bin<'a, T: Deserialize<'a>>(
                data: &'a [u8],
            ) -> Result<T, rmp_serde::decode::Error> {
                let mut de = Deserializer::new(data);
                Deserialize::deserialize(&mut de)
            }
        }
    }

    fn make_set_linker_parameters(&self) -> TokenStream2 {
        let attributes = parse_attributes(&self.attr);
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
                return last_segment.ident == "UntrustedAdvice";
            }
        }
        false
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

    // TODO(moodlezoup): fix this
    fn make_wasm_function(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let verify_wasm_fn_name = Ident::new(&format!("verify_{fn_name}"), fn_name.span());
        let attributes = parse_attributes(&self.attr);
        let max_trace_length = proc_macro2::Literal::u64_unsuffixed(attributes.max_trace_length);

        quote! {
            #[wasm_bindgen]
            #[cfg(all(target_arch = "wasm32", not(feature = "guest")))]
            pub fn #verify_wasm_fn_name(preprocessing_data: &[u8], proof_bytes: &[u8]) -> bool {
                use jolt::{RV64IMACProof, JoltRV64IMAC, Serializable};

                let decoded_preprocessing_data: DecodedData = deserialize_from_bin(preprocessing_data).unwrap();
                let proof = RV64IMACProof::deserialize_from_bytes(proof_bytes).unwrap();

                let preprocessing = JoltRV64IMAC::preprocess(
                    decoded_preprocessing_data.bytecode,
                    decoded_preprocessing_data.memory_init,
                    #max_trace_length,
                );

                let result = JoltRV64IMAC::verify(&preprocessing, proof);
                result.is_ok()
            }
        }
    }
}
