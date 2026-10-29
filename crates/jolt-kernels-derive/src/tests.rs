#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "tests should fail loudly on unexpected expansions"
)]

use quote::ToTokens;
use syn::{parse_quote, DeriveInput, File, Item, PathSegment};

use crate::expand;

const SLOT_SHAPE_ERROR: &str = "a KernelSlots slot must be exactly `Box<dyn PrepareKernel<F, R>>`";

fn slot_shape_errors(input: DeriveInput) -> usize {
    let error = expand(input).expect_err("expansion should be rejected");
    error
        .into_iter()
        .inspect(|error| assert_eq!(error.to_string(), SLOT_SHAPE_ERROR))
        .count()
}

#[test]
fn rejects_slot_with_extra_object_bound() {
    let errors = slot_shape_errors(parse_quote! {
        struct Registry<F: JoltField> {
            slot: Box<dyn PrepareKernel<F, R> + Send>,
        }
    });
    assert_eq!(errors, 1);
}

#[test]
fn rejects_slot_wrapped_in_option() {
    let errors = slot_shape_errors(parse_quote! {
        struct Registry<F: JoltField> {
            slot: Option<Box<dyn PrepareKernel<F, R>>>,
        }
    });
    assert_eq!(errors, 1);
}

#[test]
fn rejects_slot_nested_in_a_tuple() {
    let errors = slot_shape_errors(parse_quote! {
        struct Registry<F: JoltField> {
            slot: (Box<dyn PrepareKernel<F, R>>,),
        }
    });
    assert_eq!(errors, 1);
}

#[test]
fn reports_every_malformed_slot() {
    let errors = slot_shape_errors(parse_quote! {
        struct Registry<F: JoltField> {
            good: Box<dyn PrepareKernel<F, A>>,
            three: Box<dyn PrepareKernel<F, B, C>>,
            label: &'static str,
            bare: PrepareKernel,
        }
    });
    assert_eq!(errors, 2);
}

#[test]
fn two_argument_slot_resolves_and_bespoke_fields_are_skipped() {
    let tokens = expand(parse_quote! {
        struct Registry<F: JoltField> {
            label: &'static str,
            commit: Box<dyn CommitStream<F>>,
            slot: Box<dyn PrepareKernel<F, R>>,
        }
    })
    .expect("expansion should succeed");
    let file = syn::parse2::<File>(tokens).expect("expansion must parse as items");
    let [Item::Impl(item)] = &file.items[..] else {
        panic!("expected exactly one impl, got {}", file.items.len());
    };
    let (trait_path, _) = item.trait_.as_ref().expect("a trait impl");
    let trait_segment = trait_path.segments.last().expect("a trait path");
    let expected: PathSegment = parse_quote!(PrepareKernel<F, R>);
    assert_eq!(
        trait_segment.to_token_stream().to_string(),
        expected.to_token_stream().to_string()
    );
}
