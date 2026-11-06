#[cfg(all(feature = "immutable-recipes", feature = "solinas"))]
#[test]
fn coefficient_import_rejects_modulus_without_reduction() {
    use jolt_field::{Prime128OffsetA7F7 as F, Ring, IMMUTABLE_RECIPE_MODULUS};
    assert_eq!(F::immutable_recipe_import(IMMUTABLE_RECIPE_MODULUS), None);
    assert_eq!(F::immutable_recipe_import(17), Some(F::from_u64(17)));
}
