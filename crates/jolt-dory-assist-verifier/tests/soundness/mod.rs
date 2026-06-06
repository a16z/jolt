pub mod fixtures;
pub mod tampering;

use crate::support::{assert_case_metadata_matches, assert_unique_case_names};

pub fn assert_registry_is_wired() {
    assert_unique_case_names(tampering::ALL);
    for case in tampering::ALL {
        assert_case_metadata_matches(*case, fixtures::metadata(case.fixture));
    }
}
