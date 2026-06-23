pub mod advice;
pub mod cases;
pub mod fixtures;
pub mod standard;
pub mod zk;

use crate::support::{assert_case_metadata_matches, assert_unique_case_names};

pub fn assert_registry_is_wired() {
    assert_unique_case_names(cases::ALL);
    for case in cases::ALL {
        assert_case_metadata_matches(*case, fixtures::metadata(case.fixture));
    }
}
