// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {Fr} from "./Fr.sol";

library Jolt {

    struct BatchedGrandProductLayerProof {
        Fr[][] sumcheck_univariate_coeffs;
        Fr[] left_claims;
        Fr[] right_claims;
    }

    struct BatchedGrandProductProof {
        BatchedGrandProductLayerProof[] layers;
    }
}
