// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {Fr} from "./Fr.sol";

library Jolt {
    struct BatchedGrandProductLayerProof {
        Fr[][] sumcheck_univariate_coeffs;
        Fr[] leftClaims;
        Fr[] rightClaims;
    }

    struct BatchedGrandProductProof {
        BatchedGrandProductLayerProof[] layers;
    }
}
