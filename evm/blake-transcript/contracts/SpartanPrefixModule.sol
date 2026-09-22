// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;
import {SpartanInputs} from "./SpartanInputs.sol";
import {SpartanAlgebra} from "./SpartanAlgebra.sol";
import {SpartanKzgPairing} from "./SpartanPublicOpening.sol";
import {BlakeHashSponge} from "./BlakeTranscript.sol";

/// @notice Module checkpoint is meaningful only inside the fixed-code coordinator call.
contract SpartanPrefixModule {
    struct Result {
        SpartanAlgebra.Sumcheck outer;
        SpartanAlgebra.Sumcheck inner;
        BlakeHashSponge.State transcript;
        uint256[3] weights;
    }
    function check(SpartanInputs.Decoded memory d,bytes calldata key,bytes calldata setup,bytes calldata proof,uint256[4][2] calldata g2)
        external view returns(Result memory result) {
        SpartanAlgebra.IncompleteCheckpoint memory algebra=SpartanAlgebra.checkIncomplete(d,key,setup,proof);
        SpartanKzgPairing.check(d,algebra.pendingPublicKzg,setup,g2);
        result.outer=algebra.outer;result.inner=algebra.inner;
        result.transcript=d.transcript;result.weights=algebra.weights;
    }
}
