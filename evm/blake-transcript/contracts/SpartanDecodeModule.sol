// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;
import {SpartanInputs} from "./SpartanInputs.sol";

/// @notice Non-authenticating module diagnostic; policy is owned by the fixed-code coordinator.
contract SpartanDecodeModule {
    function decode(bytes32 keyId,bytes32 setupId,bytes calldata key,bytes calldata setup,bytes calldata inputs,bytes calldata proof)
        external view returns(SpartanInputs.Decoded memory) {
        return SpartanInputs.decodeAuthenticated(keyId,setupId,key,setup,inputs,proof);
    }
}
