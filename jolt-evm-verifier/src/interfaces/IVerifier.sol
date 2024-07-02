// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {Jolt} from "../reference/JoltTypes.sol";
import {Fr} from "../reference/Fr.sol";
import {Transcript} from "../subprotocols/FiatShamirTranscript.sol";

interface IVerifier {
    function verifyGrandProduct(
        Jolt.BatchedGrandProductProof memory proof,
        Fr[] memory claims,
        Transcript memory transcript
    ) external returns (Fr[] memory);
}
