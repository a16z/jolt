pragma solidity >=0.8.21;

import {Jolt} from "../reference/JoltTypes.sol";
import {Fr} from "../../src/reference/Fr.sol";

interface IVerifier {
    function verifyGrandProduct(Jolt.BatchedGrandProductProof memory proof, Fr[] memory claims) external view returns (bool verified);
}
